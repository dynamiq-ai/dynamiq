import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import requests
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Context
from dynamiq.flows import Flow
from dynamiq.nodes.tools import ContextSearchTool
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder


@pytest.fixture
def context_response():
    return {
        "results": [
            {
                "title": "Python docs",
                "url": "https://docs.python.org/3/",
                "description": "Python documentation",
                "highlights": {"code": "SUCCESS", "highlights": ["The Python standard library"]},
            }
        ],
        "partial": True,
        "key_metadata": {"credits_remaining": 100},
    }


def test_context_workflow_roundtrip_and_tracing(tmp_path, mocker, context_response):
    request = mocker.patch("requests.request", return_value=Mock(status_code=200))
    request.return_value.json.return_value = context_response
    tool = ContextSearchTool(connection=Context(api_key="synthetic-test-key"), max_results=3)
    workflow = Workflow(flow=Flow(nodes=[tool]))
    path = tmp_path / "workflow.yaml"
    workflow.to_yaml_file(path)
    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    tracing = TracingCallbackHandler()
    result = loaded.run({"query": "Python standard library", "max_results": 2}, RunnableConfig(callbacks=[tracing]))
    assert result.status == RunnableStatus.SUCCESS
    output = result.output[tool.id]["output"]["content"]
    assert output["urls"] == ["https://docs.python.org/3/"]
    assert output["partial"] is True
    assert "Python standard library" in output["result"]
    assert output["raw_response"] == context_response
    assert request.call_args.kwargs["json"] == {
        "query": "Python standard library",
        "numResults": 2,
        "highlightsOptions": {"enabled": True},
    }
    assert request.call_args.kwargs["timeout"] == 60
    assert request.call_args.kwargs["allow_redirects"] is False
    assert "synthetic-test-key" not in json.dumps(
        [run.to_dict() for run in tracing.runs.values()],
        cls=JsonWorkflowEncoder,
    )


@pytest.mark.asyncio
async def test_context_native_async_agent_output(mocker, context_response):
    tool = ContextSearchTool(connection=Context(api_key="test-key"), is_optimized_for_agents=True)
    client = Mock()
    client.request = AsyncMock(
        return_value=httpx.Response(
            200, json=context_response, request=httpx.Request("POST", "https://api.context.dev/v1/web/search")
        )
    )
    mocker.patch.object(ContextSearchTool, "get_async_client", new=AsyncMock(return_value=client))
    result = await tool.run_async({"query": "Python documentation"})
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["urls"] == ["https://docs.python.org/3/"]
    assert result.output["partial"] is True
    assert "The Python standard library" in result.output["content"]
    assert "credits_remaining" not in result.output["content"]
    assert client.request.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize(
    "input_data",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "q" * 501},
        {"query": "q", "max_results": 0},
        {"query": "q", "max_results": 101},
    ],
)
def test_context_invalid_input_does_not_dispatch(mocker, input_data):
    request = mocker.patch("requests.request", return_value=Mock(status_code=200))
    result = ContextSearchTool(connection=Context(api_key="test-key")).run(input_data)
    assert result.status == RunnableStatus.FAILURE
    request.assert_not_called()


@pytest.mark.parametrize(
    "payload", [{"error": "quota exhausted"}, {"results": None}, {"results": [{"title": "Missing URL"}]}]
)
def test_context_invalid_response_is_failure(mocker, payload):
    request = mocker.patch("requests.request", return_value=Mock(status_code=200))
    request.return_value.json.return_value = payload
    result = ContextSearchTool(connection=Context(api_key="test-key")).run({"query": "q"})
    assert result.status == RunnableStatus.FAILURE
    assert result.error.type.__name__ == "ToolExecutionException"


@pytest.mark.parametrize(
    "response",
    [
        {"results": []},
        {
            "results": [
                {
                    "url": "https://example.com",
                    "title": None,
                    "description": "Snippet",
                    "highlights": {"code": "TIMEOUT", "highlights": None},
                }
            ]
        },
    ],
)
def test_context_empty_and_unavailable_highlights_are_success(mocker, response):
    request = mocker.patch("requests.request", return_value=Mock(status_code=200))
    request.return_value.json.return_value = response
    result = ContextSearchTool(connection=Context(api_key="test-key")).run({"query": "q"})
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"]["raw_response"] == response
    assert result.output["content"]["result"]


@pytest.mark.parametrize("error", [requests.Timeout("timeout"), requests.HTTPError("429 Too Many Requests")])
def test_context_transport_failure_is_recoverable(mocker, error):
    mocker.patch("requests.request", side_effect=error)
    result = ContextSearchTool(connection=Context(api_key="test-key")).run({"query": "q"})
    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True


def test_context_missing_key(monkeypatch):
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)
    with pytest.raises((ValueError, ValidationError)):
        Context()
