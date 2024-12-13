import json

import pytest
from pydantic import ConfigDict, ValidationError

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.connections import Exa
from dynamiq.flows import Flow
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder


@pytest.fixture
def mock_exa_response(mocker):
    """Mock response from Exa API."""
    return {
        "results": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1",
                "publishedDate": "2024-01-01",
                "author": "John Doe",
                "score": 0.95,
                "highlights": ["This is a highlight from article 1"],
                "summary": "Summary of article 1",
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/article2",
                "publishedDate": "2024-01-02",
                "author": "Jane Smith",
                "score": 0.85,
                "highlights": ["This is a highlight from article 2"],
                "summary": "Summary of article 2",
            },
        ],
        "autopromptString": "Enhanced search query",
    }


@pytest.fixture
def mock_requests(mocker, mock_exa_response):
    """Mock requests library."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_exa_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_exa_basic_search(mock_requests, mock_exa_response):
    """Test basic search functionality without content retrieval."""
    exa_connection = Exa(api_key="test_key")
    exa_tool = ExaTool(connection=exa_connection, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence", "limit": 2, "query_type": "neural", "include_full_content": False}

    result = exa_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]
    assert input_dump["limit"] == input_data["limit"]
    assert input_dump["query_type"] == input_data["query_type"]
    assert input_dump["include_full_content"] == input_data["include_full_content"]

    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["query"] == "artificial intelligence"
    assert call_args[1]["json"]["numResults"] == 2
    assert "contents" not in call_args[1]["json"]


def test_exa_search_agent_optimized(mock_requests, mock_exa_response):
    """Test search with agent-optimized output format."""
    exa_connection = Exa(api_key="test_key")
    exa_tool = ExaTool(connection=exa_connection, is_optimized_for_agents=True, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence", "include_full_content": True}

    result = exa_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "<Sources with URLs>" in content
    assert "<Search results>" in content
    assert all(f"{r['title']}: ({r['url']})" in content for r in mock_exa_response["results"])

    for result in mock_exa_response["results"]:
        if "highlights" in result:
            assert any(highlight in content for highlight in result["highlights"])
        if "summary" in result:
            assert result["summary"] in content


def test_exa_with_invalid_input_schema(mock_requests, mock_exa_response):
    exa_connection = Exa(api_key="test_key")
    exa_tool = ExaTool(connection=exa_connection, model_config=ConfigDict())

    wf = Workflow(flow=Flow(nodes=[exa_tool]))
    input_data = {}
    tracing = TracingCallbackHandler()
    result = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    result_exa = result.output[exa_tool.id]
    assert result.status == RunnableStatus.SUCCESS
    assert result.input == input_data
    assert result_exa["status"] == RunnableStatus.FAILURE.value
    assert result_exa["input"] == input_data
    assert result_exa["output"]["error_type"] == ValidationError.__name__

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 3
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.output
    assert wf_run.status == RunStatus.SUCCEEDED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output
    assert flow_run.status == RunStatus.SUCCEEDED
    exa_tool_run = tracing_runs[2]
    assert exa_tool_run.metadata["node"]["id"] == exa_tool.id
    assert exa_tool_run.parent_run_id == flow_run.id
    assert exa_tool_run.input == input_data
    assert exa_tool_run.output is None
    assert exa_tool_run.error
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)
