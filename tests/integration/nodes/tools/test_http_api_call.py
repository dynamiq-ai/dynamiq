import asyncio
import codecs
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from requests_mock import ANY

from dynamiq import Workflow, connections
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import HttpApiCallInputSchema
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader


@pytest.mark.parametrize(
    ("response_type", "result"),
    [
        (ResponseType.TEXT, '{"a": 1}'),
        (ResponseType.RAW, b'{"a": "1"}'),
        (ResponseType.JSON, {"a": "1"}),
    ],
)
def test_workflow_with_httpapicall(
    mock_whisper_response_text, requests_mock, response_type, result
):
    url = "https://api.elevenlabs.io/v1/shared-voices"
    connection = connections.Http(
        method=connections.HTTPMethod.GET,
        url=url,
        headers={"xi-api-key": "api-key"},
    )
    wf_httpapicall = Workflow(
        flow=Flow(
            nodes=[
                HttpApiCall(
                    connection=connection,
                    success_codes=[200, 201, 202],
                    timeout=5,
                    response_type=response_type,
                )
            ]
        ),
    )

    if response_type == ResponseType.RAW:
        call_mock = requests_mock.get(url=url, content=result)
    elif response_type == ResponseType.JSON:
        call_mock = requests_mock.get(url=url, text=json.dumps(result))
    else:
        call_mock = requests_mock.get(url=url, text=result)
    response = wf_httpapicall.run(input_data={})

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=dict(HttpApiCallInputSchema(**{})),
        output={"content": result, "status_code": 200},
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_httpapicall.flow.nodes[0].id: expected_result}
    expected_headers = connection.headers | {"xi-api-key": "api-key"}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )

    assert call_mock.called_once
    assert call_mock.last_request.url == url
    for header, value in expected_headers.items():
        assert call_mock.last_request.headers.get(header) == value


@pytest.mark.parametrize(
    ("status_code", "headers", "requests_sent"),
    [
        (201, {}, 1),
        (402, {}, 1),
        (422, {}, 1),
        (402, {"Retry-After": "60"}, 1),
        (404, {}, 3),
        (409, {}, 3),
        (429, {}, 3),
        (503, {}, 3),
    ],
)
def test_a_request_is_sent_again_only_where_a_later_attempt_can_pass(
    requests_mock, status_code, headers, requests_sent
):
    url = "https://api.example.com/payments"
    payment = requests_mock.post(url, status_code=status_code, headers=headers, json={})
    node = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.POST, url=url),
        error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01),
    )

    result = node.run(input_data={})

    assert result.status == RunnableStatus.FAILURE
    assert payment.call_count == requests_sent


@pytest.mark.parametrize("content_type", ["application/json", "application/json; charset=utf-8"])
def test_a_json_answer_is_read_as_json_also_after_a_yaml_round_trip(requests_mock, tmp_path, content_type):
    url = "https://api.example.com/score"
    requests_mock.get(url, content=b'{"score": 742}', headers={"content-type": content_type})
    workflow = Workflow(flow=Flow(nodes=[HttpApiCall(id="bureau", connection=connections.Http(method="GET", url=url))]))
    yaml_file = tmp_path / "bureau.yaml"
    workflow.to_yaml_file(yaml_file)
    with get_connection_manager() as cm:
        loaded = Workflow.from_yaml_file_data(
            file_data=WorkflowYAMLLoader.load(file_path=yaml_file, connection_manager=cm, init_components=True)
        )

    for run in (workflow, loaded):
        assert run.run_sync(input_data={}).output["bureau"]["output"]["content"] == {"score": 742}


@pytest.mark.parametrize(("response_type", "dumped"), [(None, False), (ResponseType.TEXT, True)])
def test_a_dump_names_the_response_type_only_when_one_is_set(tmp_path, response_type, dumped):
    """dynamiq 0.73.0 and earlier fail every request of a node loaded with a null response_type."""
    node = HttpApiCall(
        id="bureau",
        connection=connections.Http(method="GET", url="https://api.example.com/score"),
        response_type=response_type,
    )
    yaml_file = tmp_path / "bureau.yaml"
    Workflow(flow=Flow(nodes=[node])).to_yaml_file(yaml_file)

    assert ("response_type" in yaml_file.read_text()) is dumped


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_both_clients_send_the_query_string_of_the_url_with_the_params(requests_mock, mode):
    url = "https://screening.example.com/v1/screen?min_score=0.75"
    node = HttpApiCall(
        connection=connections.Http(method="GET", url=url),
        params={"name": "Ana", "middle_name": None, "is_joint": False},
    )
    sent = "https://screening.example.com/v1/screen?min_score=0.75&name=Ana&is_joint=false"

    if mode == "sync":
        requests_mock.get(url, json={"matches": []})
        result = node.run(input_data={})
        assert requests_mock.last_request.url == sent
    else:
        requests = []

        def answer(request):
            requests.append(str(request.url))
            return httpx.Response(200, json={"matches": []})

        async def run():
            async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
                with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=client)):
                    return await node.run_async(input_data={})

        result = asyncio.run(run())
        assert requests == [sent]

    assert result.status == RunnableStatus.SUCCESS


def _run(node, mode, requests_mock, body=b"{}", content_type="application/json", input_data=None, config=None):
    """Run the node on the client the mode names, every request answered 200; return the result and the URLs sent."""
    if mode == "sync":
        requests_mock.register_uri(ANY, ANY, content=body, headers={"content-type": content_type})
        result = node.run(input_data=input_data or {}, config=config)
        return result, [request.url for request in requests_mock.request_history]

    sent = []

    def answer(request):
        sent.append(str(request.url))
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
            with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=client)):
                return await node.run_async(input_data=input_data or {}, config=config)

    return asyncio.run(run()), sent


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_both_clients_write_the_booleans_of_a_list_param_as_true_and_false(requests_mock, mode):
    node = HttpApiCall(
        connection=connections.Http(method="GET", url="https://screening.example.com/v1/screen"),
        params={"is_pep": [True, False]},
    )

    result, sent = _run(node, mode, requests_mock)

    assert result.status == RunnableStatus.SUCCESS
    assert sent == ["https://screening.example.com/v1/screen?is_pep=true&is_pep=false"]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("url", ["api.example.com/users/7", "/users/7"])
def test_an_invalid_url_fails_once_and_recoverably(requests_mock, mode, url):
    """An agent that wrote the URL gets an observation it can act on, where a raw error would end its run."""
    errors = []

    class Attempts(BaseCallbackHandler):
        def on_node_execute_error(self, serialized, error, **kwargs):
            errors.append(error)

    node = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.GET, url="https://api.example.com"),
        error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01),
    )
    config = RunnableConfig(callbacks=[Attempts()])

    result, sent = _run(node, mode, requests_mock, input_data={"url": url}, config=config)

    assert result.status == RunnableStatus.FAILURE
    assert result.error.type is ToolExecutionException
    assert result.error.recoverable
    assert len(errors) == 1
    assert sent == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("content_type", "body", "content"),
    [
        pytest.param("application/json; charset=utf-8", b"", b"", id="empty"),
        pytest.param("application/json", b"<html>Bad gateway</html>", b"<html>Bad gateway</html>", id="error-page"),
        pytest.param(
            "application/json; charset=windows-1251",
            '{"holder": "Олена Петренко"}'.encode("cp1251"),
            {"holder": "Олена Петренко"},
            id="declared-charset",
        ),
        pytest.param("application/json", codecs.BOM_UTF8 + b'{"score": 742}', {"score": 742}, id="byte-order-mark"),
    ],
)
def test_an_answer_labelled_json_is_read_with_its_charset_or_kept_as_bytes(
    requests_mock, mode, content_type, body, content
):
    """Middleware can label every answer as JSON, so a body that does not parse is no reason to send a payment again."""
    node = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.POST, url="https://api.example.com/payments"),
        error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01),
    )

    result, sent = _run(node, mode, requests_mock, body=body, content_type=content_type)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == content
    assert len(sent) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_an_answer_that_is_not_the_json_asked_for_fails_recoverably_and_is_not_sent_again(requests_mock, mode):
    """The server took the request, so sending it again could pay twice."""
    node = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.POST, url="https://api.example.com/payments"),
        response_type=ResponseType.JSON,
        error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01),
    )

    result, sent = _run(node, mode, requests_mock, body=b"", content_type="application/json")

    assert result.status == RunnableStatus.FAILURE
    assert result.error.type is ToolExecutionException
    assert result.error.recoverable
    assert len(sent) == 1


def test_an_agent_can_read_a_json_body_that_arrived_as_bytes(requests_mock):
    """A body the server does not mark as JSON reaches the agent as bytes."""
    from dynamiq.connections import OpenAI as OpenAIConnection
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI

    url = "https://api.example.com/answer"
    tool = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.GET, url=url),
        name="api-call",
    )
    requests_mock.get(
        url,
        content=b'{"answer": 42, "detail": "what the agent needed"}',
        headers={"content-type": "text/plain; charset=utf-8"},
    )
    agent = Agent(
        name="Agent",
        llm=OpenAI(connection=OpenAIConnection(api_key="k"), model="gpt-4o"),
        role="r",
        tools=[tool],
    )

    observation, _, _ = agent._run_tool(tool, {"data": {}}, config=None)

    assert "what the agent needed" in observation
    assert "bytes of binary data" not in observation


@pytest.mark.parametrize(
    "body",
    [{"holder": "Олена Петренко"}, {"content": [{"holder": "Олена Петренко"}], "totalElements": 1}],
    ids=["object", "page"],
)
def test_an_agent_reads_a_json_answer_in_cyrillic_as_text(requests_mock, body):
    """Escaped, each letter would take six characters of an observation that is cut at a fixed length."""
    url = "https://api.example.com/accounts/7"
    tool = HttpApiCall(connection=connections.Http(method=connections.HTTPMethod.GET, url=url), name="api-call")
    requests_mock.get(
        url,
        content=json.dumps(body, ensure_ascii=False).encode(),
        headers={"content-type": "application/json; charset=utf-8"},
    )
    agent = Agent(
        name="Agent",
        llm=OpenAI(connection=connections.OpenAI(api_key="k"), model="gpt-4o"),
        role="r",
        tools=[tool],
    )

    observation, _, _ = agent._run_tool(tool, {"data": {}}, config=None)

    assert "Олена Петренко" in observation
