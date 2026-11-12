import json

import pytest

from dynamiq import Workflow, connections
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import HttpApiCallInputSchema
from dynamiq.runnables import RunnableResult, RunnableStatus
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
