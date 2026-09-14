import json

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import HttpApiCallInputSchema
from dynamiq.runnables import RunnableResult, RunnableStatus


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


def test_an_agent_can_read_a_json_body_that_arrived_as_bytes(requests_mock):
    """`ResponseType.RAW` is the default and the auto-JSON override needs an exact
    `application/json` header, so an ordinary charset-tagged body reaches the agent as bytes."""
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
        headers={"content-type": "application/json; charset=utf-8"},
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
