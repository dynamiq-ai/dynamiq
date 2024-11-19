import json

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.tools import HttpApiCall, ResponseType
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
        input={"data": {}, "url": "", "headers": {}, "params": {}},
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
