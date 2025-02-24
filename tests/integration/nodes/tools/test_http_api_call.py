import json
from unittest.mock import MagicMock

import pytest

from dynamiq import Workflow, connections
from dynamiq.connections import Http as HttpConnection
from dynamiq.connections import HTTPMethod
from dynamiq.flows import Flow
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import HttpApiCallInputSchema, RequestPayloadType
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
        input={},
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


@pytest.fixture
def mock_client():
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    client.request.return_value = mock_response
    return client


def test_http_api_call_with_merged_params(mock_client):
    get_api_connection = HttpConnection(
        method=HTTPMethod.GET,
        url="url",
    )

    additional_params = HttpApiCallInputSchema(
        data={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"extra": "extra_value"}}},
            "numbers_list": [3, 4],
            "status": "active",
            "description": "contains value",
        },
        params={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"extra": "extra_value"}}},
            "numbers_list": [3, 4],
            "status": "active",
            "description": "contains value",
        },
        headers={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"extra": "extra_value"}}},
            "numbers_list": [3, 4],
            "status": "active",
            "description": "contains value",
        },
        url="https://example.com/api",
        payload_type=RequestPayloadType.JSON,
    )

    api_tool = HttpApiCall(
        connection=get_api_connection,
        additional_input_data=additional_params,
        client=mock_client,
    )

    input_params = HttpApiCallInputSchema(
        data={
            "nested_data": {"outer": {"middle": {"inner": "value"}}},
            "numbers_list": [1, 2],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        params={
            "nested_data": {"outer": {"middle": {"inner": "value"}}},
            "numbers_list": [1, 2],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        headers={
            "nested_data": {"outer": {"middle": {"inner": "value"}}},
            "numbers_list": [1, 2],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        payload_type=RequestPayloadType.RAW,
    )

    api_tool.execute(input_data=input_params)

    result_params = HttpApiCallInputSchema(
        data={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"inner": "value", "extra": "extra_value"}}},
            "numbers_list": [1, 2, 3, 4],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        params={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"inner": "value", "extra": "extra_value"}}},
            "numbers_list": [1, 2, 3, 4],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        headers={
            "user_id": "2",
            "nested_data": {"outer": {"middle": {"inner": "value", "extra": "extra_value"}}},
            "numbers_list": [1, 2, 3, 4],
            "api_key": "custom_key",
            "status": "inactive",
            "description": "",
        },
        url="https://example.com/api",
        payload_type=RequestPayloadType.RAW,
    )

    mock_client.request.assert_called_once_with(
        method=get_api_connection.method,
        url=result_params.url,
        headers=result_params.headers,
        params=result_params.params,
        timeout=api_tool.timeout,
        **(
            {"data": result_params.data}
            if result_params.payload_type == RequestPayloadType.RAW
            else {"json": result_params.data}
        )
    )
