import pytest
from dynamiq import Workflow
from dynamiq.connections import HuggingFace
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableResult, RunnableStatus

from dynamiq.connections import Lakera
from dynamiq.nodes.detectors import PIIDetector


@pytest.mark.parametrize(
    ("connection", "url", "mock_request_data"),
    [
        (
            HuggingFace(),
            "https://api-inference.huggingface.co/models/iiiorg/piiranha-v1-detect-personal-information",
            [
                {"entity_group": "pii/us_social_security_number"},
                {"entity_group": "prompt_attack"},
                {"entity_group": "unknown_links"},
            ],
        ),
        (
            Lakera(),
            "https://api.lakera.ai/v2/guard",
            {
                "breakdown": [
                    {"detector_type": "pii/us_social_security_number", "detected": True},
                    {"detector_type": "prompt_attack", "detected": True},
                    {"detector_type": "unknown_links", "detected": True},
                ]
            },
        ),
    ],
)
def test_workflow_with_piidetector(requests_mock, connection, url, mock_request_data):
    wf_piidetector = Workflow(
        flow=Flow(
            nodes=[
                PIIDetector(
                    connection=connection,
                    timeout=5,
                )
            ]
        ),
    )

    input_data = {"message": "123 Main Street, Springfield, IL 62704. My credit card number is 4111 1111 1111 1111."}
    output = ["pii/us_social_security_number", "prompt_attack", "unknown_links"]
    call_mock = requests_mock.post(url=url, json=mock_request_data)
    response = wf_piidetector.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"is_detected": True, "detected_pii": output},
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_piidetector.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )

    assert call_mock.called_once
    assert call_mock.last_request.url == url
