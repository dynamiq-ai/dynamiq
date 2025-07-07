import json
from io import BytesIO
from urllib.parse import urljoin

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio import WhisperSTT
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.mark.parametrize("audio", [b"bytes_data", BytesIO(b"bytes_data"), b"\xff\xfb\x90\xc4\x00\x00\n\xddu"])
def test_workflow_with_whisper_transcriber(mock_audio_transcribing_response_text, requests_mock, audio):
    model = "whisper-1"
    connection = connections.Whisper(
        url="https://your-url/",
        api_key="api-key",
    )
    connection_url = urljoin(connection.url, "audio/transcriptions")
    call_mock = requests_mock.post(url=connection_url, text=json.dumps({"text": mock_audio_transcribing_response_text}))

    wf_whisper = Workflow(
        flow=Flow(
            nodes=[WhisperSTT(name="whisper", model=model, connection=connection)]
        ),
    )
    input_data = {"audio": audio}
    response = wf_whisper.run(input_data=input_data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"content": mock_audio_transcribing_response_text},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_whisper.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    assert call_mock.called_once
    assert call_mock.last_request.url == connection_url
    assert (
        call_mock.last_request.headers.get("Authorization")
        == f"Bearer {connection.api_key}"
    )
