from io import BytesIO

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio import ElevenLabsSTS, ElevenLabsTTS, Voices
from dynamiq.runnables import RunnableResult, RunnableStatus


def test_workflow_with_elevenlabstts(mock_elevenlabs_response_text, requests_mock):
    connection = connections.ElevenLabs(
        url="https://api.elevenlabs.io/v1/text-to-speech/",
        api_key="api-key",
        headers={"Accept": "audio/mpeg", "Content-Type": "application/json"},
    )
    wf_elevenlabs = Workflow(
        flow=Flow(
            nodes=[
                ElevenLabsTTS(
                    name="elevenlabs",
                    connection=connection,
                    voice_id=Voices.Dave,
                    model="eleven_monolingual_v1",
                    stability=0.5,
                    similarity_boost=0.5,
                )
            ]
        ),
    )

    data = {"text": "Mock text"}
    call_mock = requests_mock.post(
        url=connection.url + Voices.Dave, content=mock_elevenlabs_response_text
    )
    response = wf_elevenlabs.run(input_data=data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=data,
        output={"content": mock_elevenlabs_response_text},
    ).to_dict(skip_format_types={bytes})

    expected_output = {wf_elevenlabs.flow.nodes[0].id: expected_result}
    expected_headers = connection.headers | {"xi-api-key": connection.api_key}
    expected_data = connection.data | {
        "model_id": "eleven_monolingual_v1",
        "text": data["text"],
        "voice_settings": {
            "similarity_boost": 0.5,
            "stability": 0.5,
            "style": 0,
            "use_speaker_boost": True,
        },
    }
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=data,
        output=expected_output,
    )

    assert call_mock.call_count == 1
    assert call_mock.last_request.url == connection.url + Voices.Dave
    assert call_mock.last_request.json() == expected_data
    for header, value in expected_headers.items():
        assert call_mock.last_request.headers.get(header) == value


@pytest.mark.parametrize("audio", [b"bytes_data", BytesIO(b"bytes_data"), b"\xff\xfb\x90\xc4\x00\x00\n\xddu"])
def test_workflow_with_elevenlabssts(
    mock_elevenlabs_response_text, requests_mock, audio
):
    connection = connections.ElevenLabs(
        url="https://api.elevenlabs.io/v1/speech-to-speech/",
        api_key="api-key",
        headers={"Accept": "audio/mpeg", "Content-Type": "application/json"},
    )
    wf_elevenlabs = Workflow(
        flow=Flow(
            nodes=[
                ElevenLabsSTS(
                    name="elevenlabs",
                    connection=connection,
                    voice_id=Voices.Dave,
                    stability=0.5,
                    similarity_boost=0.5,
                )
            ]
        ),
    )

    data = {"audio": audio}
    call_mock = requests_mock.post(
        url=connection.url + Voices.Dave, content=mock_elevenlabs_response_text
    )
    response = wf_elevenlabs.run(input_data=data)

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=data,
        output={"content": mock_elevenlabs_response_text},
    ).to_dict(skip_format_types={BytesIO, bytes})

    expected_output = {wf_elevenlabs.flow.nodes[0].id: expected_result}
    expected_headers = connection.headers | {"xi-api-key": connection.api_key}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=data,
        output=expected_output,
    )

    assert call_mock.called_once
    assert call_mock.last_request.url == connection.url + Voices.Dave
    for header, value in expected_headers.items():
        assert call_mock.last_request.headers.get(header) == value
