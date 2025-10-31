from io import BytesIO

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio import ElevenLabsSTS, ElevenLabsTTS, Voices
from dynamiq.runnables import RunnableStatus


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

    assert response.status == RunnableStatus.SUCCESS
    assert response.input == data

    node_output = response.output[wf_elevenlabs.flow.nodes[0].id]
    assert node_output["status"] == RunnableStatus.SUCCESS.value
    assert node_output["input"] == data

    assert node_output["output"]["content"] == mock_elevenlabs_response_text

    assert "files" in node_output["output"]
    assert isinstance(node_output["output"]["files"], list)
    assert len(node_output["output"]["files"]) == 1

    audio_file = node_output["output"]["files"][0]
    assert isinstance(audio_file, BytesIO)
    assert audio_file.name == "audio.mp3"
    assert audio_file.content_type == "audio/mpeg"
    assert audio_file.read() == mock_elevenlabs_response_text

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

    assert response.status == RunnableStatus.SUCCESS
    assert response.input == data

    node_output = response.output[wf_elevenlabs.flow.nodes[0].id]
    assert node_output["status"] == RunnableStatus.SUCCESS.value
    assert node_output["input"] == data

    assert node_output["output"]["content"] == mock_elevenlabs_response_text

    assert "files" in node_output["output"]
    assert isinstance(node_output["output"]["files"], list)
    assert len(node_output["output"]["files"]) == 1

    audio_file = node_output["output"]["files"][0]
    assert isinstance(audio_file, BytesIO)
    assert audio_file.name == "audio.mp3"
    assert audio_file.content_type == "audio/mpeg"
    assert audio_file.read() == mock_elevenlabs_response_text

    expected_headers = connection.headers | {"xi-api-key": connection.api_key}

    assert call_mock.called_once
    assert call_mock.last_request.url == connection.url + Voices.Dave
    for header, value in expected_headers.items():
        assert call_mock.last_request.headers.get(header) == value
