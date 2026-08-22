from io import BytesIO

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio import MiniMaxAudioFormat, MiniMaxOutputFormat, MiniMaxSpeechModel, MiniMaxTTS
from dynamiq.nodes.audio.minimax import MiniMaxTTSInputSchema
from dynamiq.nodes.exceptions import NodeFailedException
from dynamiq.runnables import RunnableStatus


def test_minimax_tts_decodes_hex_audio_and_sends_supported_fields(requests_mock) -> None:
    audio = b"generated audio"
    connection = connections.MiniMax(api_key="api-key")
    node = MiniMaxTTS(
        connection=connection,
        voice_setting={"voice_id": "English_expressive_narrator"},
        pronunciation_dict={"tone": ["Dynamiq/dynamic"]},
        audio_setting={"format": "mp3", "sample_rate": 32000, "bitrate": 128000, "channel": 1},
        voice_modify={"pitch": 1},
        language_boost="auto",
        subtitle_enable=True,
    )
    call = requests_mock.post(
        connection.url,
        json={
            "data": {"audio": audio.hex(), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )

    workflow = Workflow(flow=Flow(nodes=[node]))
    result = workflow.run(input_data={"text": "Hello", "output_file_name": "speech.mp3"})

    assert result.status == RunnableStatus.SUCCESS
    output = result.output[node.id]["output"]
    assert output["content"] == audio
    assert isinstance(output["files"][0], BytesIO)
    assert output["files"][0].name == "speech.mp3"
    assert output["files"][0].content_type == "audio/mpeg"
    assert call.last_request.headers["Authorization"] == "Bearer api-key"
    assert call.last_request.json() == {
        "model": "speech-2.8-hd",
        "text": "Hello",
        "stream": False,
        "language_boost": "auto",
        "output_format": "hex",
        "voice_setting": {"voice_id": "English_expressive_narrator"},
        "pronunciation_dict": {"tone": ["Dynamiq/dynamic"]},
        "audio_setting": {"format": "mp3", "sample_rate": 32000, "bitrate": 128000, "channel": 1},
        "voice_modify": {"pitch": 1},
        "subtitle_enable": True,
    }


def test_minimax_tts_uses_china_endpoint_and_downloads_url_audio(requests_mock) -> None:
    audio = b"wave audio"
    connection = connections.MiniMax(region=connections.MiniMaxRegion.CHINA, api_key="api-key")
    node = MiniMaxTTS(
        connection=connection,
        output_format=MiniMaxOutputFormat.URL,
        audio_setting={"format": MiniMaxAudioFormat.WAV.value},
    )
    audio_url = "https://audio.example.test/generated.wav"
    call = requests_mock.post(
        "https://api.minimaxi.com/v1/t2a_v2",
        json={
            "data": {"audio": audio_url, "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        },
    )
    download = requests_mock.get(audio_url, content=audio)

    result = Workflow(flow=Flow(nodes=[node])).run(input_data={"text": "Hello"})

    output = result.output[node.id]["output"]
    assert call.called_once
    assert download.called_once
    assert output["content"] == audio
    assert output["files"][0].name == "audio.wav"
    assert output["files"][0].content_type == "audio/wav"


def test_minimax_tts_raises_for_api_failure(requests_mock) -> None:
    connection = connections.MiniMax(api_key="api-key")
    node = MiniMaxTTS(connection=connection)
    requests_mock.post(
        connection.url,
        json={
            "data": {"audio": "", "status": 1},
            "base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
        },
    )
    node.init_components()

    with pytest.raises(NodeFailedException, match="insufficient balance"):
        node.execute(MiniMaxTTSInputSchema(text="Hello"))


def test_minimax_tts_exposes_current_speech_models() -> None:
    assert {model.value for model in MiniMaxSpeechModel} == {
        "speech-2.8-hd",
        "speech-2.8-turbo",
        "speech-2.6-hd",
        "speech-2.6-turbo",
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo",
    }


def test_minimax_tts_yaml_round_trip(tmp_path) -> None:
    node = MiniMaxTTS(
        connection=connections.MiniMax(region=connections.MiniMaxRegion.CHINA, api_key="api-key"),
        model=MiniMaxSpeechModel.SPEECH_2_8_TURBO,
        audio_setting={"format": MiniMaxAudioFormat.FLAC.value},
    )
    workflow = Workflow(flow=Flow(nodes=[node]))
    yaml_path = tmp_path / "minimax-tts.yaml"

    workflow.to_yaml_file(str(yaml_path))
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=False).flow.nodes[0]

    assert isinstance(loaded, MiniMaxTTS)
    assert loaded.model == MiniMaxSpeechModel.SPEECH_2_8_TURBO
    assert loaded.audio_setting == {"format": "flac"}
    assert loaded.connection.region == connections.MiniMaxRegion.CHINA
