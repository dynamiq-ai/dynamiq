from io import BytesIO

import pytest

from dynamiq import Workflow, connections
from dynamiq.components.audio.registry import resolve_tts_adapter
from dynamiq.components.audio.tts import MiniMaxTTSAdapter
from dynamiq.flows import Flow
from dynamiq.nodes.audio import TextToSpeech
from dynamiq.nodes.audio.text_to_speech import TextToSpeechInputSchema
from dynamiq.runnables import RunnableStatus
from dynamiq.types.audio import AudioFormat

GLOBAL_URL = "https://api.minimax.io/v1/t2a_v2"
CHINA_URL = "https://api.minimax.cn/v1/t2a_v2"


def success(audio: str) -> dict:
    return {
        "data": {"audio": audio, "status": 2},
        "extra_info": {"audio_format": "mp3"},
        "trace_id": "trace",
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }


def run_node(node, input_data):
    result = Workflow(flow=Flow(nodes=[node])).run(input_data=input_data)
    assert result.status == RunnableStatus.SUCCESS, result.output[node.id]
    return result.output[node.id]["output"]


def test_minimax_connection_selects_the_adapter():
    assert resolve_tts_adapter(connections.MiniMax(api_key="k")) is MiniMaxTTSAdapter


def test_minimax_connection_regions():
    assert connections.MiniMax(api_key="k").url == "https://api.minimax.io/v1"
    china = connections.MiniMax(region=connections.MiniMaxRegion.CHINA, api_key="k")
    assert china.url == "https://api.minimax.cn/v1"
    assert china.headers["Authorization"] == "Bearer k"
    custom = connections.MiniMax(url="https://proxy.example.test/v1", api_key="k")
    assert custom.url == "https://proxy.example.test/v1"


def test_minimax_text_to_speech_decodes_hex_audio(requests_mock):
    audio = b"generated audio"
    node = TextToSpeech(
        connection=connections.MiniMax(api_key="api-key"),
        voice="English_Graceful_Lady",
        speed=1.2,
        sample_rate=32000,
        provider_options={
            "voice_setting": {"emotion": "happy"},
            "audio_setting": {"bitrate": 128000, "channel": 1},
            "language_boost": "English",
            "pronunciation_dict": {"tone": ["Dynamiq/dynamic"]},
        },
    )
    call = requests_mock.post(GLOBAL_URL, json=success(audio.hex()))

    output = run_node(node, {"text": "Hello", "output_file_name": "speech.mp3"})

    assert node.model == "speech-2.8-hd"
    assert output["content"] == audio
    assert output["mime_type"] == "audio/mpeg"
    audio_file = output["files"][0]
    assert isinstance(audio_file, BytesIO)
    assert audio_file.name == "speech.mp3"
    assert audio_file.content_type == "audio/mpeg"

    assert call.last_request.headers["Authorization"] == "Bearer api-key"
    assert call.last_request.json() == {
        "model": "speech-2.8-hd",
        "text": "Hello",
        "stream": False,
        "voice_setting": {"emotion": "happy", "voice_id": "English_Graceful_Lady", "speed": 1.2},
        "audio_setting": {"bitrate": 128000, "channel": 1, "format": "mp3", "sample_rate": 32000},
        "language_boost": "English",
        "pronunciation_dict": {"tone": ["Dynamiq/dynamic"]},
    }


def test_minimax_falls_back_to_a_system_voice(requests_mock):
    node = TextToSpeech(connection=connections.MiniMax(api_key="k"), output_format=AudioFormat.FLAC)
    call = requests_mock.post(GLOBAL_URL, json=success(b"flac".hex()))

    output = run_node(node, {"text": "Hello"})

    body = call.last_request.json()
    assert body["voice_setting"] == {"voice_id": MiniMaxTTSAdapter.default_voice}
    assert body["audio_setting"] == {"format": "flac"}
    assert output["files"][0].name == "audio.flac"
    assert output["mime_type"] == "audio/flac"


def test_minimax_voice_from_provider_options_is_kept(requests_mock):
    node = TextToSpeech(
        connection=connections.MiniMax(api_key="k"),
        provider_options={"voice_setting": {"voice_id": "cloned-voice"}},
    )
    call = requests_mock.post(GLOBAL_URL, json=success(b"a".hex()))

    run_node(node, {"text": "Hello"})

    assert call.last_request.json()["voice_setting"] == {"voice_id": "cloned-voice"}


def test_minimax_china_region_and_url_output(requests_mock):
    audio = b"wave audio"
    audio_url = "https://audio.example.test/generated.wav"
    node = TextToSpeech(
        connection=connections.MiniMax(region=connections.MiniMaxRegion.CHINA, api_key="k"),
        model="speech-2.8-turbo",
        output_format=AudioFormat.WAV,
        provider_options={"output_format": "url"},
    )
    call = requests_mock.post(CHINA_URL, json=success(audio_url))
    download = requests_mock.get(audio_url, content=audio)

    output = run_node(node, {"text": "Hello"})

    assert call.last_request.json()["output_format"] == "url"
    assert call.last_request.json()["model"] == "speech-2.8-turbo"
    assert download.called_once
    assert "Authorization" not in download.last_request.headers
    assert output["content"] == audio
    assert output["files"][0].name == "audio.wav"
    assert output["mime_type"] == "audio/wav"


def test_minimax_streaming_cannot_be_enabled(requests_mock):
    node = TextToSpeech(connection=connections.MiniMax(api_key="k"), provider_options={"stream": True})
    call = requests_mock.post(GLOBAL_URL, json=success(b"a".hex()))

    run_node(node, {"text": "Hello"})

    assert call.last_request.json()["stream"] is False


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"base_resp": {"status_code": 1004, "status_msg": "login fail"}}, "status 1004: login fail"),
        ({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}}, "insufficient balance"),
        ({"data": {"audio": "", "status": 2}, "base_resp": {"status_code": 0}}, "no audio"),
        ({"data": {"audio": "zz", "status": 2}, "base_resp": {"status_code": 0}}, "not valid hex"),
        ({"data": {"audio": "00"}}, "status None"),
    ],
)
def test_minimax_reports_api_errors(requests_mock, payload, match):
    node = TextToSpeech(connection=connections.MiniMax(api_key="k"))
    requests_mock.post(GLOBAL_URL, json=payload)
    node.init_components()

    with pytest.raises(ValueError, match=match):
        node.execute(TextToSpeechInputSchema(text="Hello"))


def test_minimax_http_error_keeps_the_body(requests_mock):
    node = TextToSpeech(connection=connections.MiniMax(api_key="k"))
    requests_mock.post(GLOBAL_URL, status_code=500, text="upstream down")
    node.init_components()

    with pytest.raises(Exception, match="MiniMax request failed with status 500: upstream down"):
        node.execute(TextToSpeechInputSchema(text="Hello"))


def test_minimax_limits():
    with pytest.raises(ValueError, match="cannot produce 'opus'"):
        TextToSpeech(connection=connections.MiniMax(api_key="k"), output_format=AudioFormat.OPUS)
    with pytest.raises(ValueError, match="does not accept voice instructions"):
        TextToSpeech(connection=connections.MiniMax(api_key="k"), instructions="Whisper")

    node = TextToSpeech(connection=connections.MiniMax(api_key="k"), sample_rate=48000)
    with pytest.raises(ValueError, match="supports sample rates"):
        node.execute(TextToSpeechInputSchema(text="Hello"))

    node = TextToSpeech(connection=connections.MiniMax(api_key="k"))
    with pytest.raises(ValueError, match="at most 9999 characters"):
        node.execute(TextToSpeechInputSchema(text="x" * 10000))


def test_minimax_text_to_speech_yaml_round_trip(tmp_path):
    node = TextToSpeech(
        connection=connections.MiniMax(region=connections.MiniMaxRegion.CHINA, api_key="k"),
        model="speech-2.8-turbo",
        voice="male-qn-qingse",
        output_format=AudioFormat.FLAC,
        provider_options={"language_boost": "Chinese"},
    )
    yaml_path = tmp_path / "minimax-tts.yaml"

    Workflow(flow=Flow(nodes=[node])).to_yaml_file(str(yaml_path))
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=False).flow.nodes[0]

    assert isinstance(loaded, TextToSpeech)
    assert isinstance(loaded.connection, connections.MiniMax)
    assert loaded.connection.region == connections.MiniMaxRegion.CHINA
    assert loaded.connection.url == "https://api.minimax.cn/v1"
    assert loaded.model == "speech-2.8-turbo"
    assert loaded.voice == "male-qn-qingse"
    assert loaded.output_format == AudioFormat.FLAC
    assert loaded.provider_options == {"language_boost": "Chinese"}
