import base64
from io import BytesIO
from types import SimpleNamespace

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.audio import TextToSpeech
from dynamiq.nodes.audio.text_to_speech import TextToSpeechInputSchema
from dynamiq.runnables import RunnableStatus
from dynamiq.types.audio import AudioFormat


def run_node(node, input_data):
    result = Workflow(flow=Flow(nodes=[node])).run(input_data=input_data)
    assert result.status == RunnableStatus.SUCCESS, result.output[node.id]
    return result.output[node.id]["output"]


def test_elevenlabs_text_to_speech(requests_mock):
    node = TextToSpeech(
        connection=connections.ElevenLabs(api_key="xi-key"),
        voice="voice-123",
        language="en",
        speed=1.1,
        provider_options={"voice_settings": {"stability": 0.4}, "seed": 7},
    )
    call = requests_mock.post("https://api.elevenlabs.io/v1/text-to-speech/voice-123", content=b"mp3-bytes")

    output = run_node(node, {"text": "Hello", "output_file_name": "hello.mp3"})

    assert node.model == "eleven_multilingual_v2"
    assert output["content"] == b"mp3-bytes"
    assert output["mime_type"] == "audio/mpeg"
    audio_file = output["files"][0]
    assert isinstance(audio_file, BytesIO)
    assert audio_file.name == "hello.mp3"
    assert audio_file.content_type == "audio/mpeg"
    assert audio_file.read() == b"mp3-bytes"

    assert call.last_request.qs["output_format"] == ["mp3_44100_128"]
    assert call.last_request.json() == {
        "text": "Hello",
        "model_id": "eleven_multilingual_v2",
        "language_code": "en",
        "voice_settings": {"stability": 0.4, "speed": 1.1},
        "seed": 7,
    }
    assert call.last_request.headers["xi-api-key"] == "xi-key"


def test_elevenlabs_maps_format_and_sample_rate(requests_mock):
    node = TextToSpeech(connection=connections.ElevenLabs(api_key="xi-key"), output_format="pcm", sample_rate=16000)
    call = requests_mock.post("https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM", content=b"pcm")

    output = run_node(node, {"text": "Hello"})

    assert call.last_request.qs["output_format"] == ["pcm_16000"]
    assert output["files"][0].name == "audio.pcm"
    assert output["mime_type"] == "audio/pcm"

    node = TextToSpeech(connection=connections.ElevenLabs(api_key="xi-key"), sample_rate=12345)
    with pytest.raises(ValueError, match="supports sample rates"):
        node.execute(TextToSpeechInputSchema(text="Hello"))


def test_deepgram_text_to_speech(requests_mock):
    node = TextToSpeech(
        connection=connections.Deepgram(api_key="dg-key"),
        voice="aura-2-helena-en",
        output_format=AudioFormat.WAV,
        sample_rate=16000,
        speed=1.2,
    )
    call = requests_mock.post("https://api.deepgram.com/v1/speak", content=b"wav-bytes")

    output = run_node(node, {"text": "Hi"})

    query = call.last_request.qs
    assert query["model"] == ["aura-2-helena-en"]
    assert query["encoding"] == ["linear16"]
    assert query["container"] == ["wav"]
    assert query["sample_rate"] == ["16000"]
    assert query["speed"] == ["1.2"]
    assert call.last_request.json() == {"text": "Hi"}
    assert call.last_request.headers["Authorization"] == "Token dg-key"
    assert output["files"][0].name == "audio.wav"
    assert output["content"] == b"wav-bytes"


def test_deepgram_uses_model_as_voice_and_enforces_text_limit(requests_mock):
    node = TextToSpeech(connection=connections.Deepgram(api_key="dg-key"))
    call = requests_mock.post("https://api.deepgram.com/v1/speak", content=b"mp3")

    run_node(node, {"text": "Hi"})

    assert call.last_request.qs["model"] == ["aura-2-thalia-en"]
    assert call.last_request.qs["encoding"] == ["mp3"]
    assert "container" not in call.last_request.qs

    with pytest.raises(ValueError, match="at most 2000 characters"):
        node.execute(TextToSpeechInputSchema(text="x" * 2001))


def test_mistral_text_to_speech(requests_mock):
    node = TextToSpeech(connection=connections.Mistral(api_key="m-key"), voice="voice-abc", output_format="wav")
    call = requests_mock.post(
        "https://api.mistral.ai/v1/audio/speech",
        json={"audio_data": base64.b64encode(b"wav-bytes").decode()},
    )

    output = run_node(node, {"text": "Hi"})

    assert output["content"] == b"wav-bytes"
    assert output["files"][0].name == "audio.wav"
    assert call.last_request.json() == {
        "model": "voxtral-mini-tts-latest",
        "input": "Hi",
        "response_format": "wav",
        "voice_id": "voice-abc",
    }
    assert call.last_request.headers["Authorization"] == "Bearer m-key"


def test_mistral_falls_back_to_a_preset_voice(requests_mock):
    # Voxtral rejects a speech request that names neither a voice nor reference audio.
    node = TextToSpeech(connection=connections.Mistral(api_key="m-key"))
    call = requests_mock.post(
        "https://api.mistral.ai/v1/audio/speech",
        json={"audio_data": base64.b64encode(b"mp3-bytes").decode()},
    )

    run_node(node, {"text": "Hi"})

    assert call.last_request.json()["voice_id"] == "en_paul_neutral"


def test_mistral_reference_audio_replaces_the_voice(requests_mock):
    node = TextToSpeech(connection=connections.Mistral(api_key="m-key"), provider_options={"ref_audio": "base64-clip"})
    call = requests_mock.post(
        "https://api.mistral.ai/v1/audio/speech",
        json={"audio_data": base64.b64encode(b"mp3-bytes").decode()},
    )

    run_node(node, {"text": "Hi"})

    body = call.last_request.json()
    assert body["ref_audio"] == "base64-clip"
    assert "voice_id" not in body


def test_openai_text_to_speech(mocker):
    create = mocker.Mock(return_value=SimpleNamespace(content=b"audio"))
    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    mocker.patch("dynamiq.components.audio.tts.openai.resolve_openai_client", return_value=client)
    node = TextToSpeech(
        connection=connections.OpenAI(api_key="sk-test"),
        voice="marin",
        instructions="Calm and unhurried",
        speed=0.9,
        output_format="wav",
        output_file_name="reply.wav",
    )

    output = run_node(node, {"text": "Hi"})

    assert create.call_args.kwargs == {
        "model": "gpt-4o-mini-tts",
        "voice": "marin",
        "input": "Hi",
        "response_format": "wav",
        "speed": 0.9,
        "instructions": "Calm and unhurried",
    }
    assert output["content"] == b"audio"
    assert output["files"][0].name == "reply.wav"
    assert output["mime_type"] == "audio/wav"


def test_openai_default_voice_is_used_when_none_is_set(mocker):
    create = mocker.Mock(return_value=SimpleNamespace(content=b"audio"))
    client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    mocker.patch("dynamiq.components.audio.tts.openai.resolve_openai_client", return_value=client)

    run_node(TextToSpeech(connection=connections.OpenAI(api_key="sk-test")), {"text": "Hi"})

    assert create.call_args.kwargs["voice"] == "alloy"
    assert create.call_args.kwargs["response_format"] == "mp3"


def test_groq_text_to_speech_limits():
    with pytest.raises(ValueError, match="cannot produce 'mp3'"):
        TextToSpeech(connection=connections.Groq(api_key="gsk"))

    node = TextToSpeech(connection=connections.Groq(api_key="gsk"), output_format="wav")
    assert node.model == "canopylabs/orpheus-v1-english"
    with pytest.raises(ValueError, match="at most 200 characters"):
        node.execute(TextToSpeechInputSchema(text="x" * 201))


def test_unsupported_options_fail_at_construction():
    with pytest.raises(ValueError, match="does not support a speed setting"):
        TextToSpeech(connection=connections.Mistral(api_key="k"), speed=1.2)
    with pytest.raises(ValueError, match="does not accept voice instructions"):
        TextToSpeech(connection=connections.ElevenLabs(api_key="k"), instructions="Whisper")
    with pytest.raises(ValueError, match="does not accept a language setting"):
        TextToSpeech(connection=connections.Deepgram(api_key="k"), language="en")
    with pytest.raises(ValueError):
        TextToSpeech(connection=connections.Anthropic(api_key="k"))


def test_text_to_speech_yaml_round_trip(tmp_path):
    node = TextToSpeech(
        connection=connections.ElevenLabs(api_key="xi-key"),
        voice="voice-123",
        output_format="pcm",
        sample_rate=24000,
        provider_options={"voice_settings": {"stability": 0.3}},
    )
    yaml_path = tmp_path / "text-to-speech.yaml"

    Workflow(flow=Flow(nodes=[node])).to_yaml_file(str(yaml_path))
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=False).flow.nodes[0]

    assert isinstance(loaded, TextToSpeech)
    assert isinstance(loaded.connection, connections.ElevenLabs)
    assert loaded.model == "eleven_multilingual_v2"
    assert loaded.voice == "voice-123"
    assert loaded.output_format == AudioFormat.PCM
    assert loaded.sample_rate == 24000
    assert loaded.provider_options == {"voice_settings": {"stability": 0.3}}
