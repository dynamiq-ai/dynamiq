from io import BytesIO
from types import SimpleNamespace

import pytest

from dynamiq import Workflow, connections
from dynamiq.components.audio.openai_client import resolve_openai_client
from dynamiq.components.audio.stt.elevenlabs import elevenlabs_api_base
from dynamiq.flows import Flow
from dynamiq.nodes.audio import SpeechToText
from dynamiq.nodes.audio.speech_to_text import SpeechToTextInputSchema
from dynamiq.runnables import RunnableStatus
from dynamiq.types.audio import TimestampGranularity

DEEPGRAM_RESPONSE = {
    "metadata": {"request_id": "req-1", "duration": 6.8},
    "results": {
        "channels": [
            {
                "detected_language": "en",
                "alternatives": [
                    {
                        "transcript": "Hi, this is Ana from support. Hello Ana, I'm calling about invoice 4471.",
                        "confidence": 0.98,
                        "words": [
                            {
                                "word": "hi",
                                "punctuated_word": "Hi,",
                                "start": 0.4,
                                "end": 0.61,
                                "confidence": 0.99,
                                "speaker": 0,
                            },
                            {
                                "word": "hello",
                                "punctuated_word": "Hello",
                                "start": 3.2,
                                "end": 3.5,
                                "confidence": 0.97,
                                "speaker": 1,
                            },
                        ],
                    }
                ],
            }
        ],
        "utterances": [
            {
                "id": "u1",
                "start": 0.4,
                "end": 2.9,
                "transcript": "Hi, this is Ana from support.",
                "speaker": 0,
                "confidence": 0.97,
            },
            {
                "id": "u2",
                "start": 3.2,
                "end": 6.8,
                "transcript": "Hello Ana, I'm calling about invoice 4471.",
                "speaker": 1,
                "confidence": 0.95,
            },
        ],
    },
}

MISTRAL_RESPONSE = {
    "model": "voxtral-mini-latest",
    "text": "Bonjour à tous. Merci.",
    "language": "fr",
    "segments": [
        {
            "text": "Bonjour à tous.",
            "start": 0.0,
            "end": 1.4,
            "type": "transcription_segment",
            "speaker_id": "speaker_1",
            "score": 0.9,
        },
        {"text": "Merci.", "start": 1.6, "end": 2.1, "type": "transcription_segment", "speaker_id": "speaker_2"},
    ],
    "usage": {"prompt_audio_seconds": 2, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

ELEVENLABS_RESPONSE = {
    "language_code": "en",
    "language_probability": 0.99,
    "text": "Hi there. (laughs) Bye.",
    "words": [
        {"text": "Hi", "type": "word", "start": 0.0, "end": 0.2, "speaker_id": "speaker_0"},
        {"text": " ", "type": "spacing", "start": 0.2, "end": 0.3, "speaker_id": "speaker_0"},
        {"text": "there.", "type": "word", "start": 0.3, "end": 0.6, "speaker_id": "speaker_0"},
        {"text": "(laughs)", "type": "audio_event", "start": 0.7, "end": 1.0, "speaker_id": "speaker_1"},
        {"text": "Bye.", "type": "word", "start": 1.1, "end": 1.4, "speaker_id": "speaker_1"},
    ],
}


class FakeTranscription:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def fake_openai_client(mocker, payload):
    create = mocker.Mock(return_value=FakeTranscription(payload))
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    mocker.patch("dynamiq.components.audio.stt.openai.resolve_openai_client", return_value=client)
    return create


def run_node(node, input_data):
    result = Workflow(flow=Flow(nodes=[node])).run(input_data=input_data)
    assert result.status == RunnableStatus.SUCCESS, result.output[node.id]
    return result.output[node.id]["output"]


def test_deepgram_diarized_transcription(requests_mock):
    node = SpeechToText(
        connection=connections.Deepgram(api_key="dg-key"),
        model="nova-3",
        language="en",
        diarize=True,
        timestamps="word",
        prompt="Dynamiq, Nexus",
        provider_options={"numerals": True},
    )
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=DEEPGRAM_RESPONSE)

    output = run_node(node, {"audio": b"\x00\x01"})

    assert output["content"].startswith("Hi, this is Ana")
    assert output["transcript"] == (
        "Speaker 0: Hi, this is Ana from support.\nSpeaker 1: Hello Ana, I'm calling about invoice 4471."
    )
    assert [segment["speaker"] for segment in output["segments"]] == ["0", "1"]
    assert output["segments"][0]["start"] == 0.4
    assert output["segments"][0]["synthetic"] is False
    assert output["words"][0] == {
        "word": "Hi,",
        "start": 0.4,
        "end": 0.61,
        "speaker": "0",
        "confidence": 0.99,
        "type": "word",
    }
    assert output["speakers"] == [{"id": "0", "label": "Speaker 0"}, {"id": "1", "label": "Speaker 1"}]
    assert output["language"] == "en"
    assert output["duration"] == 6.8
    assert output["usage"] == {"seconds": 6.8}
    assert "raw" not in output

    assert call.called_once
    query = call.last_request.qs
    assert query["model"] == ["nova-3"]
    assert query["diarize"] == ["true"]
    assert query["utterances"] == ["true"]
    assert query["smart_format"] == ["true"]
    assert query["language"] == ["en"]
    assert query["keyterm"] == ["dynamiq", "nexus"]
    assert query["numerals"] == ["true"]
    assert call.last_request.headers["Authorization"] == "Token dg-key"
    assert call.last_request.headers["Content-Type"] == "audio/wav"
    assert call.last_request.body == b"\x00\x01"


def test_deepgram_accepts_audio_url_and_returns_raw_response_on_request(requests_mock):
    node = SpeechToText(connection=connections.Deepgram(api_key="dg-key"), timestamps="none", include_raw_response=True)
    response = {
        "metadata": {"duration": 1.0},
        "results": {"channels": [{"alternatives": [{"transcript": "Hello", "words": []}]}]},
    }
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=response)

    output = run_node(node, {"audio_url": "https://files.example.com/call.mp3"})

    assert node.model == "nova-3"
    assert output["content"] == "Hello"
    assert output["transcript"] == "Hello"
    assert len(output["segments"]) == 1 and output["segments"][0]["synthetic"] is True
    assert output["raw"] == response
    assert call.last_request.json() == {"url": "https://files.example.com/call.mp3"}
    assert "utterances" not in call.last_request.qs


def test_deepgram_word_timestamps_keep_utterances(requests_mock):
    node = SpeechToText(connection=connections.Deepgram(api_key="dg-key"), timestamps="word")
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=DEEPGRAM_RESPONSE)

    output = run_node(node, {"audio": b"\x00\x01"})

    assert call.last_request.qs["utterances"] == ["true"]
    assert "diarize" not in call.last_request.qs
    assert len(output["segments"]) == 2
    assert all(segment["synthetic"] is False for segment in output["segments"])
    assert output["words"][0]["start"] == 0.4


def test_deepgram_prefers_diarize_model_when_provided(requests_mock):
    node = SpeechToText(
        connection=connections.Deepgram(api_key="dg-key"), diarize=True, provider_options={"diarize_model": "latest"}
    )
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=DEEPGRAM_RESPONSE)

    run_node(node, {"audio": b"abc"})

    assert call.last_request.qs["diarize_model"] == ["latest"]
    assert "diarize" not in call.last_request.qs


def test_mistral_diarized_transcription(requests_mock):
    node = SpeechToText(connection=connections.Mistral(api_key="m-key"), diarize=True, language="fr", prompt="Dynamiq")
    call = requests_mock.post("https://api.mistral.ai/v1/audio/transcriptions", json=MISTRAL_RESPONSE)
    audio = BytesIO(b"abc")
    audio.name = "call.mp3"
    audio.content_type = "audio/mpeg"

    output = run_node(node, {"audio": audio})

    assert node.model == "voxtral-mini-latest"
    assert output["language"] == "fr"
    assert output["transcript"] == "Speaker 1: Bonjour à tous.\nSpeaker 2: Merci."
    assert [segment["speaker"] for segment in output["segments"]] == ["speaker_1", "speaker_2"]
    assert output["segments"][0]["confidence"] == 0.9
    assert output["duration"] == 2
    assert output["usage"]["prompt_tokens"] == 10
    assert output["words"] == []

    body = call.last_request.body
    assert b'name="model"\r\n\r\nvoxtral-mini-latest' in body
    assert b'name="diarize"\r\n\r\ntrue' in body
    assert b'name="language"\r\n\r\nfr' in body
    assert b'name="context_bias"\r\n\r\nDynamiq' in body
    # Diarization is refused without segment granularity, so it is always sent alongside.
    assert b'name="timestamp_granularities"\r\n\r\nsegment' in body
    assert b'filename="call.mp3"' in body
    assert call.last_request.headers["Authorization"] == "Bearer m-key"


def test_mistral_word_timestamps_and_file_url(requests_mock):
    node = SpeechToText(connection=connections.Mistral(api_key="m-key"), timestamps="word")
    call = requests_mock.post("https://api.mistral.ai/v1/audio/transcriptions", json=MISTRAL_RESPONSE)

    run_node(node, {"audio_url": "https://files.example.com/call.mp3"})

    body = call.last_request.body
    assert b'name="timestamp_granularities"\r\n\r\nword' in body
    assert b'name="file_url"\r\n\r\nhttps://files.example.com/call.mp3' in body
    assert b"filename=" not in body


def test_mistral_rejects_word_timestamps_with_diarization():
    with pytest.raises(ValueError, match="segment timings only"):
        SpeechToText(connection=connections.Mistral(api_key="k"), diarize=True, timestamps="word")


def test_elevenlabs_diarized_transcription(requests_mock):
    node = SpeechToText(
        connection=connections.ElevenLabs(api_key="xi-key"),
        diarize=True,
        speakers={"expected": 2},
        timestamps="word",
    )
    call = requests_mock.post("https://api.elevenlabs.io/v1/speech-to-text", json=ELEVENLABS_RESPONSE)

    output = run_node(node, {"audio": b"abc"})

    assert node.model == "scribe_v2"
    assert output["language"] == "en"
    assert output["duration"] == 1.4
    assert [segment["text"] for segment in output["segments"]] == ["Hi there.", "(laughs) Bye."]
    assert [segment["speaker"] for segment in output["segments"]] == ["speaker_0", "speaker_1"]
    assert all(segment["synthetic"] for segment in output["segments"])
    assert [word["type"] for word in output["words"]] == ["word", "word", "audio_event", "word"]
    assert output["transcript"] == "Speaker 0: Hi there.\nSpeaker 1: (laughs) Bye."

    body = call.last_request.body
    assert b'name="model_id"\r\n\r\nscribe_v2' in body
    assert b'name="diarize"\r\n\r\ntrue' in body
    assert b'name="num_speakers"\r\n\r\n2' in body
    assert b'name="timestamps_granularity"\r\n\r\nword' in body
    assert call.last_request.headers["xi-api-key"] == "xi-key"


def test_elevenlabs_api_base_accepts_legacy_endpoint_urls():
    assert elevenlabs_api_base("https://api.elevenlabs.io/v1/") == "https://api.elevenlabs.io/v1"
    assert elevenlabs_api_base("https://api.elevenlabs.io/v1/text-to-speech/") == "https://api.elevenlabs.io/v1"
    assert elevenlabs_api_base("https://proxy.internal/") == "https://proxy.internal"


def test_openai_diarization_uses_the_diarized_json_contract(mocker):
    payload = {
        "task": "transcribe",
        "duration": 6.8,
        "text": "Hi. Hello.",
        "segments": [
            {"id": "seg_1", "type": "transcript.text.segment", "start": 0.4, "end": 2.9, "text": "Hi.", "speaker": "A"},
            {
                "id": "seg_2",
                "type": "transcript.text.segment",
                "start": 3.2,
                "end": 6.8,
                "text": "Hello.",
                "speaker": "B",
            },
        ],
        "usage": {"type": "duration", "seconds": 7},
    }
    create = fake_openai_client(mocker, payload)
    node = SpeechToText(
        connection=connections.OpenAI(api_key="sk-test"),
        model="gpt-4o-transcribe-diarize",
        diarize=True,
        prompt="ignored by the diarization model",
    )

    output = run_node(node, {"audio": b"abc"})

    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe-diarize"
    assert kwargs["response_format"] == "diarized_json"
    assert kwargs["chunking_strategy"] == "auto"
    assert "prompt" not in kwargs
    assert kwargs["file"][0] == "audio.wav"
    assert output["speakers"] == [{"id": "A", "label": "Speaker A"}, {"id": "B", "label": "Speaker B"}]
    assert output["transcript"] == "Speaker A: Hi.\nSpeaker B: Hello."
    assert output["duration"] == 6.8
    assert output["segments"][0]["id"] == "seg_1"
    assert output["words"] == []


def test_openai_whisper_requests_verbose_json_for_word_timestamps(mocker):
    payload = {
        "task": "transcribe",
        "language": "english",
        "duration": 1.0,
        "text": "Hello world",
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": " Hello world"}],
        "words": [{"word": "Hello", "start": 0.0, "end": 0.5}, {"word": "world", "start": 0.6, "end": 1.0}],
    }
    create = fake_openai_client(mocker, payload)
    node = SpeechToText(connection=connections.OpenAI(api_key="sk-test"), model="whisper-1", timestamps="word")

    output = run_node(node, {"audio": b"abc"})

    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == "verbose_json"
    assert kwargs["timestamp_granularities"] == ["word", "segment"]
    assert output["language"] == "english"
    assert [word["word"] for word in output["words"]] == ["Hello", "world"]
    assert output["segments"][0]["id"] == "0"


def test_openai_default_model_uses_plain_json(mocker):
    payload = {"text": "Hi", "languages": [{"code": "en"}], "usage": {"type": "tokens", "total_tokens": 12}}
    create = fake_openai_client(mocker, payload)
    node = SpeechToText(connection=connections.OpenAI(api_key="sk-test"), prompt="Dynamiq")

    output = run_node(node, {"audio": b"abc"})

    assert node.model == "gpt-4o-transcribe"
    assert create.call_args.kwargs["response_format"] == "json"
    assert create.call_args.kwargs["prompt"] == "Dynamiq"
    assert output["language"] == "en"
    assert output["languages"] == ["en"]
    assert output["usage"]["total_tokens"] == 12


def test_groq_uses_the_openai_contract(mocker):
    create = fake_openai_client(mocker, {"text": "Hi"})
    node = SpeechToText(connection=connections.Groq(api_key="gsk"), model="whisper-large-v3-turbo")

    output = run_node(node, {"audio": b"abc"})

    assert output["content"] == "Hi"
    assert create.call_args.kwargs["response_format"] == "verbose_json"

    client = resolve_openai_client(connections.Groq(api_key="gsk"), None)
    assert "api.groq.com" in str(client.base_url)


def test_default_models_follow_the_provider():
    assert SpeechToText(connection=connections.Deepgram(api_key="k")).model == "nova-3"
    assert SpeechToText(connection=connections.OpenAI(api_key="k")).model == "gpt-4o-transcribe"
    assert SpeechToText(connection=connections.Mistral(api_key="k")).model == "voxtral-mini-latest"
    assert SpeechToText(connection=connections.ElevenLabs(api_key="k")).model == "scribe_v2"
    assert SpeechToText(connection=connections.Groq(api_key="k")).model == "whisper-large-v3-turbo"
    assert SpeechToText(connection=connections.Whisper(api_key="k")).model == "whisper-1"
    assert SpeechToText(connection=connections.HttpApiKey(url="http://vllm:8000/v1", api_key="k")).model == "whisper-1"


def test_unsupported_options_fail_at_construction():
    with pytest.raises(ValueError, match="does not support speaker diarization"):
        SpeechToText(connection=connections.Groq(api_key="k"), diarize=True)
    with pytest.raises(ValueError, match="does not accept a transcription prompt"):
        SpeechToText(connection=connections.ElevenLabs(api_key="k"), prompt="Dynamiq")
    with pytest.raises(ValueError, match="'min' must not exceed 'max'"):
        SpeechToText(connection=connections.Deepgram(api_key="k"), diarize=True, speakers={"min": 3, "max": 2})
    with pytest.raises(ValueError):
        SpeechToText(connection=connections.Anthropic(api_key="k"))


def test_agent_file_injection_contract(requests_mock):
    # Agents fill map_from_storage fields with every stored file and must not show raw audio to the LLM.
    field = SpeechToText.input_schema.model_fields["audio"]
    assert field.json_schema_extra == {"map_from_storage": True, "is_accessible_to_agent": False}

    node = SpeechToText(connection=connections.Deepgram(api_key="dg-key"), timestamps="none")
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=DEEPGRAM_RESPONSE)

    output = run_node(node, {"audio": [b"\x00\x01", b"\x02\x03"]})

    assert call.last_request.body == b"\x00\x01"
    assert output["content"].startswith("Hi, this is Ana")


def test_audio_url_wins_over_injected_files(requests_mock):
    node = SpeechToText(connection=connections.Deepgram(api_key="dg-key"), timestamps="none")
    call = requests_mock.post("https://api.deepgram.com/v1/listen", json=DEEPGRAM_RESPONSE)

    run_node(node, {"audio": [b"\x00\x01"], "audio_url": "https://cdn.example.com/call.wav"})

    assert call.last_request.json() == {"url": "https://cdn.example.com/call.wav"}


def test_unsupported_input_fails_at_execution():
    node = SpeechToText(connection=connections.ElevenLabs(api_key="k"))
    with pytest.raises(ValueError, match="does not accept audio by URL"):
        node.execute(SpeechToTextInputSchema(audio_url="https://files.example.com/a.mp3"))

    node = SpeechToText(connection=connections.OpenAI(api_key="k"), diarize=True)
    with pytest.raises(ValueError, match="requires model 'gpt-4o-transcribe-diarize'"):
        node.execute(SpeechToTextInputSchema(audio=b"abc"))

    with pytest.raises(ValueError, match="Either `audio` or `audio_url`"):
        SpeechToTextInputSchema()


def test_speech_to_text_yaml_round_trip(tmp_path):
    node = SpeechToText(
        connection=connections.Deepgram(api_key="dg-key"),
        diarize=True,
        timestamps="word",
        speakers={"min": 2, "max": 4},
        provider_options={"smart_format": False},
    )
    yaml_path = tmp_path / "speech-to-text.yaml"

    Workflow(flow=Flow(nodes=[node])).to_yaml_file(str(yaml_path))
    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=False).flow.nodes[0]

    assert isinstance(loaded, SpeechToText)
    assert isinstance(loaded.connection, connections.Deepgram)
    assert loaded.model == "nova-3"
    assert loaded.diarize is True
    assert loaded.timestamps == TimestampGranularity.WORD
    assert loaded.speakers.max == 4
    assert loaded.provider_options == {"smart_format": False}
