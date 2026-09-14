"""Live provider checks for the unified speech nodes, and for using them from an Agent.

Requires real credentials in ``.env`` at the repository root. Each test skips on its own
provider's key, so a partial set still runs what it can:

* ``OPENAI_API_KEY``     — the agent's LLM, text-to-speech, transcription and diarization
* ``MISTRAL_API_KEY``    — Voxtral transcription and its diarization/granularity rule
* ``ELEVENLABS_API_KEY`` — Scribe diarization

Run with: ``uv run pytest tests/integration_with_creds/nodes/audio -q -s``
"""

import io
import os
import wave

import pytest
import requests

from dynamiq import connections
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.audio import SpeechToText, TextToSpeech
from dynamiq.nodes.audio.text_to_speech import TextToSpeechInputSchema
from dynamiq.nodes.llms import OpenAI as OpenAILLM
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore

requires_openai = pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
requires_mistral = pytest.mark.skipif(not os.getenv("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY is not set")
requires_elevenlabs = pytest.mark.skipif(not os.getenv("ELEVENLABS_API_KEY"), reason="ELEVENLABS_API_KEY is not set")

# Two turns, two voices, so diarization has something to separate. The rare words are what the
# assertions look for: they cannot come from anywhere but this recording.
CONVERSATION = [
    ("alloy", "Hello, this is Ana from the Zurich office. Invoice four four seven one is overdue."),
    ("onyx", "Thanks Ana. I will approve the payment for invoice four four seven one today."),
]

# A real, if minimal, PDF. It is the wrong file to transcribe, which is the point.
CONTRACT_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)


@pytest.fixture(scope="module")
def conversation_wav() -> io.BytesIO:
    """A genuine two-speaker recording, synthesized once per run with OpenAI TTS."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    turns = [
        TextToSpeech(
            connection=connections.OpenAI(), model="gpt-4o-mini-tts", voice=voice, output_format="wav"
        ).execute(TextToSpeechInputSchema(text=text))["content"]
        for voice, text in CONVERSATION
    ]

    merged = io.BytesIO()
    with wave.open(merged, "wb") as out:
        for index, turn in enumerate(turns):
            with wave.open(io.BytesIO(turn), "rb") as part:
                if index == 0:
                    # Not setparams: that would also copy the first turn's frame count into the
                    # header, and the second turn's frames would then overrun it.
                    out.setnchannels(part.getnchannels())
                    out.setsampwidth(part.getsampwidth())
                    out.setframerate(part.getframerate())
                out.writeframes(part.readframes(part.getnframes()))

    merged.name = "meeting.wav"
    merged.content_type = "audio/wav"
    merged.seek(0)
    return merged


def _agent(tools: list, store: InMemoryFileStore | None = None) -> Agent:
    return Agent(
        name="Audio agent",
        role="You handle recordings for the finance team. Use the tools you have.",
        llm=OpenAILLM(connection=connections.OpenAI(), model="gpt-4o", temperature=0),
        tools=tools,
        max_loops=6,
        file_store=FileStoreConfig(enabled=store is not None, backend=store or InMemoryFileStore()),
    )


def _store_the_way_an_agent_does(audio: bytes | None = None) -> InMemoryFileStore:
    """The agent's own upload path never sets a content type, so the store stamps every uploaded
    file `application/octet-stream`. Storing them any other way hides the case that matters."""
    store = InMemoryFileStore()
    store.store("contract.pdf", CONTRACT_PDF, content_type="application/octet-stream")
    if audio is not None:
        store.store("meeting.wav", audio, content_type="application/octet-stream")
    return store


def _observations(agent: Agent) -> list[str]:
    """What the agent actually showed its model after each tool call."""
    return [
        message.content.removeprefix("Observation: ").strip()
        for message in agent._prompt.messages
        if isinstance(message.content, str) and message.content.startswith("Observation:")
    ]


@requires_openai
def test_an_agent_transcribes_the_recording_and_not_the_other_upload(conversation_wav):
    """The fix under test: an agent injects every stored file, and the node has to choose."""
    store = _store_the_way_an_agent_does(conversation_wav.getvalue())
    tool = SpeechToText(connection=connections.OpenAI(), model="gpt-4o-transcribe", timestamps="none")
    agent = _agent([tool], store)

    result = agent.run(input_data={"input": "Transcribe the recording and quote the invoice number in it."})

    assert result.status.value == "success", result.output
    answer = result.output["content"]
    print("\nANSWER:", answer)
    assert "4471" in answer.replace(" ", "") or "four four seven one" in answer.lower()


@requires_openai
def test_an_agent_is_shown_a_description_of_generated_speech_not_its_bytes():
    """Without the fix the observation is thousands of characters of escaped MP3 bytes."""
    tool = TextToSpeech(connection=connections.OpenAI(), model="gpt-4o-mini-tts", voice="alloy")
    agent = _agent([tool])

    result = agent.run(input_data={"input": "Say 'your invoice is approved' out loud and return the audio."})

    assert result.status.value == "success", result.output
    observations = _observations(agent)
    print("\nOBSERVATIONS:", observations)
    speech = [text for text in observations if text.startswith("Produced audio/")]
    assert speech, f"no speech observation found in {observations}"
    assert "\\xff" not in " ".join(observations)
    assert all(len(text) < 500 for text in speech)


@requires_openai
def test_openai_diarization_live(conversation_wav):
    node = SpeechToText(
        connection=connections.OpenAI(), model="gpt-4o-transcribe-diarize", diarize=True, timestamps="segment"
    )

    output = node.execute(node.input_schema(audio=conversation_wav))

    print("\nOPENAI TRANSCRIPT:\n", output["transcript"])
    assert len({speaker["id"] for speaker in output["speakers"]}) >= 2
    assert output["transcript"].startswith("Speaker ")


@requires_mistral
def test_voxtral_diarization_live(conversation_wav):
    node = SpeechToText(connection=connections.Mistral(), diarize=True)

    output = node.execute(node.input_schema(audio=conversation_wav))

    print("\nVOXTRAL TRANSCRIPT:\n", output["transcript"])
    assert len({speaker["id"] for speaker in output["speakers"]}) >= 2
    assert output["segments"]


@requires_mistral
def test_voxtral_diarizes_with_a_language_hint_too(conversation_wav):
    """Voxtral's only rule is that diarization carries segment granularity — a language hint
    alongside it is fine. Guards against re-introducing a restriction the API does not have."""
    node = SpeechToText(connection=connections.Mistral(), diarize=True, language="en")

    output = node.execute(node.input_schema(audio=conversation_wav))

    print("\nVOXTRAL (language + diarize):\n", output["transcript"])
    assert len({speaker["id"] for speaker in output["speakers"]}) >= 2


@requires_mistral
def test_voxtral_really_does_require_segment_granularity_to_diarize(conversation_wav):
    """The control for the adapter always pairing `diarize` with segment granularity: drop it and
    the API refuses the request outright."""
    response = requests.post(
        "https://api.mistral.ai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
        files=[
            ("model", (None, "voxtral-mini-latest")),
            ("diarize", (None, "true")),
            ("file", ("meeting.wav", io.BytesIO(conversation_wav.getvalue()), "audio/wav")),
        ],
    )

    print("\nMISTRAL CONTROL:", response.status_code, response.text[:200])
    assert response.status_code >= 400
    assert "diarize" in response.text


@requires_elevenlabs
def test_elevenlabs_diarization_live(conversation_wav):
    node = SpeechToText(connection=connections.ElevenLabs(), diarize=True)

    output = node.execute(node.input_schema(audio=conversation_wav))

    print("\nSCRIBE TRANSCRIPT:\n", output["transcript"])
    assert len({speaker["id"] for speaker in output["speakers"]}) >= 2


@requires_openai
def test_an_agent_can_say_who_said_what(conversation_wav):
    """Diarization is only useful to an agent if the speaker labels survive into the observation."""
    store = _store_the_way_an_agent_does(conversation_wav.getvalue())
    tool = SpeechToText(
        connection=connections.OpenAI(), model="gpt-4o-transcribe-diarize", diarize=True, timestamps="segment"
    )
    agent = _agent([tool], store)

    result = agent.run(input_data={"input": "Who says the invoice is overdue, and who approves it? Name the speakers."})

    assert result.status.value == "success", result.output
    answer = result.output["content"]
    print("\nWHO SAID WHAT:", answer)
    observation = next(text for text in _observations(agent) if "overdue" in text)
    assert observation.startswith("Speaker "), observation[:200]
