import pytest

from dynamiq.connections import Deepgram as DeepgramConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.components.schema_generator import generate_input_formats
from dynamiq.nodes.audio import SpeechToText
from dynamiq.nodes.audio.speech_to_text import SpeechToTextInputSchema
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.schema_utils import strip_inaccessible_fields
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def _agent_with_store(llm, tool, store):
    return Agent(name="Agent", llm=llm, role="r", tools=[tool], file_store=FileStoreConfig(enabled=True, backend=store))


def _stt() -> SpeechToText:
    return SpeechToText(connection=DeepgramConnection(api_key="dg-key"))


def test_the_recording_is_transcribed_not_the_first_upload(test_llm):
    """An agent injects every stored file; a contract uploaded first must not become the transcript."""
    tool = _stt()
    store = InMemoryFileStore()
    store.store("contract.pdf", b"%PDF-1.4", content_type="application/pdf")
    store.store("meeting.wav", b"RIFF", content_type="audio/wav")
    agent = _agent_with_store(test_llm, tool, store)

    merged: dict = {}
    agent._inject_files_into_tool(tool, merged)

    assert SpeechToTextInputSchema(**merged).audio.name == "meeting.wav"


def test_the_llm_can_name_which_recording_to_transcribe(test_llm):
    tool = _stt()
    store = InMemoryFileStore()
    store.store("call-1.mp3", b"one", content_type="audio/mpeg")
    store.store("call-2.mp3", b"two", content_type="audio/mpeg")
    agent = _agent_with_store(test_llm, tool, store)

    merged = {"audio": "call-2.mp3"}
    agent._inject_files_into_tool(tool, merged)

    assert SpeechToTextInputSchema(**merged).audio.read() == b"two"


def test_the_file_the_llm_named_survives_the_hidden_field_strip():
    """The agent strips fields it is not allowed to set before running a tool; the chosen file
    has to come through it, or the sandbox can never be asked for the recording."""
    kept, stripped = strip_inaccessible_fields(_stt().resolved_input_schema, {"audio": "/workspace/call.wav"})

    assert kept == {"audio": "/workspace/call.wav"}
    assert stripped == []


def test_the_llm_is_told_it_can_choose_the_recording():
    formats = generate_input_formats([_stt()], sanitize_tool_name=lambda name: name)

    assert "audio (tuple[str, ...])" in formats


def test_audio_url_is_hidden_from_the_llm_when_the_provider_cannot_fetch_one():
    openai_node = SpeechToText(connection=OpenAIConnection(api_key="k"))
    deepgram_node = _stt()

    assert "audio_url" not in generate_input_formats([openai_node], sanitize_tool_name=lambda name: name)
    assert "audio_url" in generate_input_formats([deepgram_node], sanitize_tool_name=lambda name: name)


def test_hiding_audio_url_leaves_it_usable_outside_an_agent():
    node = SpeechToText(connection=OpenAIConnection(api_key="k"))

    assert node.input_schema(audio_url="https://cdn.example.com/a.mp3").audio_url
