import json

from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.prompts import Message, MessageRole


USER_ID = "u1"
SESSION_ID = "s1"


def _scope_metadata() -> dict:
    return {"user_id": USER_ID, "session_id": SESSION_ID}


def test_add_preserves_tool_calls_on_assistant_message():
    memory = Memory(backend=InMemory())
    tool_calls = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
        }
    ]

    memory.add(
        role=MessageRole.ASSISTANT,
        content="Calling: get_weather",
        metadata=_scope_metadata(),
        tool_calls=tool_calls,
    )

    [restored] = memory.get_all()
    assert restored.role == MessageRole.ASSISTANT
    assert restored.content == "Calling: get_weather"
    assert restored.tool_calls == tool_calls
    # Stash keys must not leak into user-visible metadata after hydration.
    assert "_tool_calls" not in restored.metadata


def test_add_preserves_tool_call_id_and_name_on_tool_message():
    memory = Memory(backend=InMemory())

    memory.add(
        role=MessageRole.TOOL,
        content="Acknowledged.",
        metadata=_scope_metadata(),
        tool_call_id="call_abc",
        name="provide_final_answer",
    )

    [restored] = memory.get_all()
    assert restored.role == MessageRole.TOOL
    assert restored.tool_call_id == "call_abc"
    assert restored.name == "provide_final_answer"
    assert "_tool_call_id" not in restored.metadata
    assert "_name" not in restored.metadata


def test_stash_is_json_string_for_flat_metadata_backends():
    """The tool_calls stash is JSON-encoded as a string so backends that only
    accept flat scalars in metadata (e.g. Pinecone) can still persist it."""
    memory = Memory(backend=InMemory())
    tool_calls = [{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}]

    memory.add(
        role=MessageRole.ASSISTANT,
        content="...",
        metadata=_scope_metadata(),
        tool_calls=tool_calls,
    )

    [stored] = memory.backend.messages
    raw = stored.metadata.get("_tool_calls")
    assert isinstance(raw, str)
    assert json.loads(raw) == tool_calls


def test_replace_messages_round_trips_fc_fields():
    """replace_messages must thread tool_calls/tool_call_id/name to add(),
    so a snapshot save in FULL mode preserves the FC trace."""
    memory = Memory(backend=InMemory())
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    snapshot = [
        Message(role=MessageRole.USER, content="hi", metadata=_scope_metadata()),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: f",
            metadata=_scope_metadata(),
            tool_calls=tool_calls,
        ),
        Message(
            role=MessageRole.TOOL,
            content="ok",
            metadata=_scope_metadata(),
            tool_call_id="call_1",
            name="f",
        ),
    ]

    memory.replace_messages(filters=_scope_metadata(), messages=snapshot)
    restored = memory.get_all()

    assert [m.role for m in restored] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL]
    assert restored[1].tool_calls == tool_calls
    assert restored[2].tool_call_id == "call_1"
    assert restored[2].name == "f"


def test_transform_does_not_discard_falsy_but_present_tool_call_id_or_name():
    """Regression: the prior ``or``-based fallback collapsed an empty-string
    tool_call_id/name to None and silently fell through to the stash. The
    serializer only strips ``None``; a present-but-falsy value must round-trip
    unchanged, matching how tool_calls already handles ``[]``."""
    msg = Message(
        role=MessageRole.TOOL,
        content="ok",
        metadata={"timestamp": 1.0},
        tool_call_id="",
        name="",
    )

    [restored] = Memory._transform_function_calling_tool_fields([msg])

    assert restored.tool_call_id == ""
    assert restored.name == ""


def test_add_without_fc_fields_does_not_inject_stash_keys():
    """Plain user/assistant messages must not accumulate reserved metadata keys."""
    memory = Memory(backend=InMemory())

    memory.add(role=MessageRole.USER, content="hello", metadata=_scope_metadata())

    [restored] = memory.get_all()
    assert restored.tool_calls is None
    assert restored.tool_call_id is None
    assert restored.name is None
    assert "_tool_calls" not in restored.metadata
    assert "_tool_call_id" not in restored.metadata
    assert "_name" not in restored.metadata
