from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.memory.backends.dynamiq import Dynamiq
from dynamiq.prompts import Message, MessageRole


def _make_backend(mocker) -> Dynamiq:
    """Create a Dynamiq backend with HTTP calls patched out."""
    connection = DynamiqConnection(url="https://api.example/v1", api_key="test")
    mocker.patch.object(DynamiqConnection, "connect", return_value=mocker.MagicMock())
    return Dynamiq(connection=connection, memory_id="mem-1", user_id="u1", session_id="s1")


def test_add_sends_tool_calls_as_first_class_data_field(mocker):
    backend = _make_backend(mocker)
    request = mocker.patch.object(backend, "_request", return_value=None)

    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
    ]
    msg = Message(
        role=MessageRole.ASSISTANT,
        content="Calling: get_weather",
        metadata={"user_id": "u1", "session_id": "s1", "timestamp": 1.0},
        tool_calls=tool_calls,
    )
    backend.add(msg)

    sent_payload = request.call_args.kwargs["json"]
    data = sent_payload["data"]

    assert data["role"] == "assistant"
    assert data["tool_calls"] == tool_calls
    assert "_tool_calls" not in data.get("metadata", {})


def test_add_sends_tool_call_id_and_name_as_first_class_data_fields(mocker):
    backend = _make_backend(mocker)
    request = mocker.patch.object(backend, "_request", return_value=None)

    msg = Message(
        role=MessageRole.TOOL,
        content="ok",
        metadata={"user_id": "u1", "session_id": "s1", "timestamp": 1.0},
        tool_call_id="call_1",
        name="get_weather",
    )
    backend.add(msg)

    data = request.call_args.kwargs["json"]["data"]
    assert data["tool_call_id"] == "call_1"
    assert data["name"] == "get_weather"
    md = data.get("metadata", {})
    assert "_tool_call_id" not in md
    assert "_name" not in md


def test_add_strips_memory_layer_stash_from_metadata(mocker):
    """Even if the Memory layer added stash keys, the Dynamiq backend must not
    forward them — it owns its own wire shape."""
    backend = _make_backend(mocker)
    request = mocker.patch.object(backend, "_request", return_value=None)

    msg = Message(
        role=MessageRole.ASSISTANT,
        content="...",
        metadata={
            "user_id": "u1",
            "session_id": "s1",
            "timestamp": 1.0,
            "_tool_calls": '[{"id":"x"}]',
            "_tool_call_id": "x",
            "_name": "f",
        },
        tool_calls=[{"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    )
    backend.add(msg)

    sent_metadata = request.call_args.kwargs["json"]["data"].get("metadata", {})
    assert "_tool_calls" not in sent_metadata
    assert "_tool_call_id" not in sent_metadata
    assert "_name" not in sent_metadata


def test_items_to_messages_reads_tool_calls_from_first_class_data(mocker):
    backend = _make_backend(mocker)

    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]
    api_items = [
        {
            "user_id": "u1",
            "session_id": "s1",
            "type": "message",
            "created_at": "2026-01-01T00:00:00Z",
            "data": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Calling: f"}],
                "tool_calls": tool_calls,
            },
        },
        {
            "user_id": "u1",
            "session_id": "s1",
            "type": "message",
            "created_at": "2026-01-01T00:00:01Z",
            "data": {
                "role": "tool",
                "content": [{"type": "text", "text": "ok"}],
                "tool_call_id": "call_1",
                "name": "f",
            },
        },
    ]

    messages = backend._items_to_messages(api_items)

    assert messages[0].role == MessageRole.ASSISTANT
    assert messages[0].tool_calls == tool_calls
    assert messages[1].role == MessageRole.TOOL
    assert messages[1].tool_call_id == "call_1"
    assert messages[1].name == "f"
