import base64
import io

import pytest

from dynamiq.prompts.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageFileContent,
    VisionMessageFileData,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageType,
)


@pytest.fixture
def sample_png_bytes():
    """
    Returns bytes for a very small valid PNG image (1x1 pixel).
    This base64 string is properly padded and will decode without error.
    """
    # "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    # "YGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="
    # is just a 1x1 px transparent PNG.
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk" "YGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="
    return base64.b64decode(data)


def test_vision_prompt_with_bytesio(sample_png_bytes):
    """
    Tests that a BytesIO image is recognized and transformed into a proper
    base64-encoded data URL in the formatted vision message.
    """
    # Wrap our PNG bytes in a BytesIO object.
    image_bytes_io = io.BytesIO(sample_png_bytes)

    # Create a Prompt that expects an image URL with a Jinja placeholder {{ image }}.
    prompt = Prompt(
        id="test_vision",
        messages=[
            VisionMessage(
                content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image }}"))],
                role=MessageRole.USER,
            )
        ],
    )

    # Format the messages with the image parameter pointing to our BytesIO.
    out_messages = prompt.format_messages(image=image_bytes_io)

    # We expect exactly one message in the output.
    assert len(out_messages) == 1
    msg = out_messages[0]

    # Ensure the role is "user".
    assert msg["role"] == MessageRole.USER

    # Ensure the content is a list with one vision item.
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1

    # Extract the image content dictionary.
    image_content = msg["content"][0]
    assert image_content["type"] == VisionMessageType.IMAGE_URL

    # Check that the URL has a base64-encoded PNG.
    image_url = image_content["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")

    # Optional: decode and check that it starts with the PNG header bytes.
    encoded_part = image_url.split("base64,")[-1]
    decoded_part = base64.b64decode(encoded_part)
    assert decoded_part.startswith(b"\x89PNG\r\n\x1a\n")


def test_vision_prompt_with_regular_url():
    """
    Tests that if the user passes a normal string URL (no bytes), it
    remains unchanged in the formatted messages.
    """
    prompt = Prompt(
        id="test_vision_url",
        messages=[
            VisionMessage(content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image_url }}"))])
        ],
    )

    test_image_url = "https://example.com/sample.png"
    out_messages = prompt.format_messages(image_url=test_image_url)
    msg = out_messages[0]

    assert len(out_messages) == 1
    assert msg["role"] == MessageRole.USER
    assert len(msg["content"]) == 1

    image_content = msg["content"][0]
    assert image_content["type"] == VisionMessageType.IMAGE_URL
    assert image_content["image_url"]["url"] == test_image_url


def test_vision_prompt_unsupported_type():
    """
    Tests that using an unsupported type for the image parameter
    (e.g., an integer) raises a ValueError.
    """
    prompt = Prompt(
        id="test_unsupported",
        messages=[
            VisionMessage(content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image }}"))])
        ],
    )

    with pytest.raises(ValueError):
        prompt.format_messages(image=12345)


@pytest.fixture
def sample_mp4_bytes():
    """Minimal bytes recognized by `filetype` as video/mp4 (a bare ftyp box, no real media)."""
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 20


def test_vision_prompt_with_video_file_content(sample_mp4_bytes):
    """
    Tests that a video passed via VisionMessageFileContent is base64-encoded with the
    correct video mime type, and that video_metadata survives formatting — this is the
    wire shape litellm's Gemini transformation expects for native video input.
    """
    video_bytes_io = io.BytesIO(sample_mp4_bytes)

    prompt = Prompt(
        id="test_video",
        messages=[
            VisionMessage(
                content=[
                    VisionMessageFileContent(
                        file=VisionMessageFileData(
                            file_data="{{ video }}",
                            video_metadata={"fps": 1, "start_offset": "0s", "end_offset": "10s"},
                        )
                    )
                ],
                role=MessageRole.USER,
            )
        ],
    )

    out_messages = prompt.format_messages(video=video_bytes_io)

    assert len(out_messages) == 1
    msg = out_messages[0]
    assert msg["role"] == MessageRole.USER
    assert len(msg["content"]) == 1

    file_content = msg["content"][0]
    assert file_content["type"] == VisionMessageType.FILE

    file_data = file_content["file"]["file_data"]
    assert file_data.startswith("data:video/mp4;base64,")

    encoded_part = file_data.split("base64,")[-1]
    decoded_part = base64.b64decode(encoded_part)
    assert decoded_part.startswith(b"\x00\x00\x00\x18ftypmp42")

    assert file_content["file"]["video_metadata"] == {"fps": 1, "start_offset": "0s", "end_offset": "10s"}


def test_vision_prompt_video_file_content_without_metadata():
    """video_metadata is optional and excluded from the dumped dict when unset."""
    prompt = Prompt(
        id="test_video_no_metadata",
        messages=[
            VisionMessage(
                content=[VisionMessageFileContent(file=VisionMessageFileData(file_data="{{ video_url }}"))],
                role=MessageRole.USER,
            )
        ],
    )

    out_messages = prompt.format_messages(video_url="https://example.com/clip.mp4")
    file_content = out_messages[0]["content"][0]

    assert file_content["file"]["file_data"] == "https://example.com/clip.mp4"
    assert "video_metadata" not in file_content["file"]


class TestFunctionCallingProtocolMessages:
    """Tests for OpenAI function-calling protocol fields on the Message model."""

    def test_message_role_tool_exists(self):
        assert MessageRole.TOOL.value == "tool"

    def test_message_accepts_tool_calls_field(self):
        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]
        msg = Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls

    def test_message_accepts_tool_call_id_field(self):
        msg = Message(role=MessageRole.TOOL, content="result", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"

    def test_format_messages_strips_null_tool_calls_from_user(self):
        prompt = Prompt(messages=[Message(role=MessageRole.USER, content="hi")])
        out = prompt.format_messages()
        assert out == [{"role": "user", "content": "hi"}]
        assert "tool_calls" not in out[0]
        assert "tool_call_id" not in out[0]

    def test_format_messages_keeps_tool_calls_on_assistant(self):
        tool_calls = [
            {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": "{}"}}
        ]
        prompt = Prompt(
            messages=[Message(role=MessageRole.ASSISTANT, content="", tool_calls=tool_calls)]
        )
        out = prompt.format_messages()
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["tool_calls"] == tool_calls
        assert "tool_call_id" not in out[0]

    def test_format_messages_keeps_tool_call_id_on_tool_message(self):
        prompt = Prompt(
            messages=[
                Message(role=MessageRole.TOOL, content="Cats sleep 12-16h", tool_call_id="call_a")
            ]
        )
        out = prompt.format_messages()
        assert out == [{"role": "tool", "content": "Cats sleep 12-16h", "tool_call_id": "call_a"}]

    def test_format_messages_round_trip_well_formed_fc_conversation(self):
        """End-to-end: a complete FC round-trip serializes to OpenAI's expected shape."""
        prompt = Prompt(
            messages=[
                Message(role=MessageRole.USER, content="Call cat and dog tools in parallel"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {"id": "call_a", "type": "function",
                         "function": {"name": "CatFacts", "arguments": "{}"}},
                        {"id": "call_b", "type": "function",
                         "function": {"name": "DogFacts", "arguments": "{}"}},
                    ],
                ),
                Message(role=MessageRole.TOOL, content="Cats sleep 12-16h", tool_call_id="call_a"),
                Message(role=MessageRole.TOOL, content="Dogs have 40x smell", tool_call_id="call_b"),
            ]
        )
        out = prompt.format_messages()
        assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool"]
        assert out[1]["tool_calls"][0]["id"] == "call_a"
        assert out[2]["tool_call_id"] == "call_a"
        assert out[3]["tool_call_id"] == "call_b"
        # No null pollution anywhere
        for m in out:
            assert None not in m.values()

    def test_model_dump_drops_unset_fc_fields_when_nested(self):
        """Plain messages must not leak None FC fields, even when dumped via a
        parent model — the path that broke orchestrator output."""
        from pydantic import BaseModel

        class _Parent(BaseModel):
            msg: Message

        data = _Parent(msg=Message(role=MessageRole.USER, content="hi")).model_dump()
        assert data == {"msg": {"content": "hi", "role": MessageRole.USER, "metadata": None}}
