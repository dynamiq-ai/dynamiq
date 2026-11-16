import base64
import json
from io import BytesIO
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.tools.python import PythonInputSchema
from dynamiq.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.telemetry import content
from dynamiq.telemetry.types import CaptureSettings
from dynamiq.types import Document
from dynamiq.utils.utils import TRUNCATE_EMBEDDINGS_LIMIT, TRUNCATE_LIST_LIMIT

from .helpers import validate_messages

PNG_B64 = base64.b64encode(b"\x89PNG fake image bytes").decode()
PNG_DATA_URL = f"data:image/png;base64,{PNG_B64}"


def convert(messages, settings):
    return validate_messages(content.to_json(content.input_messages(messages, settings), settings), "input")


def test_text_messages_become_text_parts(settings):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi", "name": "alice"},
    ]

    assert convert(messages, settings) == [
        {"role": "system", "parts": [{"type": "text", "content": "Be brief."}]},
        {"role": "user", "parts": [{"type": "text", "content": "Hi"}], "name": "alice"},
    ]


def test_messages_formatted_by_a_prompt_convert(settings):
    """The real caller: LLM nodes report ``Prompt.format_messages()`` output, whose roles are enum members."""
    prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You answer about {{topic}}.", metadata={"k": "v"}),
            VisionMessage(
                role=MessageRole.USER,
                content=[
                    VisionMessageTextContent(text="What is this?"),
                    VisionMessageImageContent(image_url=VisionMessageImageURL(url="https://example.com/cat.png")),
                ],
            ),
        ]
    )

    assert convert(prompt.format_messages(topic="cats"), settings) == [
        {"role": "system", "parts": [{"type": "text", "content": "You answer about cats."}]},
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "What is this?"},
                {"type": "uri", "modality": "image", "uri": "https://example.com/cat.png"},
            ],
        },
    ]


def test_fields_outside_the_request_are_dropped(settings):
    messages = [{"role": "assistant", "content": "x", "static": True, "metadata": {"a": 1}, "type": "message"}]

    assert convert(messages, settings) == [{"role": "assistant", "parts": [{"type": "text", "content": "x"}]}]


def test_multi_part_content(settings):
    messages = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]

    assert convert(messages, settings)[0]["parts"] == [
        {"type": "text", "content": "a"},
        {"type": "text", "content": "b"},
    ]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"q": "weather", "days": 3}', {"q": "weather", "days": 3}),
        ({"q": "weather"}, {"q": "weather"}),
        ('{"q": "weath', '{"q": "weath'),
    ],
    ids=["json-string", "parsed", "invalid-json"],
)
def test_tool_calls_become_tool_call_parts(settings, arguments, expected):
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": arguments}}
            ],
        }
    ]

    assert convert(messages, settings) == [
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call_1", "name": "search", "arguments": expected}],
        }
    ]


def test_assistant_text_next_to_tool_calls_is_kept(settings):
    messages = [
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        }
    ]

    assert [part["type"] for part in convert(messages, settings)[0]["parts"]] == ["text", "tool_call"]


def test_tool_message_becomes_tool_call_response(settings):
    messages = [{"role": "tool", "content": "sunny", "tool_call_id": "call_1", "name": "search"}]

    assert convert(messages, settings) == [
        {
            "role": "tool",
            "parts": [{"type": "tool_call_response", "id": "call_1", "response": "sunny"}],
            "name": "search",
        }
    ]


def test_inline_image_without_binary_capture_is_a_sized_placeholder(settings):
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]}]

    [message] = convert(messages, settings)

    assert message["parts"] == [
        {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "", "size": len(PNG_B64)}
    ]


def test_inline_image_with_binary_capture_keeps_the_data():
    settings = CaptureSettings(binary_content_enabled=True)
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}]}]

    [message] = convert(messages, settings)

    assert message["parts"] == [{"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG_B64}]


def test_inline_file_is_a_blob_with_its_modality(settings):
    pdf = base64.b64encode(b"%PDF-1.4").decode()
    messages = [
        {"role": "user", "content": [{"type": "file", "file": {"file_data": f"data:application/pdf;base64,{pdf}"}}]}
    ]

    [part] = convert(messages, settings)[0]["parts"]

    assert part == {"type": "blob", "modality": "document", "mime_type": "application/pdf", "content": "", "size": 12}


def test_file_by_id_is_a_file_part(settings):
    messages = [{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-123"}}]}]

    assert convert(messages, settings)[0]["parts"] == [{"type": "file", "modality": "document", "file_id": "file-123"}]


def test_unknown_part_is_kept_unchanged(settings):
    part = {"type": "refusal", "refusal": "I can't help with that."}

    assert convert([{"role": "assistant", "content": [part]}], settings)[0]["parts"] == [part]


def test_every_message_of_a_long_conversation_is_kept(settings):
    """The 50-item list cut of traces does not apply to the message list: a span carries the whole request."""
    messages = [{"role": "user", "content": str(i)} for i in range(TRUNCATE_LIST_LIMIT + 10)]

    assert len(convert(messages, settings)) == TRUNCATE_LIST_LIMIT + 10


def test_output_message_has_finish_reason_and_tool_calls(settings):
    output = {
        "content": "Calling.",
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"x": 1}}}],
    }

    value = content.to_json([content.output_message(output, "tool_call", settings)], settings)

    assert validate_messages(value, "output") == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Calling."},
                {"type": "tool_call", "id": "c1", "name": "f", "arguments": {"x": 1}},
            ],
            "finish_reason": "tool_call",
        }
    ]


def test_redacted_keys_are_masked_at_any_depth(settings):
    value = {
        "input": "q",
        "tool_params": {"by_name": {"mcp": {"mcp_http_headers": {"Authorization": "Bearer s3cr3t"}}}},
    }

    encoded = content.to_json(value, settings)

    assert "s3cr3t" not in encoded
    assert json.loads(encoded)["tool_params"]["by_name"]["mcp"]["mcp_http_headers"] == "***"


def test_redacted_keys_inside_tool_call_arguments_are_masked(settings):
    arguments = json.dumps({"mcp_http_headers": {"X-Key": "s3cr3t"}})
    messages = [{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "mcp", "arguments": arguments}}]}]

    assert "s3cr3t" not in content.to_json(content.input_messages(messages, settings), settings)


def test_lists_are_cut_to_the_trace_limit(settings):
    encoded = content.to_json({"items": list(range(TRUNCATE_LIST_LIMIT * 2))}, settings)

    assert json.loads(encoded)["items"] == list(range(TRUNCATE_LIST_LIMIT))


def test_document_embeddings_are_cut_to_the_trace_limit(settings):
    documents = [Document(id="d1", content="text", embedding=[0.5] * 300)]

    [document] = json.loads(content.to_json({"documents": documents}, settings))["documents"]

    assert len(document["embedding"]) == TRUNCATE_EMBEDDINGS_LIMIT


def test_content_max_length_cuts_strings_and_keeps_json_valid():
    settings = CaptureSettings(content_max_length=5)
    value = {"a": "x" * 100, "nested": [{"b": "y" * 100}], "n": 12345678}

    decoded = json.loads(content.to_json(value, settings))

    assert decoded == {"a": "xxxxx", "nested": [{"b": "yyyyy"}], "n": 12345678}


def test_content_max_length_cuts_message_content_but_not_its_structure():
    settings = CaptureSettings(content_max_length=3)
    messages = [
        {"role": "assistant", "content": "abcdef", "tool_calls": [{"id": "call_long", "function": {"name": "search"}}]},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_long"},
    ]

    assert convert(messages, settings) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "abc"}, {"type": "tool_call", "id": "call_long", "name": "search"}],
        },
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_long", "response": "sun"}]},
    ]


@pytest.mark.parametrize("value", [b"\x00\x01binary", BytesIO(b"\x00\x01binary")], ids=["bytes", "bytesio"])
def test_binary_values_are_replaced_without_binary_capture(settings, value):
    encoded = content.to_json({"file": value}, settings)

    assert json.loads(encoded)["file"] == "<binary omitted: 8 bytes>"


def test_data_urls_in_content_are_replaced_without_binary_capture(settings):
    encoded = content.to_json({"image": PNG_DATA_URL}, settings)

    assert PNG_B64 not in encoded


def test_data_urls_in_content_are_kept_with_binary_capture():
    encoded = content.to_json({"image": PNG_DATA_URL}, CaptureSettings(binary_content_enabled=True))

    assert json.loads(encoded) == {"image": PNG_DATA_URL}


def test_tool_definitions_take_the_function_fields():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web.",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                "strict": True,
            },
        }
    ]

    assert content.tool_definitions(tools) == [
        {
            "type": "function",
            "name": "search",
            "description": "Search the web.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }
    ]


def test_retrieval_documents_keep_id_and_score():
    documents = [Document(id="a", content="x", score=0.9), {"id": "b", "score": 0.1, "content": "y"}]

    assert content.retrieval_documents(documents) == [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.1}]


class _Attachment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "a.png"
    data: Any = None
    files: list[Any] = Field(default_factory=list)


class _DictSubclass(dict):
    pass


_BLOB = b"\x89PNG" + b"SECRETBYTES" * 200
_BLOB_B64 = base64.b64encode(_BLOB).decode()


@pytest.mark.parametrize(
    "value",
    [
        {"holder": _Attachment(data=_BLOB)},
        {"holder": _Attachment(data=BytesIO(_BLOB))},
        {"holder": _Attachment(files=[BytesIO(_BLOB)])},
        _Attachment(data=BytesIO(_BLOB)),
        {"dep": RunnableResult(status=RunnableStatus.SUCCESS, output={"file": BytesIO(_BLOB)})},
        RunnableResult(status=RunnableStatus.SUCCESS, output={"file": _BLOB}),
        {"set": {_BLOB}},
        {"frozen": frozenset({_BLOB})},
        {"subclass": _DictSubclass(file=_BLOB)},
        {"documents": [Document(content="x", metadata={"raw": _BLOB})]},
        {"python_input": PythonInputSchema(file=_BLOB)},
    ],
    ids=[
        "model-bytes",
        "model-bytesio",
        "model-list-of-bytesio",
        "top-level-model",
        "runnable-result",
        "top-level-runnable-result",
        "set",
        "frozenset",
        "dict-subclass",
        "document-metadata",
        "python-input-schema",
    ],
)
def test_binary_nested_in_any_container_is_replaced_without_binary_capture(settings, value):
    encoded = content.to_json(value, settings)

    assert "SECRETBYTES" not in encoded
    assert _BLOB_B64[:40] not in encoded
    assert "SerializationIterator" not in encoded
    assert f"<binary omitted: {len(_BLOB)} bytes>" in encoded


def test_models_without_binary_are_recorded_as_before(settings):
    document = Document(id="d1", content="text", metadata={"k": "v"}, embedding=[0.5] * 300)
    result = RunnableResult(status=RunnableStatus.SUCCESS, input={"q": "x"}, output={"documents": [document]})

    decoded = json.loads(content.to_json({"dep": result}, settings))

    assert decoded["dep"]["status"] == "success"
    assert decoded["dep"]["output"]["documents"][0]["metadata"] == {"k": "v"}


def test_binary_nested_in_a_model_is_kept_with_binary_capture():
    encoded = content.to_json({"holder": _Attachment(data=_BLOB)}, CaptureSettings(binary_content_enabled=True))

    assert "<binary omitted" not in encoded
    assert json.loads(encoded)["holder"]["name"] == "a.png"


def test_redaction_applies_inside_a_model_that_holds_binary(settings):
    holder = _Attachment(data=_BLOB, files=[{"mcp_http_headers": {"Authorization": "Bearer s3cr3t"}}])

    encoded = content.to_json({"holder": holder}, settings)

    assert "s3cr3t" not in encoded
    assert json.loads(encoded)["holder"]["data"] == f"<binary omitted: {len(_BLOB)} bytes>"


def test_retrieval_documents_match_the_schema_when_id_or_score_is_missing():
    documents = [
        Document(id="a", content="x", score=0.9),
        {"content": "no id, no score"},
        {"id": 7, "score": 0.2},
        {"id": None, "score": "high"},
        {"id": "nan", "score": float("nan")},
    ]

    assert content.retrieval_documents(documents) == [
        {"id": "a", "score": 0.9},
        {"id": "1"},
        {"id": "7", "score": 0.2},
        {"id": "3"},
        {"id": "nan"},
    ]
