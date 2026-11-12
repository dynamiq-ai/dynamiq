"""Content recorded on spans: redaction, binary placeholders, truncation, JSON encoding and message parts.

Everything here passes through the run-based trace redaction (``to_dict(for_tracing=True)``,
``format_value(for_tracing=True)``, ``TRACING_REDACTED_KEYS``) before it is encoded. Messages follow the input and
output message JSON schemas of the GenAI semantic conventions v1.41.1.
"""

import json
import math
import mimetypes
import re
import warnings
from collections.abc import Mapping
from io import BytesIO
from types import NoneType
from typing import Any

import orjson
from pydantic import BaseModel

from dynamiq.telemetry import attributes as attrs
from dynamiq.telemetry.types import CaptureSettings
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import UntruncatedList, format_value, orjson_encode, redact_tracing_keys

_SCALAR_TYPES = frozenset({str, int, float, bool, NoneType})
_BINARY_TYPES = (bytes, bytearray, BytesIO)

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;,]*)(?P<params>(?:;[^;,]*)*?);base64,(?P<data>.*)$", re.DOTALL)

# Keys whose values identify rather than carry content; DYNAMIQ_TRACING_CONTENT_MAX_LENGTH leaves them whole.
_STRUCTURAL_KEYS = frozenset({"type", "role", "id", "name", "finish_reason", "modality", "mime_type", "file_id"})


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def binary_placeholder(size: int, name: str | None = None) -> str:
    """Text that stands in for binary content left out of a span."""
    return f"<binary omitted: {name}, {size} bytes>" if name else f"<binary omitted: {size} bytes>"


def parse_data_url(value: Any) -> tuple[str | None, str] | None:
    """``(mime type, base64 data)`` of a ``data:...;base64,`` URL, or None for anything else."""
    if not isinstance(value, str) or not value.startswith("data:"):
        return None
    match = _DATA_URL_RE.match(value)
    if match is None:
        return None
    return match.group("mime") or None, match.group("data")


def _holds_binary(value: Any, seen: set[int]) -> bool:
    """Whether bytes or a file object sits anywhere inside ``value``, models and their extra fields included."""
    if type(value) in _SCALAR_TYPES:
        return False
    if isinstance(value, _BINARY_TYPES):
        return True
    if isinstance(value, Mapping):
        return any(_holds_binary(item, seen) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_binary(item, seen) for item in value)
    if isinstance(value, BaseModel):
        # A model can refer back to itself through its fields.
        if id(value) in seen:
            return False
        seen.add(id(value))
        return any(_holds_binary(item, seen) for _, item in value)
    return False


def _dump_model(model: BaseModel) -> Any:
    """``model`` as ``format_value(for_tracing=True)`` dumps it."""
    if hasattr(model, "to_dict"):
        try:
            return model.to_dict(for_tracing=True)
        except TypeError:
            return model.to_dict()
    return model.model_dump()


def _model_without_binary(model: BaseModel) -> Any:
    """The dump of ``model`` with its binary replaced before its own ``to_dict()`` encodes it as base64.

    The fields that hold binary are replaced on a copy, so ``to_dict()`` still applies its exclusions and redaction.
    """
    update = {name: _strip_binary(item) for name, item in model if _holds_binary(item, set())}
    copy = model.model_copy(update=update)
    try:
        # The placeholders do not match the fields' declared types.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dumped = _dump_model(copy)
    except Exception as e:
        # Fall back to the original: recording its binary is better than skipping the exclusions of its to_dict().
        logger.debug(f"Tracing: could not leave binary out of {type(model).__name__}: {e}")
        return _dump_model(model)
    return _strip_binary(dumped)


def _strip_binary(value: Any) -> Any:
    """Replace bytes and file objects at any depth before ``format_value`` turns them into base64.

    Containers become plain dicts and lists, so the later passes see every value; models that hold binary are dumped
    with it replaced, and models that hold none are left for ``format_value``.
    """
    if type(value) in _SCALAR_TYPES:
        return value
    if isinstance(value, (bytes, bytearray)):
        return binary_placeholder(len(value))
    if isinstance(value, BytesIO):
        return binary_placeholder(len(value.getbuffer()), getattr(value, "name", None))
    if isinstance(value, UntruncatedList):
        return UntruncatedList(_strip_binary(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _strip_binary(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_strip_binary(item) for item in value)
    if isinstance(value, (list, set, frozenset)):
        return [_strip_binary(item) for item in value]
    if isinstance(value, BaseModel) and _holds_binary(value, set()):
        return _model_without_binary(value)
    return value


def _strip_data_urls(value: Any) -> Any:
    """Replace inline base64 data URLs in an already formatted structure."""
    if isinstance(value, str):
        parsed = parse_data_url(value)
        if parsed is None:
            return value
        return binary_placeholder(len(parsed[1]))
    if isinstance(value, dict):
        return {key: _strip_data_urls(item) for key, item in value.items()}
    if isinstance(value, list):
        return type(value)(_strip_data_urls(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_strip_data_urls(item) for item in value)
    return value


def truncate_strings(value: Any, max_length: int) -> Any:
    """Cut every string inside ``value`` to ``max_length`` characters, keeping the structure.

    Values of the message schema's structural keys (part types, roles, ids, names) are kept whole: a cut one would
    make a message invalid rather than shorter.
    """
    if isinstance(value, str):
        return value[:max_length] if len(value) > max_length else value
    if isinstance(value, dict):
        return {
            key: item if key in _STRUCTURAL_KEYS and isinstance(item, str) else truncate_strings(item, max_length)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return type(value)(truncate_strings(item, max_length) for item in value)
    if isinstance(value, tuple):
        return tuple(truncate_strings(item, max_length) for item in value)
    return value


def prepare(value: Any, settings: CaptureSettings) -> Any:
    """A JSON-ready copy of ``value`` with the redaction, binary and length rules of the spans applied."""
    if isinstance(value, BaseException):
        value = {"message": str(value), "type": type(value).__name__}
    elif not settings.binary_content_enabled and isinstance(value, BaseModel) and _holds_binary(value, set()):
        value = _model_without_binary(value)
    elif hasattr(value, "to_dict") and not isinstance(value, type):
        try:
            value = value.to_dict(for_tracing=True)
        except TypeError:
            pass
    if not settings.binary_content_enabled:
        value = _strip_binary(value)
    value = format_value(value, for_tracing=True)
    # `format_value` returns structures from `to_dict` and `model_dump` without walking them.
    value = redact_tracing_keys(value)
    if not settings.binary_content_enabled:
        value = _strip_data_urls(value)
    if settings.content_max_length:
        value = truncate_strings(value, settings.content_max_length)
    return value


def dumps(value: Any) -> str:
    """JSON text of an already prepared value."""
    try:
        return orjson.dumps(value, default=orjson_encode, option=orjson.OPT_NON_STR_KEYS).decode()
    except (TypeError, orjson.JSONEncodeError):
        return json.dumps(value, default=str, ensure_ascii=False)


def to_json(value: Any, settings: CaptureSettings) -> str:
    """Redacted, length-limited JSON text of ``value`` for a content attribute."""
    return dumps(prepare(value, settings))


def _parse_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            return json.loads(arguments, strict=False)
        except ValueError:
            return arguments
    return arguments


def _modality_of(mime_type: str | None, default: str = attrs.MODALITY_DOCUMENT) -> str:
    if mime_type:
        major = mime_type.split("/", 1)[0]
        if major in (attrs.MODALITY_IMAGE, attrs.MODALITY_VIDEO, attrs.MODALITY_AUDIO):
            return major
    return default


def _blob_part(mime_type: str | None, data: str, modality: str, settings: CaptureSettings) -> dict[str, Any]:
    part: dict[str, Any] = {"type": attrs.PART_BLOB, "modality": modality}
    if mime_type:
        part["mime_type"] = mime_type
    if settings.binary_content_enabled:
        part["content"] = data
    else:
        part["content"] = ""
        part["size"] = len(data)
    return part


def _url_part(url: Any, default_modality: str, settings: CaptureSettings) -> dict[str, Any]:
    parsed = parse_data_url(url)
    if parsed is not None:
        mime_type, data = parsed
        return _blob_part(mime_type, data, _modality_of(mime_type, default_modality), settings)
    guessed, _ = mimetypes.guess_type(str(url))
    return {"type": attrs.PART_URI, "modality": _modality_of(guessed, default_modality), "uri": url}


def _content_part(part: Any, settings: CaptureSettings) -> dict[str, Any] | Any:
    """One part of a message ``content`` list, in the convention's part shape."""
    if isinstance(part, str):
        return {"type": attrs.PART_TEXT, "content": part}
    if hasattr(part, "model_dump"):
        part = part.model_dump()
    if not isinstance(part, dict):
        return part
    part_type = str(_enum_value(part.get("type")))
    if part_type == attrs.PART_TEXT and "text" in part:
        return {"type": attrs.PART_TEXT, "content": part["text"]}
    if part_type == "image_url":
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if isinstance(url, str):
            return _url_part(url, attrs.MODALITY_IMAGE, settings)
    if part_type == "file" and isinstance(part.get("file"), dict):
        file = part["file"]
        if file.get("file_id"):
            out = {"type": attrs.PART_FILE, "modality": attrs.MODALITY_DOCUMENT, "file_id": file["file_id"]}
            if file.get("format"):
                out["mime_type"] = file["format"]
            return out
        if isinstance(file.get("file_data"), str):
            return _url_part(file["file_data"], attrs.MODALITY_DOCUMENT, settings)
    if part_type == "input_audio" and isinstance(part.get("input_audio"), dict):
        audio = part["input_audio"]
        if isinstance(audio.get("data"), str):
            audio_format = audio.get("format")
            mime_type = f"audio/{audio_format}" if audio_format else None
            return _blob_part(mime_type, audio["data"], attrs.MODALITY_AUDIO, settings)
    return part


def _tool_call_part(call: Any) -> dict[str, Any]:
    if hasattr(call, "model_dump"):
        call = call.model_dump()
    if not isinstance(call, dict):
        return {"type": attrs.PART_TOOL_CALL, "name": str(call)}
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    part: dict[str, Any] = {"type": attrs.PART_TOOL_CALL}
    if call.get("id") is not None:
        part["id"] = call["id"]
    part["name"] = function.get("name") or ""
    if "arguments" in function:
        part["arguments"] = _parse_arguments(function["arguments"])
    return part


def message_parts(message: dict[str, Any], settings: CaptureSettings) -> dict[str, Any]:
    """One request message (OpenAI format, as LLM nodes send it) in the convention's message shape."""
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    role = _enum_value(message.get("role"))
    content = message.get("content")
    parts: list[Any] = []
    if role == attrs.ROLE_TOOL or message.get("tool_call_id") is not None:
        response = content
        if isinstance(content, list):
            response = [_content_part(part, settings) for part in content]
        part: dict[str, Any] = {"type": attrs.PART_TOOL_CALL_RESPONSE}
        if message.get("tool_call_id") is not None:
            part["id"] = message["tool_call_id"]
        part["response"] = response
        parts.append(part)
    elif isinstance(content, list):
        parts.extend(_content_part(part, settings) for part in content)
    elif content is not None and (content != "" or not message.get("tool_calls")):
        parts.append({"type": attrs.PART_TEXT, "content": content if isinstance(content, str) else str(content)})
    for call in message.get("tool_calls") or []:
        parts.append(_tool_call_part(call))

    out: dict[str, Any] = {"role": role if role is not None else attrs.ROLE_USER, "parts": UntruncatedList(parts)}
    if message.get("name") is not None:
        out["name"] = message["name"]
    return out


def input_messages(messages: Any, settings: CaptureSettings) -> list[dict[str, Any]]:
    """Request messages in the convention's input message shape, before redaction."""
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    return UntruncatedList(message_parts(message, settings) for message in messages if message is not None)


def text_message(role: str, text: Any, settings: CaptureSettings, finish_reason: str | None = None) -> dict[str, Any]:
    """A message with one text part; ``finish_reason`` makes it an output message."""
    if not isinstance(text, str):
        text = to_json(text, settings)
    message: dict[str, Any] = {"role": role, "parts": UntruncatedList([{"type": attrs.PART_TEXT, "content": text}])}
    if finish_reason is not None:
        message["finish_reason"] = finish_reason
    return message


def output_message(output: dict[str, Any], finish_reason: str, settings: CaptureSettings) -> dict[str, Any]:
    """The assistant message of an LLM node's ``execute()`` result (``{"content", "tool_calls"?}``)."""
    message = message_parts(
        {"role": attrs.ROLE_ASSISTANT, "content": output.get("content"), "tool_calls": output.get("tool_calls")},
        settings,
    )
    message["finish_reason"] = finish_reason
    return message


def tool_definitions(tools: Any) -> list[dict[str, Any]]:
    """Tools passed to a model, in the convention's tool definition shape."""
    definitions = UntruncatedList()
    for tool in tools or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else None
        if function is None and tool.get("type", attrs.TOOL_TYPE_FUNCTION) != attrs.TOOL_TYPE_FUNCTION:
            definitions.append(tool)
            continue
        source = function or tool
        definition: dict[str, Any] = {"type": attrs.TOOL_TYPE_FUNCTION, "name": source.get("name") or ""}
        if source.get("description") is not None:
            definition["description"] = source["description"]
        if source.get("parameters") is not None:
            definition["parameters"] = source["parameters"]
        definitions.append(definition)
    return definitions


def retrieval_documents(documents: Any) -> list[dict[str, Any]]:
    """``[{"id", "score"}]`` of retrieved documents.

    ``id`` is always a string: a document without one gets its position in the list. ``score`` is left out when the
    retriever reports none, rather than given a made-up number.
    """
    out = []
    for position, document in enumerate(documents or []):
        if isinstance(document, dict):
            doc_id, score = document.get("id"), document.get("score")
        else:
            doc_id, score = getattr(document, "id", None), getattr(document, "score", None)
        item: dict[str, Any] = {"id": str(position) if doc_id is None else str(doc_id)}
        # JSON has no NaN or infinity, so they are as unknown as a missing score.
        if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
            item["score"] = score
        out.append(item)
    return out
