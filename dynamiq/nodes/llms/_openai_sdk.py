"""OpenAI SDK transport for the OpenAI LLM node, used in place of LiteLLM when `OpenAI.use_native_sdk` is on.

LiteLLM re-wraps every streamed chunk in its own objects and rebuilds the whole response at the end of a stream for its
own logging. That costs about 0.4 ms of event-loop CPU per chunk, several times what the SDK spends on it, and caps how
many concurrent streams one process can serve. This module sends the request LiteLLM would send through the OpenAI SDK
and keeps everything around it the same: SDK errors are re-raised as the LiteLLM exceptions the retry and fallback
logic classify, streamed chunks get the shape LiteLLM gives them, and the assembled response has the attributes the
node reads.

`build_request` returns None for a request using anything it does not translate, and the node then sends that request
through LiteLLM.
"""

import copy
import datetime
import inspect
import types
import typing
from enum import Enum
from functools import cache
from typing import Any

import httpx
import openai
import pydantic
from openai import AsyncOpenAI, BadRequestError, OpenAI, OpenAIError, UnprocessableEntityError
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from pydantic_core import PydanticUndefined

from dynamiq.utils.logger import logger

# Chat Completions arguments, as the SDK names them. Ones the installed SDK predates travel in `extra_body`.
_REQUEST_PARAMS = frozenset(
    {
        "audio",
        "extra_body",
        "extra_headers",
        "extra_query",
        "frequency_penalty",
        "function_call",
        "functions",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "modalities",
        "model",
        "n",
        "parallel_tool_calls",
        "prediction",
        "presence_penalty",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning_effort",
        "response_format",
        "safety_identifier",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "timeout",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "user",
        "verbosity",
        "web_search_options",
    }
)

# LiteLLM's own routing and filtering params. The SDK client already holds the endpoint and key, and `build_request`
# applies LiteLLM's per-model param filter itself, so none of them goes into the request.
_LITELLM_PARAMS = frozenset({"api_base", "api_key", "client", "drop_params", "allowed_openai_params"})

# What travels with the request rather than in the model's params, so LiteLLM's param mapping never sees it.
_REQUEST_ENVELOPE = frozenset({"model", "messages", "extra_headers", "extra_query", "extra_body", "timeout"})

# What the SDK can raise from a request or mid-stream; anything else is not a transport failure.
_TRANSPORT_ERRORS = (OpenAIError, httpx.HTTPError)


@cache
def _sdk_create_params() -> frozenset[str]:
    """The keyword arguments the installed SDK's `chat.completions.create` accepts."""
    return frozenset(inspect.signature(Completions.create).parameters) - {"self"}


def _has_cache_control(messages: Any) -> bool:
    """Whether any message carries a `cache_control` breakpoint, which LiteLLM strips for OpenAI and the API rejects."""
    for message in messages:
        if not isinstance(message, dict):
            return True
        if "cache_control" in message:
            return True
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(part, dict) and "cache_control" in part for part in content):
            return True
    return False


def _litellm_model_params(model: str, params: dict[str, Any], allowed_openai_params: list[str] | None) -> dict:
    """`params` as LiteLLM sends them to `model` with `drop_params`, by LiteLLM's own per-model mapping.

    That mapping drops what the model rejects (penalties, top_p and logprobs on reasoning models, a temperature other
    than 1 on gpt-5) and renames what it must, so the request carries what the LiteLLM path would.
    """
    import litellm

    return litellm.get_optional_params(
        model=model,
        custom_llm_provider="openai",
        drop_params=True,
        allowed_openai_params=allowed_openai_params,
        **params,
    )


def build_request(params: dict[str, Any], model_prefix: str | None) -> dict[str, Any] | None:
    """The SDK arguments for LiteLLM completion `params`, or None if this transport does not translate them.

    Params set to None, and `stream` when off, are left out, as LiteLLM leaves them out, and the model's params go
    through LiteLLM's per-model filter, as `drop_params` has LiteLLM do.
    """
    request: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or key in _LITELLM_PARAMS or (key == "stream" and not value):
            continue
        if key not in _REQUEST_PARAMS:
            logger.debug("OpenAI SDK transport: '%s' is not an SDK param; sending the request through LiteLLM.", key)
            return None
        request[key] = value.value if isinstance(value, Enum) else value

    # A pydantic class as response_format is LiteLLM's conversion to make, and cache breakpoints its to strip.
    if not isinstance(request.get("response_format", {}), dict) or _has_cache_control(request.get("messages", ())):
        return None
    if model_prefix:
        request["model"] = request["model"].removeprefix(model_prefix)

    envelope = {key: value for key, value in request.items() if key in _REQUEST_ENVELOPE}
    model_params = {key: value for key, value in request.items() if key not in _REQUEST_ENVELOPE}
    mapped = _litellm_model_params(request["model"], model_params, params.get("allowed_openai_params"))
    if not mapped.get("stream"):  # the mapping fills in its default, which LiteLLM leaves out of the request
        mapped.pop("stream", None)
    if extra_body := (mapped.pop("extra_body", None) or {}) | envelope.get("extra_body", {}):
        envelope["extra_body"] = extra_body
    request = envelope | mapped

    known = _sdk_create_params() | {"extra_body"}
    unknown = {key: request.pop(key) for key in list(request) if key not in known}
    if unknown:
        request["extra_body"] = unknown | request.get("extra_body", {})
    return request


# Endpoints that rejected the `stream_options` this transport adds: their streams go out as asked, and usage is counted.
_ENDPOINTS_WITHOUT_STREAM_OPTIONS: set[str] = set()


def _ask_for_usage(request: dict[str, Any], endpoint: str) -> tuple[dict[str, Any], bool]:
    """`request` asking its stream to report usage, and whether the caller asked to have usage streamed.

    Usage arrives only on request, in a last chunk with no choices; LiteLLM otherwise counts it with a tokenizer. The
    request comes back unchanged (the same object) when it needs no option added or its endpoint rejected the option.
    """
    stream_options = request.get("stream_options") or {}
    emit_usage = bool(stream_options.get("include_usage"))
    if not request.get("stream") or emit_usage or endpoint in _ENDPOINTS_WITHOUT_STREAM_OPTIONS:
        return request, emit_usage
    return request | {"stream_options": stream_options | {"include_usage": True}}, emit_usage


def _rejects_stream_options(exc: Exception) -> bool:
    """Whether an endpoint refused the request over `stream_options`, which some OpenAI-compatible servers lack."""
    return isinstance(exc, (BadRequestError, UnprocessableEntityError)) and "stream_options" in str(exc).lower()


def _as_litellm_error(exc: Exception, model: str) -> Exception:
    """The exception LiteLLM raises for this SDK or httpx error: the types retries and fallbacks classify."""
    import litellm

    try:
        litellm.exception_type(model=model, original_exception=exc, custom_llm_provider="openai")
    except Exception as mapped:  # exception_type raises its result
        return mapped
    return exc


# A field `_chunk_dict` leaves to the SDK: one that may hold one of several models, or values it converts in containers.
_NEEDS_SDK = object()


def _field_kind(annotation: Any) -> Any:
    """How `_as_model_dump` shapes a field's value: None to keep it, "float", ("model", cls), ("list", cls), or
    `_NEEDS_SDK`."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        kinds = [kind for kind in map(_field_kind, typing.get_args(annotation)) if kind is not None]
        if not kinds:
            return None
        return kinds[0] if len(kinds) == 1 else _NEEDS_SDK
    if origin is list:
        args = typing.get_args(annotation)
        item = _field_kind(args[0]) if args else None
        if item is None:
            return None
        return ("list", item[1]) if isinstance(item, tuple) and item[0] == "model" else _NEEDS_SDK
    if annotation is float:
        return "float"
    if annotation in (datetime.datetime, datetime.date):
        return _NEEDS_SDK
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        return ("model", annotation)
    if any(_field_kind(arg) is not None for arg in typing.get_args(annotation)):
        return _NEEDS_SDK
    return None


@cache
def _fields(model: type[pydantic.BaseModel]) -> tuple[tuple[tuple[str, str, Any, Any], ...], frozenset[str]]:
    """Each field of `model` as (JSON key, name, default, kind), and the set of JSON keys."""
    fields = tuple(
        (
            field.alias or name,
            name,
            None if field.default is PydanticUndefined else field.default,
            _field_kind(field.annotation),
        )
        for name, field in model.model_fields.items()
    )
    return fields, frozenset(key for key, *_ in fields)


class _NeedsSDK(Exception):
    """A value `_as_model_dump` does not shape itself."""


def _as_model_dump(model: type[pydantic.BaseModel], data: dict) -> dict:
    """`data` as `model.construct(**data).model_dump()` returns it, without building the model.

    Values are kept as the JSON has them, as the SDK keeps values of their declared types, except an integral number in
    a float field, which the SDK turns into a float.
    """
    fields, keys = _fields(model)
    out = {}
    for key, name, default, kind in fields:
        if key not in data:
            out[name] = default
            continue
        value = data[key]
        if value is not None and kind is not None:
            if kind is _NEEDS_SDK:
                raise _NeedsSDK
            if kind == "float":
                if isinstance(value, int) and float(value) == value:
                    value = float(value)
            elif kind[0] == "model" and isinstance(value, dict):
                value = _as_model_dump(kind[1], value)
            elif kind[0] == "list" and isinstance(value, list):
                value = [_as_model_dump(kind[1], item) if isinstance(item, dict) else item for item in value]
        out[name] = value
    for key, value in data.items():
        if key not in keys:
            out[key] = value
    return out


def _chunk_dict(data: dict) -> dict:
    """A chunk's JSON as `ChatCompletionChunk.model_dump()` returns it: every field, extra keys after them.

    Building the SDK's pydantic models for a chunk and dumping them again costs about 25 µs; filling in the dict
    costs about 1 µs. A chunk with a field the SDK has to tell apart, such as `moderation`, goes through the SDK.
    """
    try:
        return _as_model_dump(ChatCompletionChunk, data)
    except _NeedsSDK:
        return ChatCompletionChunk.construct(**data).model_dump()


def _empty_choice() -> dict:
    """The one choice LiteLLM streams a usage chunk with."""
    return _as_model_dump(ChunkChoice, {"index": 0, "delta": {}, "finish_reason": None})


class _Assembly:
    """A stream's response, merged from its chunks as LiteLLM merges them, as each chunk arrives.

    It keeps strings and copies, so a stream callback editing its chunk cannot change the response. Content joins
    the content deltas and stays None when none had any. A tool call keeps its last id and name and joins its argument
    deltas. Only the first choice is assembled, the only one the node reads.
    """

    def __init__(self):
        self._first: tuple[Any, Any, Any, Any] | None = None
        self._content: list[str] = []
        self._has_content = False
        self._refusal: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason = None
        self._usage: dict | None = None

    def add(self, chunk: dict) -> None:
        if self._first is None:
            self._first = (chunk["id"], chunk["created"], chunk["model"], chunk["system_fingerprint"])
        if chunk["usage"]:
            self._usage = copy.deepcopy(chunk["usage"])
        for choice in chunk["choices"] or ():
            if choice["index"] != 0:
                continue
            delta = choice["delta"] or {}
            if delta.get("content") is not None:
                self._has_content = True
                self._content.append(delta["content"])
            if delta.get("refusal"):
                self._refusal.append(delta["refusal"])
            for tool_call in delta.get("tool_calls") or ():
                call = self._tool_calls.setdefault(
                    tool_call["index"], {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                )
                call["id"] = tool_call["id"] or call["id"]
                if function := tool_call["function"]:
                    call["function"]["name"] = function["name"] or call["function"]["name"]
                    call["function"]["arguments"] += function["arguments"] or ""
            self._finish_reason = choice["finish_reason"] or self._finish_reason

    def response(self, messages: list[dict], model: str) -> ChatCompletion:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self._content) if self._has_content else None,
        }
        if self._refusal:
            message["refusal"] = "".join(self._refusal)
        if self._tool_calls:
            message["tool_calls"] = [self._tool_calls[index] for index in sorted(self._tool_calls)]

        usage_data = self._usage
        if usage_data is None:
            completion_text = (message["content"] or "") + "".join(
                call["function"]["arguments"] for call in self._tool_calls.values()
            )
            usage_data = _estimate_usage(model, messages, completion_text)

        chunk_id, created, chunk_model, system_fingerprint = self._first or ("", 0, model, None)
        return ChatCompletion.construct(
            id=chunk_id,
            created=created,
            model=chunk_model,
            object="chat.completion",
            system_fingerprint=system_fingerprint,
            choices=[
                {"index": 0, "message": message, "finish_reason": self._finish_reason or "stop", "logprobs": None}
            ],
            usage=usage_data,
        )


class _ChunkShaper:
    """Gives chunks the shape LiteLLM gives them, which the stream callbacks read, and assembles the response.

    The SDK hands over each chunk's JSON as a dict, and it stays a dict: the node passes it to the callbacks, which
    read `ChatCompletionChunk.model_dump()` before.
    """

    def __init__(self, stream: Any, model: str, emit_usage: bool):
        self._stream = stream
        self._model = model
        self._emit_usage = emit_usage
        self._yielded = False
        self._assembly = _Assembly()

    def _shape(self, data: Any) -> dict | None:
        """`data` as LiteLLM streams it, or None for a usage chunk the caller did not ask to have streamed."""
        chunk = _chunk_dict(data) if isinstance(data, dict) else data.model_dump()
        self._assembly.add(chunk)
        if not chunk["choices"]:
            if not self._emit_usage and self._yielded:
                # Usage was asked for only to assemble the response, which has it now.
                return None
            # LiteLLM streams the usage chunk with one empty choice, and consumers read choices[0].
            chunk["choices"] = [_empty_choice()]
        else:
            for choice in chunk["choices"]:
                for tool_call in (choice["delta"] or {}).get("tool_calls") or ():
                    # The API types only a call's first delta; LiteLLM types all of them, and parsers check it.
                    if tool_call["type"] is None:
                        tool_call["type"] = "function"
        self._yielded = True
        return chunk

    def build_response(self, messages: list[dict], model: str) -> ChatCompletion:
        """The response an unstreamed call returns, from the chunks streamed so far."""
        return self._assembly.response(messages, model)


class Stream(_ChunkShaper):
    """A sync SDK stream yielding LiteLLM-shaped chunk dicts and raising LiteLLM's exceptions."""

    def __iter__(self):
        try:
            for data in self._stream:
                if (chunk := self._shape(data)) is not None:
                    yield chunk
        except _TRANSPORT_ERRORS as e:
            raise _as_litellm_error(e, self._model) from e

    def close(self) -> None:
        self._stream.close()


class AsyncStream(_ChunkShaper):
    """An async SDK stream yielding LiteLLM-shaped chunk dicts and raising LiteLLM's exceptions."""

    async def __aiter__(self):
        try:
            async for data in self._stream:
                if (chunk := self._shape(data)) is not None:
                    yield chunk
        except _TRANSPORT_ERRORS as e:
            raise _as_litellm_error(e, self._model) from e

    async def aclose(self) -> None:
        await self._stream.close()


def _send(client: OpenAI, request: dict[str, Any]) -> Any:
    """The response to `request`; a stream yields each chunk's JSON as a dict, not as the SDK's model."""
    if not request.get("stream"):
        return client.chat.completions.create(**request)
    return client.chat.completions.with_raw_response.create(**request).parse(to=openai.Stream[object])


async def _asend(client: AsyncOpenAI, request: dict[str, Any]) -> Any:
    """Async `_send`."""
    if not request.get("stream"):
        return await client.chat.completions.create(**request)
    response = await client.chat.completions.with_raw_response.create(**request)
    return response.parse(to=openai.AsyncStream[object])


def create(client: OpenAI, request: dict[str, Any]) -> ChatCompletion | Stream:
    """Send `request` (from `build_request`) and return the response, or the stream of a streamed request.

    A stream whose endpoint rejects the usage option added to it is sent again as asked, once per endpoint.
    """
    endpoint = str(client.base_url)
    sent, emit_usage = _ask_for_usage(request, endpoint)
    try:
        response = _send(client, sent)
    except _TRANSPORT_ERRORS as e:
        if sent is request or not _rejects_stream_options(e):
            raise _as_litellm_error(e, request["model"]) from e
        _ENDPOINTS_WITHOUT_STREAM_OPTIONS.add(endpoint)
        try:
            response = _send(client, request)
        except _TRANSPORT_ERRORS as retry_error:
            raise _as_litellm_error(retry_error, request["model"]) from retry_error
    return Stream(response, request["model"], emit_usage) if request.get("stream") else response


async def acreate(client: AsyncOpenAI, request: dict[str, Any]) -> ChatCompletion | AsyncStream:
    """Async `create`."""
    endpoint = str(client.base_url)
    sent, emit_usage = _ask_for_usage(request, endpoint)
    try:
        response = await _asend(client, sent)
    except _TRANSPORT_ERRORS as e:
        if sent is request or not _rejects_stream_options(e):
            raise _as_litellm_error(e, request["model"]) from e
        _ENDPOINTS_WITHOUT_STREAM_OPTIONS.add(endpoint)
        try:
            response = await _asend(client, request)
        except _TRANSPORT_ERRORS as retry_error:
            raise _as_litellm_error(retry_error, request["model"]) from retry_error
    return AsyncStream(response, request["model"], emit_usage) if request.get("stream") else response


def _estimate_usage(model: str, messages: list[dict], completion_text: str) -> dict[str, int] | None:
    """Token counts for a stream that reported no usage, counted as LiteLLM counts them."""
    from litellm import token_counter

    try:
        prompt_tokens = token_counter(model=model, messages=messages)
        completion_tokens = token_counter(model=model, text=completion_text) if completion_text else 0
    except Exception:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
