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

import inspect
from enum import Enum
from functools import cache
from typing import Any

import httpx
from openai import AsyncOpenAI, BadRequestError, OpenAI, OpenAIError, UnprocessableEntityError
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

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


class _ChunkShaper:
    """Gives SDK chunks the shape LiteLLM gives them, which the stream callbacks read."""

    def __init__(self, stream: Any, model: str, emit_usage: bool):
        self._stream = stream
        self._model = model
        self._emit_usage = emit_usage
        self._last: ChatCompletionChunk | None = None

    def _shape(self, chunk: ChatCompletionChunk) -> ChatCompletionChunk | None:
        """`chunk` as LiteLLM streams it, or None for a usage chunk the caller did not ask to have streamed."""
        if not chunk.choices:
            if not self._emit_usage and self._last is not None:
                # Usage was asked for only to assemble the response: hand it over without streaming it.
                self._last.usage = chunk.usage
                return None
            # LiteLLM streams the usage chunk with one empty choice, and consumers read choices[0].
            chunk.choices = [ChunkChoice.construct(index=0, delta=ChoiceDelta.construct(), finish_reason=None)]
        else:
            for choice in chunk.choices:
                for tool_call in choice.delta.tool_calls or ():
                    # The API types only a call's first delta; LiteLLM types all of them, and parsers check it.
                    if tool_call.type is None:
                        tool_call.type = "function"
        self._last = chunk
        return chunk


class Stream(_ChunkShaper):
    """A sync SDK stream yielding LiteLLM-shaped chunks and raising LiteLLM's exceptions."""

    def __iter__(self):
        try:
            for chunk in self._stream:
                if (chunk := self._shape(chunk)) is not None:
                    yield chunk
        except _TRANSPORT_ERRORS as e:
            raise _as_litellm_error(e, self._model) from e

    def close(self) -> None:
        self._stream.close()


class AsyncStream(_ChunkShaper):
    """An async SDK stream yielding LiteLLM-shaped chunks and raising LiteLLM's exceptions."""

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                if (chunk := self._shape(chunk)) is not None:
                    yield chunk
        except _TRANSPORT_ERRORS as e:
            raise _as_litellm_error(e, self._model) from e

    async def aclose(self) -> None:
        await self._stream.close()


def create(client: OpenAI, request: dict[str, Any]) -> ChatCompletion | Stream:
    """Send `request` (from `build_request`) and return the response, or the stream of a streamed request.

    A stream whose endpoint rejects the usage option added to it is sent again as asked, once per endpoint.
    """
    endpoint = str(client.base_url)
    sent, emit_usage = _ask_for_usage(request, endpoint)
    try:
        response = client.chat.completions.create(**sent)
    except _TRANSPORT_ERRORS as e:
        if sent is request or not _rejects_stream_options(e):
            raise _as_litellm_error(e, request["model"]) from e
        _ENDPOINTS_WITHOUT_STREAM_OPTIONS.add(endpoint)
        try:
            response = client.chat.completions.create(**request)
        except _TRANSPORT_ERRORS as retry_error:
            raise _as_litellm_error(retry_error, request["model"]) from retry_error
    return Stream(response, request["model"], emit_usage) if request.get("stream") else response


async def acreate(client: AsyncOpenAI, request: dict[str, Any]) -> ChatCompletion | AsyncStream:
    """Async `create`."""
    endpoint = str(client.base_url)
    sent, emit_usage = _ask_for_usage(request, endpoint)
    try:
        response = await client.chat.completions.create(**sent)
    except _TRANSPORT_ERRORS as e:
        if sent is request or not _rejects_stream_options(e):
            raise _as_litellm_error(e, request["model"]) from e
        _ENDPOINTS_WITHOUT_STREAM_OPTIONS.add(endpoint)
        try:
            response = await client.chat.completions.create(**request)
        except _TRANSPORT_ERRORS as retry_error:
            raise _as_litellm_error(retry_error, request["model"]) from retry_error
    return AsyncStream(response, request["model"], emit_usage) if request.get("stream") else response


def is_sdk_stream(chunks: list) -> bool:
    """Whether `chunks` came from this transport rather than from LiteLLM."""
    return bool(chunks) and isinstance(chunks[0], ChatCompletionChunk)


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


def build_response(chunks: list[ChatCompletionChunk], messages: list[dict], model: str) -> ChatCompletion:
    """Assemble streamed chunks into the response an unstreamed call returns, merged as LiteLLM merges them.

    Content joins the content deltas and stays None when none had any. A tool call keeps its last id and name and
    joins its argument deltas. Only the first choice is assembled, the only one the node reads.
    """
    content: list[str] = []
    has_content = False
    refusal: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage = None
    for chunk in chunks:
        usage = chunk.usage or usage
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            delta = choice.delta
            if delta.content is not None:
                has_content = True
                content.append(delta.content)
            if delta.refusal:
                refusal.append(delta.refusal)
            for tool_call in delta.tool_calls or ():
                call = tool_calls.setdefault(
                    tool_call.index, {"id": None, "type": "function", "function": {"name": None, "arguments": ""}}
                )
                call["id"] = tool_call.id or call["id"]
                if tool_call.function:
                    call["function"]["name"] = tool_call.function.name or call["function"]["name"]
                    call["function"]["arguments"] += tool_call.function.arguments or ""
            finish_reason = choice.finish_reason or finish_reason

    message: dict[str, Any] = {"role": "assistant", "content": "".join(content) if has_content else None}
    if refusal:
        message["refusal"] = "".join(refusal)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

    if usage is not None:
        usage_data = usage.model_dump()
    else:
        completion_text = (message["content"] or "") + "".join(
            call["function"]["arguments"] for call in tool_calls.values()
        )
        usage_data = _estimate_usage(model, messages, completion_text)

    first = chunks[0] if chunks else None
    return ChatCompletion.construct(
        id=first.id if first else "",
        created=first.created if first else 0,
        model=first.model if first else model,
        object="chat.completion",
        system_fingerprint=first.system_fingerprint if first else None,
        choices=[{"index": 0, "message": message, "finish_reason": finish_reason or "stop", "logprobs": None}],
        usage=usage_data,
    )
