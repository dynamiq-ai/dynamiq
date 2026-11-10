import json
from typing import Any

from dynamiq.connections import Groq as GroqConnection
from dynamiq.nodes.llms.base import BaseLLM, provider_error_text
from dynamiq.utils.logger import logger

# Groq checks a reply against the request's tools or JSON schema itself, and answers one that fails with these error
# codes and the reply in "failed_generation", instead of returning the reply.
_REJECTED_GENERATION_CODES = ("tool_use_failed", "json_validate_failed")


def _is_rejection(error: Any) -> bool:
    return (
        isinstance(error, dict)
        and error.get("code") in _REJECTED_GENERATION_CODES
        and isinstance(error.get("failed_generation"), str)
    )


def _rejected_generation(exc: BaseException) -> str | None:
    """The reply Groq generated but refused to return because it failed Groq's own validation, if ``exc`` is that."""
    text = provider_error_text(exc)
    start = text.find("{")
    if start < 0:
        return None
    try:
        error = json.loads(text[start:]).get("error")
    except (json.JSONDecodeError, AttributeError):
        return None
    return error["failed_generation"] if _is_rejection(error) else None


def _stream_rejection(exc: BaseException | None) -> dict | None:
    """Groq's rejection of a streamed reply, if reading the stream failed with one.

    LiteLLM raises Groq's error event with the error as its body, then fails to wrap it, so the error is the body of
    ``exc`` or of an exception that was being handled when ``exc`` was raised.
    """
    while exc is not None:
        if _is_rejection(body := getattr(exc, "body", None)):
            return body
        exc = exc.__context__
    return None


class _RejectedReplyAsText:
    """A Groq stream that ends with a reply Groq rejected, as text, instead of raising while it is read.

    Groq streams such a reply's text first and the rejection after it, which carries what was not streamed (a tool
    call that failed validation) in "failed_generation".
    """

    def __init__(self, stream: Any, llm_name: str):
        self._stream = stream
        self._llm_name = llm_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self):
        try:
            yield from self._stream
        except Exception as exc:
            yield self._text_chunk(exc)

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                yield chunk
        except Exception as exc:
            yield self._text_chunk(exc)

    def _text_chunk(self, exc: Exception):
        error = _stream_rejection(exc)
        if error is None:
            raise exc
        logger.warning(
            f"LLM '{self._llm_name}': Groq rejected the model's reply ({error.get('message')}); returning it as text."
        )
        from litellm import ModelResponseStream

        delta = {"content": error["failed_generation"]}
        return ModelResponseStream(
            model=self._stream.model, choices=[{"index": 0, "delta": delta, "finish_reason": "stop"}]
        )


class Groq(BaseLLM):
    """Groq LLM node.

    This class provides an implementation for the Groq Language Model node.

    A reply Groq rejects for not matching the request's tools or JSON schema is returned as the reply's text, so a
    caller such as an agent can ask the model to correct it, as it would with any other provider.

    Attributes:
        connection (GroqConnection | None): The connection to use for the Groq LLM.
        MODEL_PREFIX (str): The prefix for the Groq model name.
    """
    connection: GroqConnection | None = None
    MODEL_PREFIX = "groq/"

    def __init__(self, **kwargs):
        """Initialize the Groq LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GroqConnection()
        super().__init__(**kwargs)

    def _rejected_reply(self, exc: BaseException, params: dict[str, Any]):
        generation = _rejected_generation(exc)
        if generation is None or params.get("stream"):
            raise exc
        logger.warning(f"LLM '{self.name}': Groq rejected the model's reply ({exc}); returning it as text.")
        from litellm import ModelResponse

        return ModelResponse(
            model=params.get("model"),
            choices=[{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": generation}}],
        )

    def _reply(self, response: Any, params: dict[str, Any]):
        return _RejectedReplyAsText(response, self.name) if params.get("stream") else response

    def _call_completion(self, params: dict[str, Any]):
        try:
            return self._reply(super()._call_completion(params), params)
        except Exception as exc:
            return self._rejected_reply(exc, params)

    async def _acall_completion(self, params: dict[str, Any]):
        try:
            return self._reply(await super()._acall_completion(params), params)
        except Exception as exc:
            return self._rejected_reply(exc, params)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        # LiteLLM sends a streamed request that has a response format without streaming, so Groq rejects its reply
        # when the request is sent. The text returned for a rejected reply is a whole response, not a stream, so the
        # request is sent again without streaming.
        if common_params.get("stream") and _rejected_generation(exc) is not None:
            recovered = {key: value for key, value in common_params.items() if key != "stream_options"}
            recovered["stream"] = False
            return recovered
        return super()._recover_completion_params(exc, common_params)
