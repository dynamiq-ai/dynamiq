import json
from typing import Any

from dynamiq.connections import Groq as GroqConnection
from dynamiq.nodes.llms.base import BaseLLM, provider_error_text
from dynamiq.utils.logger import logger

# Groq checks a reply against the request's tools or JSON schema itself, and answers one that fails with these error
# codes and the reply in "failed_generation", instead of returning the reply.
_REJECTED_GENERATION_CODES = ("tool_use_failed", "json_validate_failed")


def _rejected_generation(exc: BaseException) -> str | None:
    """The reply Groq generated but refused to return because it failed Groq's own validation, if ``exc`` is that."""
    text = provider_error_text(exc)
    start = text.find("{")
    if start < 0:
        return None
    try:
        error = json.loads(text[start:]).get("error") or {}
    except (json.JSONDecodeError, AttributeError):
        return None
    if error.get("code") not in _REJECTED_GENERATION_CODES or not isinstance(error.get("failed_generation"), str):
        return None
    return error["failed_generation"]


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

    def _call_completion(self, params: dict[str, Any]):
        try:
            return super()._call_completion(params)
        except Exception as exc:
            return self._rejected_reply(exc, params)

    async def _acall_completion(self, params: dict[str, Any]):
        try:
            return await super()._acall_completion(params)
        except Exception as exc:
            return self._rejected_reply(exc, params)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        # A rejected reply can only be handed back from a request that was not streamed.
        if common_params.get("stream") and _rejected_generation(exc) is not None:
            recovered = {key: value for key, value in common_params.items() if key != "stream_options"}
            recovered["stream"] = False
            return recovered
        return super()._recover_completion_params(exc, common_params)
