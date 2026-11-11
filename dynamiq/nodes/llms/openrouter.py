from typing import TYPE_CHECKING, Any

from dynamiq.connections import OpenRouter as OpenRouterConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.litellm import on_litellm_import
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from litellm import ModelResponse

# OpenRouter ends a reply that the provider cut off inside a tool call with finish_reason "tool_calls", for OpenAI
# models among others. What the provider itself said, one of these, is passed on as "native_finish_reason".
_CUT_OFF_NATIVE_FINISH_REASONS = ("length", "max_tokens", "max_output_tokens")


def _is_cut_off(native_finish_reason: Any) -> bool:
    return isinstance(native_finish_reason, str) and native_finish_reason.lower() in _CUT_OFF_NATIVE_FINISH_REASONS


def _patch_litellm_openrouter_cut_off_stream() -> bool:
    """Make LiteLLM report a streamed OpenRouter reply that was cut off inside a tool call as cut off.

    LiteLLM drops "native_finish_reason" from a stream's chunks, so the finish reason is corrected before LiteLLM
    reads the chunk. Returns False, leaving LiteLLM untouched, if its internals have changed.
    """
    try:
        from litellm.llms.openrouter.chat import transformation
    except Exception as exc:
        logger.debug("LiteLLM OpenRouter stream patch: import failed: %s", exc)
        return False

    handler = getattr(transformation, "OpenRouterChatCompletionStreamingHandler", None)
    original = getattr(handler, "chunk_parser", None)
    if original is None:
        logger.warning("LiteLLM OpenRouter stream patch: the stream handler's chunk_parser was not found; skipping.")
        return False
    if getattr(original, "__dynamiq_patch__", False):
        return True

    def _patched(self, chunk: dict):
        for choice in chunk.get("choices") or []:
            if (
                isinstance(choice, dict)
                and choice.get("finish_reason")
                and _is_cut_off(choice.get("native_finish_reason"))
            ):
                choice["finish_reason"] = "length"
        return original(self, chunk)

    _patched.__dynamiq_patch__ = True
    handler.chunk_parser = _patched
    return True


on_litellm_import(_patch_litellm_openrouter_cut_off_stream)


class OpenRouter(BaseLLM):
    """OpenRouter LLM node.

    This class provides an implementation for Large Language Model node that routes
    requests through OpenRouter to various underlying providers.

    Attributes:
        connection (OpenRouterConnection): The connection to use for the OpenRouter LLM.
        MODEL_PREFIX (str): The prefix for the OpenRouter provider.
    """

    connection: OpenRouterConnection
    MODEL_PREFIX = "openrouter/"

    def __init__(self, **kwargs):
        """Initialize the OpenRouter LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenRouterConnection()
        super().__init__(**kwargs)

    def _handle_completion_response(self, response: "ModelResponse", config: RunnableConfig = None, **kwargs) -> dict:
        # A response that was not streamed still has the provider's own reason; a stream's is corrected as it is read.
        choice = response.choices[0]
        fields = getattr(choice, "provider_specific_fields", None)
        if isinstance(fields, dict) and _is_cut_off(fields.get("native_finish_reason")):
            choice.finish_reason = "length"
        return super()._handle_completion_response(response, config, **kwargs)
