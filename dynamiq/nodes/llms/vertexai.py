from typing import Any, Literal

from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.gemini import (
    GeminiCacheControl,
    apply_gemini_cache_control,
    has_gemini_cache_control,
    uncached_retry_params,
)


class VertexAI(BaseLLM):
    """VertexAI LLM node.

    This class provides an implementation for the VertexAI Language Model node.

    Attributes:
        connection (VertexAIConnection | None): The connection to use for the VertexAI LLM.
        MODEL_PREFIX (str): The prefix for the VertexAI model name.
        cache_control (GeminiCacheControl | Literal[False] | None): Context caching config,
            applied to Gemini models only. ``None`` (the default) and ``False`` send the
            request uncached. Opt-in only: an :class:`Agent` never enables it, because the
            cache bills storage per hour.
    """

    connection: VertexAIConnection | None = None
    MODEL_PREFIX = "vertex_ai/"
    cache_control: GeminiCacheControl | Literal[False] | None = None

    def __init__(self, **kwargs):
        """Initialize the VertexAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = VertexAIConnection()
        super().__init__(**kwargs)

    def supports_prompt_caching(self) -> bool:
        """Keeps an :class:`Agent` from enabling caching by default: Gemini caches bill storage per hour.

        Also covers Claude on Vertex, which would otherwise be handed a Gemini config.
        """
        return False

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach the node's own context caching configuration to completion params."""
        params = super().update_completion_params(params)
        return apply_gemini_cache_control(self.model, params, self.cache_control)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        """Retry uncached when Google finds the system prompt below the model's cache minimum."""
        return uncached_retry_params(self, exc, common_params) or super()._recover_completion_params(exc, common_params)

    def _persist_completion_recovery(self, common_params: dict, recovered: dict) -> None:
        """Stop caching once the prompt proved too small, so later calls skip the failing attempt."""
        if self.cache_control and not has_gemini_cache_control(recovered["messages"]):
            self.cache_control = False
        super()._persist_completion_recovery(common_params, recovered)
