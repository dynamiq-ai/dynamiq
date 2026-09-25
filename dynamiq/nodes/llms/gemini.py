from typing import Any, ClassVar, Literal

from pydantic import BaseModel, PositiveInt

from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

# Google rejects a cache below a per-model minimum (2048 tokens for gemini-2.5-pro) that
# LiteLLM's own 1024-token gate does not know about, and the whole request fails.
_CACHE_TOO_SMALL_INDICATOR = "cached content is too small"


class GeminiCacheControl(BaseModel):
    """Gemini explicit context caching configuration.

    The leading system messages and the tool schemas are stored as a Google
    ``cachedContent`` resource on first use; later requests with the same system prompt
    and tools reuse it and are billed at the cached-token rate.

    Unlike Anthropic, every distinct cached prefix is a separate resource billed for
    storage until it expires, so only the system prompt is cached -- a rolling point on
    the message tail would create a new cache on every agent step. Prompts below the
    model's minimum are sent uncached.

    Attributes:
        ttl_seconds: Cache lifetime, counted from creation. ``None`` keeps Google's
            default of one hour.
    """

    ttl_seconds: PositiveInt | None = None


class GeminiCachingLLM(BaseLLM):
    """Base for nodes serving Gemini models through LiteLLM's context caching.

    Attributes:
        cache_control (GeminiCacheControl | Literal[False] | None): Context caching config.
            ``None`` (the default) and ``False`` send the request uncached. Opt-in only: an
            :class:`Agent` never enables it, because the cache bills storage per hour.
    """

    # Keeps `Agent` from injecting a default config (see `default_cache_control`).
    BREAKPOINT_MODEL_FAMILIES: ClassVar[tuple[str, ...]] = ()
    cache_control: GeminiCacheControl | Literal[False] | None = None

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Mark the system prompt for context caching when the node has a config."""
        params = super().update_completion_params(params)
        return self._apply_cache_control(params, self.cache_control)

    def _apply_cache_control(self, params: dict[str, Any], cache_control: Any) -> dict[str, Any]:
        """Put ``cache_control`` on the text block of each leading system message.

        The marker goes on a content block, not the message: LiteLLM reads the TTL only
        from there, and would silently fall back to the default one. Messages are copied,
        never mutated -- they may belong to the caller's prompt.
        """
        # The `VertexAI` node also serves Claude and self-deployed models, `Gemini` also Gemma.
        is_gemini = self.model.rsplit("/", 1)[-1].startswith("gemini")
        if not cache_control or not is_gemini or not params.get("messages"):
            return params

        control = {"type": "ephemeral"}
        if cache_control.ttl_seconds is not None:
            control["ttl"] = f"{cache_control.ttl_seconds}s"

        messages = list(params["messages"])
        for idx, message in enumerate(messages):
            if message.get("role") != "system":
                break
            messages[idx] = self._with_cache_control(message, control)

        return params | {"messages": messages}

    @staticmethod
    def _with_cache_control(message: dict, control: dict) -> dict:
        content = message.get("content")
        if isinstance(content, str):
            blocks: list[Any] = [{"type": "text", "text": content}]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            blocks = [*content[:-1], dict(content[-1])]
        else:
            return message

        blocks[-1]["cache_control"] = dict(control)
        return {**message, "content": blocks}

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        """Retry uncached when Google finds the system prompt below the model's cache minimum."""
        messages = common_params.get("messages") or []
        if _CACHE_TOO_SMALL_INDICATOR in str(exc).lower() and self._has_cache_control(messages):
            logger.warning(
                "LLM '%s': model '%s' rejected the context cache as too small; retrying uncached.",
                self.name,
                self.model,
            )
            return common_params | {"messages": self._strip_cache_control(messages)}
        return super()._recover_completion_params(exc, common_params)

    @staticmethod
    def _strip_cache_control(messages: list[dict]) -> list[dict]:
        stripped = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                content = [
                    {k: v for k, v in block.items() if k != "cache_control"} if isinstance(block, dict) else block
                    for block in content
                ]
                message = {**message, "content": content}
            stripped.append(message)
        return stripped

    def _persist_completion_recovery(self, common_params: dict, recovered: dict) -> None:
        """Stop caching on this node once the prompt proved too small, so later calls skip the failing attempt."""
        if self.cache_control and not self._has_cache_control(recovered["messages"]):
            self.cache_control = False
        super()._persist_completion_recovery(common_params, recovered)

    @staticmethod
    def _has_cache_control(messages: list[dict]) -> bool:
        return any(
            isinstance(block, dict) and "cache_control" in block
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
        )


class Gemini(GeminiCachingLLM):
    """Gemini LLM node.

    This class provides an implementation for the Gemini Language Model node.

    Attributes:
        connection (GeminiConnection): The connection to use for the Gemini LLM.
    """

    connection: GeminiConnection
    MODEL_PREFIX = "gemini/"

    def __init__(self, **kwargs):
        """Initialize the Gemini LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)
