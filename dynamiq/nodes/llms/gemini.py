from typing import Any, Literal

from pydantic import BaseModel, PositiveInt

from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

# Google rejects a cache below a per-model minimum that LiteLLM's own 1024-token gate does
# not know about, and the whole request fails: 2048 on gemini-2.5-pro (Gemini API), 4096 on
# gemini-3.8-flash (Vertex). Vertex `us-central1` also miscounts caches as "1 tokens".
_CACHE_TOO_SMALL_INDICATORS = (
    "cached content is too small",  # Gemini API
    "minimum token count to start explicit caching",  # Vertex AI
)


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


def apply_gemini_cache_control(model: str, params: dict[str, Any], cache_control: Any) -> dict[str, Any]:
    """Put ``cache_control`` on the text block of each leading system message.

    The marker goes on a content block, not the message: LiteLLM reads the TTL only from
    there, and would silently fall back to the default one. Messages are copied, never
    mutated -- they may belong to the caller's prompt.
    """
    # `VertexAI` also serves Claude and self-deployed models, `Gemini` also Gemma.
    is_gemini = model.rsplit("/", 1)[-1].startswith("gemini")
    if not cache_control or not is_gemini or not params.get("messages"):
        return params

    control = {"type": "ephemeral"}
    if cache_control.ttl_seconds is not None:
        control["ttl"] = f"{cache_control.ttl_seconds}s"

    messages = list(params["messages"])
    for idx, message in enumerate(messages):
        if message.get("role") != "system":
            break
        messages[idx] = _with_cache_control(message, control)

    return params | {"messages": messages}


def uncached_retry_params(llm: BaseLLM, exc: BaseException, params: dict) -> dict | None:
    """Params to retry with when Google rejects the cache as below the model's minimum, else ``None``."""
    messages = params.get("messages") or []
    msg = str(exc).lower()
    if not any(ind in msg for ind in _CACHE_TOO_SMALL_INDICATORS) or not has_gemini_cache_control(messages):
        return None

    logger.warning(
        "LLM '%s': model '%s' rejected the context cache as too small; retrying uncached.",
        llm.name,
        llm.model,
    )
    return params | {"messages": _strip_cache_control(messages)}


def has_gemini_cache_control(messages: list[dict]) -> bool:
    return any(
        isinstance(block, dict) and "cache_control" in block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
    )


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


class Gemini(BaseLLM):
    """Gemini LLM node.

    This class provides an implementation for the Gemini Language Model node.

    Attributes:
        connection (GeminiConnection): The connection to use for the Gemini LLM.
        cache_control (GeminiCacheControl | Literal[False] | None): Context caching config.
            ``None`` (the default) and ``False`` send the request uncached. Opt-in only: an
            :class:`Agent` never enables it, because the cache bills storage per hour.
    """

    connection: GeminiConnection
    MODEL_PREFIX = "gemini/"
    cache_control: GeminiCacheControl | Literal[False] | None = None

    def __init__(self, **kwargs):
        """Initialize the Gemini LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)

    def supports_prompt_caching(self) -> bool:
        """Keeps an :class:`Agent` from enabling caching by default: Gemini caches bill storage per hour."""
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
