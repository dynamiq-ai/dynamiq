from typing import Any, Literal

from litellm.llms.bedrock.common_utils import BedrockModelInfo
from pydantic import BaseModel

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

_BEDROCK_STOP_UNSUPPORTED_INDICATORS = (
    "doesn't support the stopSequences field",
    "does not support the stopSequences field",
)


def _routes_to_converse(model: str) -> bool:
    """Whether LiteLLM sends this model through the Converse API.

    Only the Converse transform pops ``cache_control_injection_points``. On the
    Invoke route the leftover ``tool_config`` point is spread into the request
    body and Bedrock 400s, so the tool breakpoint is Converse-only.
    """
    return BedrockModelInfo.get_bedrock_route(model) == "converse"


class BedrockCacheControl(BaseModel):
    """Bedrock (Converse API) prompt caching configuration.

    A breakpoint caches everything before it. Converse renders the request as
    ``tools -> system -> messages``, so one point in the message list also
    covers the tool schemas and system prompt.

    Attributes:
        ttl: Cache lifetime. Applies to the tool breakpoint only; LiteLLM drops
            it on the message path, which always uses Bedrock's default.
        cache_injection_point_index: Message index for the rolling breakpoint.
            ``-1`` marks the last message, which is what the next agent loop
            reads back, so each call writes only the delta.
        cache_tools: Also pin a breakpoint after the tool schemas, so they stay
            cached when the message tail is rewritten. Skipped without tools.
    """

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = "5m"
    cache_injection_point_index: int = -1
    cache_tools: bool = True


class Bedrock(BaseLLM):
    """Bedrock LLM node.

    This class provides an implementation for the Bedrock Language Model node.

    Attributes:
        connection (AWSConnection | None): The connection to use for the Bedrock LLM.
        MODEL_PREFIX (str): The prefix for the Bedrock model name.
        cache_control (BedrockCacheControl | None): Prompt caching config.
            ``None`` (the default) requests no caching.
    """
    connection: AWSConnection | None = None
    MODEL_PREFIX = "bedrock/"
    cache_control: BedrockCacheControl | None = None

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach Bedrock prompt caching configuration to completion params."""
        params = super().update_completion_params(params)
        if not self.cache_control:
            return params

        control = self.cache_control.model_dump(
            exclude_none=True,
            exclude={"cache_injection_point_index", "cache_tools"},
        )
        points = params.setdefault("cache_control_injection_points", [])

        # Bedrock allows 4 breakpoints, so don't spend one when there are no
        # tools to cache.
        if self.cache_control.cache_tools and params.get("tools"):
            if _routes_to_converse(self.model):
                points.append({"location": "tool_config", "control": control})
            else:
                logger.debug(
                    "LLM '%s': model '%s' routes to Bedrock Invoke, which has no toolConfig section; "
                    "caching the conversation prefix only.",
                    self.name,
                    self.model,
                )

        points.append(
            {
                "location": "message",
                "index": self.cache_control.cache_injection_point_index,
                "control": control,
            }
        )
        return params

    def __init__(self, **kwargs):
        """Initialize the Bedrock LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        """Bedrock-specific recovery for known-bad completion params.

        Currently handles: hosted models (e.g. openai.gpt-oss-*, moonshotai.kimi-k2.5)
        that reject the `stopSequences` field even though LiteLLM's per-model registry
        claims they support it. We strip `stop`; the removal is persisted on the instance
        only after a successful retry (see :meth:`_persist_completion_recovery`) so later
        calls do not repeat the same failing first attempt. Anything else falls through to
        the base sampling-param backstop.
        """
        msg = str(exc)
        if any(ind in msg for ind in _BEDROCK_STOP_UNSUPPORTED_INDICATORS) and common_params.get("stop"):
            logger.warning(
                "LLM '%s': Bedrock rejected stopSequences for model '%s'; "
                "retrying without `stop` (LiteLLM registry may be stale).",
                self.name,
                self.model,
            )
            recovered = dict(common_params)
            recovered.pop("stop", None)
            return recovered
        return super()._recover_completion_params(exc, common_params)

    def _persist_completion_recovery(self, common_params: dict, recovered: dict) -> None:
        """Persist Bedrock's `stop` drop, then the base sampling-param drops."""
        if common_params.get("stop") and "stop" not in recovered:
            self.stop = None
        super()._persist_completion_recovery(common_params, recovered)
