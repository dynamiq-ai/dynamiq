from typing import Any, Literal

from pydantic import BaseModel, model_validator

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

_BEDROCK_STOP_UNSUPPORTED_INDICATORS = (
    "doesn't support the stopSequences field",
    "does not support the stopSequences field",
)


class BedrockCacheControl(BaseModel):
    """Bedrock (Converse API) prompt caching configuration.

    A breakpoint caches everything before it. Converse renders the request as
    ``tools -> system -> messages``, so the system point also covers the tool schemas
    (measured: adding a ``tool_config`` point on top of it moves neither tokens nor cost).

    Attributes:
        ttl: Cache lifetime for both breakpoints.
        cache_injection_point_index: Message index for the rolling breakpoint.
            ``-1`` marks the last message, which is what the next agent loop
            reads back, so each call writes only the delta.
        cache_system: Also pin a breakpoint on the system message, so the system
            prompt and tool schemas stay cached when the message tail is rewritten
            (history compaction). Resolves to nothing without a system message.
    """

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = "5m"
    cache_injection_point_index: int = -1
    cache_system: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_cache_tools(cls, data: Any) -> Any:
        """``cache_tools`` (v0.64.0) named the same head pin, back when it targeted toolConfig."""
        if isinstance(data, dict) and "cache_tools" in data:
            data = dict(data)
            data.setdefault("cache_system", data.pop("cache_tools"))
            logger.warning("BedrockCacheControl: `cache_tools` is deprecated, use `cache_system`.")
        return data


class Bedrock(BaseLLM):
    """Bedrock LLM node.

    This class provides an implementation for the Bedrock Language Model node.

    Attributes:
        connection (AWSConnection | None): The connection to use for the Bedrock LLM.
        MODEL_PREFIX (str): The prefix for the Bedrock model name.
        cache_control (BedrockCacheControl | Literal[False] | None): Prompt caching config.
            ``None`` (the default) chooses nothing, leaving an :class:`Agent` free to
            enable caching for its own calls; ``False`` opts out. A bare node caches
            only when given a config -- ``None`` and ``False`` both send no breakpoints.
    """
    connection: AWSConnection | None = None
    MODEL_PREFIX = "bedrock/"
    # ``None`` = nothing chosen (survives a YAML round trip, lets an Agent inject the
    # default); ``False`` = opt out.
    cache_control: BedrockCacheControl | Literal[False] | None = None

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach the node's own prompt caching configuration to completion params."""
        params = super().update_completion_params(params)
        return self._apply_cache_control(params, self.cache_control)

    def _apply_cache_control(self, params: dict[str, Any], cache_control: Any) -> dict[str, Any]:
        """Attach Bedrock cache points for the given configuration."""
        if not cache_control:
            return params

        control = cache_control.model_dump(
            exclude_none=True,
            exclude={"cache_injection_point_index", "cache_system"},
        )
        points = params.setdefault("cache_control_injection_points", [])

        # Head first: points are honored in order, so the durable one wins if the
        # 4-block budget runs short. Separate copies -- LiteLLM assigns by reference.
        if cache_control.cache_system:
            points.append({"location": "message", "role": "system", "control": dict(control)})

        points.append(
            {
                "location": "message",
                "index": cache_control.cache_injection_point_index,
                "control": dict(control),
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
