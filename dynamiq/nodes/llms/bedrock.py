from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

_BEDROCK_STOP_UNSUPPORTED_INDICATORS = (
    "doesn't support the stopSequences field",
    "does not support the stopSequences field",
)


def _ensure_parallel_tool_choice_type(optional_params: dict) -> None:
    """Add Anthropic's required ``type`` to litellm's Bedrock parallel-tool config.

    Workaround for BerriAI/litellm#22637 (fix PR #22638 unmerged): for Claude 4.5+ on
    Bedrock, litellm builds ``_parallel_tool_use_config`` with a ``tool_choice`` that
    omits the required ``type`` discriminator, so Bedrock rejects the request with
    "tool_choice.type: Field required". We add ``type="auto"`` only when it is missing,
    which preserves litellm's ``disable_parallel_tool_use`` (so a user's
    ``parallel_tool_calls`` True/False is honoured) and is a no-op once litellm is fixed.
    """
    cfg = optional_params.get("_parallel_tool_use_config")
    if isinstance(cfg, dict):
        tool_choice = cfg.get("tool_choice")
        if isinstance(tool_choice, dict) and "type" not in tool_choice:
            tool_choice["type"] = "auto"


def _install_litellm_bedrock_parallel_tool_patch() -> None:
    """Idempotently wrap ``AmazonConverseConfig.map_openai_params`` to apply the fix.

    Wrapping (rather than replacing) keeps litellm's behaviour intact and merely repairs
    the malformed parallel-tool config on the way out. Guarded so a litellm refactor can
    never break importing/constructing this node.
    """
    try:
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    except Exception:
        return

    if getattr(AmazonConverseConfig, "_dynamiq_parallel_tc_patched", False):
        return

    original_map_openai_params = AmazonConverseConfig.map_openai_params

    def map_openai_params_with_parallel_tool_fix(self, *args, **kwargs):
        optional_params = original_map_openai_params(self, *args, **kwargs)
        try:
            _ensure_parallel_tool_choice_type(optional_params)
        except Exception as exc:  # never let the workaround break litellm's normal mapping
            logger.warning("Bedrock parallel-tool-choice fix skipped (litellm#22637): %s", exc)
        return optional_params

    AmazonConverseConfig.map_openai_params = map_openai_params_with_parallel_tool_fix
    AmazonConverseConfig._dynamiq_parallel_tc_patched = True


class Bedrock(BaseLLM):
    """Bedrock LLM node.

    This class provides an implementation for the Bedrock Language Model node.

    Attributes:
        connection (AWSConnection | None): The connection to use for the Bedrock LLM.
        MODEL_PREFIX (str): The prefix for the Bedrock model name.
    """
    connection: AWSConnection | None = None
    MODEL_PREFIX = "bedrock/"

    def __init__(self, **kwargs):
        """Initialize the Bedrock LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        _install_litellm_bedrock_parallel_tool_patch()
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        """Bedrock-specific recovery for known-bad completion params.

        Currently handles: hosted models (e.g. openai.gpt-oss-*, moonshotai.kimi-k2.5)
        that reject the `stopSequences` field even though LiteLLM's per-model registry
        claims they support it. We strip `stop` and persist that recovery on the
        instance so later calls do not repeat the same failing first attempt.
        """
        msg = str(exc)
        if any(ind in msg for ind in _BEDROCK_STOP_UNSUPPORTED_INDICATORS) and common_params.get("stop"):
            logger.warning(
                "LLM '%s': Bedrock rejected stopSequences for model '%s'; "
                "retrying without `stop` (LiteLLM registry may be stale).",
                self.name,
                self.model,
            )
            self.stop = None
            recovered = dict(common_params)
            recovered.pop("stop", None)
            return recovered
        return None
