from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

_BEDROCK_STOP_UNSUPPORTED_INDICATORS = (
    "doesn't support the stopSequences field",
    "does not support the stopSequences field",
)


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
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)

    def _recover_completion_params(self, exc: BaseException, common_params: dict) -> dict | None:
        """Bedrock-specific recovery for known-bad completion params.

        Currently handles: hosted models (e.g. openai.gpt-oss-*, moonshotai.kimi-k2.5)
        that reject the `stopSequences` field even though LiteLLM's per-model registry
        claims they support it. We strip `stop` and signal a one-shot retry.
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
            self.stop = None
            return recovered
        return None
