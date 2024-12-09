from typing import TYPE_CHECKING, ClassVar

from dynamiq.connections import Ollama as OllamaConnection
from dynamiq.nodes.llms.base import BaseLLM, BaseLLMUsageData

if TYPE_CHECKING:
    from litellm import ModelResponse


class Ollama(BaseLLM):
    """Ollama LLM node.

    This class provides an implementation for the Ollama Language Model node.
    It supports both chat and completion endpoints through model prefixes.

    Attributes:
        MODEL_PREFIX (ClassVar[str | None]): Optional model prefix, None to allow both ollama/ and ollama_chat/.
        connection (OllamaConnection | None): The connection to use for the Ollama LLM.
    """

    MODEL_PREFIX: ClassVar[str | None] = "ollama/"
    connection: OllamaConnection | None = None

    def __init__(self, **kwargs):
        """Initialize the Ollama LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OllamaConnection()

        super().__init__(**kwargs)

    @classmethod
    def get_usage_data(cls, model: str, completion: "ModelResponse") -> "BaseLLMUsageData":
        """Get usage data for the Ollama LLM.

        Args:
            model (str): The model used for generation.
            completion (ModelResponse): The completion response from the LLM.

        Returns:
            BaseLLMUsageData: A model containing the usage data for the LLM.
        """
        usage = completion.model_extra.get("usage", {})
        prompt_tokens = usage.get("prompt_eval_count", 0)
        completion_tokens = usage.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        return BaseLLMUsageData(
            prompt_tokens=prompt_tokens,
            prompt_tokens_cost_usd=None,
            completion_tokens=completion_tokens,
            completion_tokens_cost_usd=None,
            total_tokens=total_tokens,
            total_tokens_cost_usd=None,
        )
