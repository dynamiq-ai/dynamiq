from dynamiq.connections import CheaperInference as CheaperInferenceConnection
from dynamiq.nodes.llms.base import BaseLLM


class CheaperInference(BaseLLM):
    """Cheaper Inference LLM node.

    This class provides an implementation for Large Language Model node that routes
    requests through Cheaper Inference's OpenAI-compatible API to various underlying providers.

    Cheaper Inference model ids are bare names (e.g. `gpt-5.4-mini`, `glm-5.3`,
    `deepseek-v4-flash`). LiteLLM has no Cheaper Inference provider, so requests go
    through its OpenAI-compatible client pointed at the Cheaper Inference endpoint.

    Attributes:
        connection (CheaperInferenceConnection): The connection to use for the Cheaper Inference LLM.
        MODEL_PREFIX (str): The LiteLLM prefix for OpenAI-compatible endpoints.
    """

    connection: CheaperInferenceConnection
    MODEL_PREFIX = "openai/"

    def __init__(self, **kwargs):
        """Initialize the Cheaper Inference LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CheaperInferenceConnection()
        super().__init__(**kwargs)
