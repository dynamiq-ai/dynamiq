from dynamiq.connections import OpenRouter as OpenRouterConnection
from dynamiq.nodes.llms.base import BaseLLM


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
