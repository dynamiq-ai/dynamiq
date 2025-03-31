from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.nodes.llms.base import BaseLLM


class Gemini(BaseLLM):
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
