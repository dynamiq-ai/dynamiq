from dynamiq.connections import xAI as xAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class xAI(BaseLLM):
    """xAI LLM node.

    This class provides an implementation for the xAI Language Model node.

    Attributes:
        connection (xAIConnection | None): The connection to use for the xAI LLM.
        MODEL_PREFIX (str): The prefix for the xAI model name.
    """

    connection: xAIConnection | None = None
    MODEL_PREFIX = "xai/"

    def __init__(self, **kwargs):
        """Initialize the xAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = xAIConnection()
        super().__init__(**kwargs)
