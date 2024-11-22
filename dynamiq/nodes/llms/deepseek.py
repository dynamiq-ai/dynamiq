from dynamiq.connections import DeepSeek as DeepSeekConnection
from dynamiq.nodes.llms.base import BaseLLM


class DeepSeek(BaseLLM):
    """DeepSeek LLM node.

    This class provides an implementation for the DeepSeek Language Model node.

    Attributes:
        connection (DeepSeekConnection): The connection to use for the DeepSeek LLM.
        MODEL_PREFIX (str): The prefix for the DeepSeek model name.
    """

    connection: DeepSeekConnection
    MODEL_PREFIX = "deepseek/"

    def __init__(self, **kwargs):
        """Initialize the Replicate LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = DeepSeekConnection()
        super().__init__(**kwargs)
