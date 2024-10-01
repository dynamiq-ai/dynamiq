from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class TogetherAI(BaseLLM):
    """TogetherAI LLM node.

    This class provides an implementation for the TogetherAI Language Model node.

    Attributes:
        connection (TogetherAIConnection | None): The connection to use for the TogetherAI LLM.
        MODEL_PREFIX (str): The prefix for the TogetherAI model name.
    """
    connection: TogetherAIConnection | None = None
    MODEL_PREFIX = "together_ai/"

    def __init__(self, **kwargs):
        """Initialize the TogetherAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = TogetherAIConnection()
        super().__init__(**kwargs)
