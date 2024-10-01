from dynamiq.connections import AI21 as AI21Connection
from dynamiq.nodes.llms.base import BaseLLM


class AI21(BaseLLM):
    """AI21 LLM node.

    This class provides an implementation for the AI21 Language Model node.

    Attributes:
        connection (AI21Connection): The connection to use for the AI21 LLM.
        MODEL_PREFIX (str): The prefix for the AI21 model name.
    """
    connection: AI21Connection
    MODEL_PREFIX = "ai21/"

    def __init__(self, **kwargs):
        """Initialize the AI21 LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AI21Connection()
        super().__init__(**kwargs)
