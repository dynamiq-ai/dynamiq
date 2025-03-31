from dynamiq.connections import FireworksAI as FireworksAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class FireworksAI(BaseLLM):
    """FireworksAI LLM node.

    This class provides an implementation for the Fireworks AI Language Model node.

    Attributes:
        connection (FireworksAIConnection | None): The connection to use for the Fireworks AI LLM.
        MODEL_PREFIX (str): The prefix for the Fireworks AI model name.
    """

    connection: FireworksAIConnection | None = None
    MODEL_PREFIX = "fireworks_ai/"

    def __init__(self, **kwargs):
        """Initialize the FireworksAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = FireworksAIConnection()
        super().__init__(**kwargs)
