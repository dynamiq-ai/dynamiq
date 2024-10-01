from dynamiq.connections import Mistral as MistralConnection
from dynamiq.nodes.llms.base import BaseLLM


class Mistral(BaseLLM):
    """Mistral LLM node.

    This class provides an implementation for the Mistral Language Model node.

    Attributes:
        connection (MistralConnection | None): The connection to use for the Mistral LLM.
        MODEL_PREFIX (str): The prefix for the Mistral model name.
    """
    connection: MistralConnection | None = None
    MODEL_PREFIX = "mistral/"

    def __init__(self, **kwargs):
        """Initialize the Mistral LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralConnection()
        super().__init__(**kwargs)
