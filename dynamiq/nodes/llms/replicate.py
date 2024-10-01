from dynamiq.connections import Replicate as ReplicateConnection
from dynamiq.nodes.llms.base import BaseLLM


class Replicate(BaseLLM):
    """Replicate LLM node.

    This class provides an implementation for the Replicate Language Model node.

    Attributes:
        connection (ReplicateConnection): The connection to use for the Replicate LLM.
        MODEL_PREFIX (str): The prefix for the Replicate model name.
    """
    connection: ReplicateConnection
    MODEL_PREFIX = "replicate/"

    def __init__(self, **kwargs):
        """Initialize the Replicate LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = ReplicateConnection()
        super().__init__(**kwargs)
