from dynamiq.connections import Cohere as CohereConnection
from dynamiq.nodes.llms.base import BaseLLM


class Cohere(BaseLLM):
    """Cohere LLM node.

    This class provides an implementation for the Cohere Language Model node.

    Attributes:
        connection (CohereConnection): The connection to use for the Cohere LLM.
    """
    connection: CohereConnection

    def __init__(self, **kwargs):
        """Initialize the Cohere LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)
