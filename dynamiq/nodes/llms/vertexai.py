from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class VertexAI(BaseLLM):
    """VertexAI LLM node.

    This class provides an implementation for the VertexAI Language Model node.

    Attributes:
        connection (VertexAIConnection | None): The connection to use for the VertexAI LLM.
        MODEL_PREFIX (str): The prefix for the VertexAI model name.
    """

    connection: VertexAIConnection | None = None
    MODEL_PREFIX = "vertex_ai/"

    def __init__(self, **kwargs):
        """Initialize the VertexAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = VertexAIConnection()
        super().__init__(**kwargs)
