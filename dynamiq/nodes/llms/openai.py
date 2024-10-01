from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class OpenAI(BaseLLM):
    """OpenAI LLM node.

    This class provides an implementation for the OpenAI Language Model node.

    Attributes:
        connection (OpenAIConnection | None): The connection to use for the OpenAI LLM.
    """
    connection: OpenAIConnection | None = None

    def __init__(self, **kwargs):
        """Initialize the OpenAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)
