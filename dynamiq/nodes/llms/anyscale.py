from dynamiq.connections import Anyscale as AnyscaleConnection
from dynamiq.nodes.llms.base import BaseLLM


class Anyscale(BaseLLM):
    """Anyscale LLM node.

    This class provides an implementation for the Anyscale Language Model node.

    Attributes:
        connection (AnyscaleConnection | None): The connection to use for the Anyscale LLM.
        MODEL_PREFIX (str): The prefix for the Anyscale model name.
    """
    connection: AnyscaleConnection | None = None
    MODEL_PREFIX = "anyscale/"

    def __init__(self, **kwargs):
        """Initialize the Anyscale LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AnyscaleConnection()
        super().__init__(**kwargs)
