from dynamiq.connections import Novita as NovitaConnection
from dynamiq.nodes.llms.base import BaseLLM


class Novita(BaseLLM):
    """Novita LLM node.

    This class provides an implementation for the Novita Language Model node.
    Novita provides an OpenAI-compatible API endpoint.

    Attributes:
        connection (NovitaConnection | None): The connection to use for the Novita LLM.
        MODEL_PREFIX (str): The prefix for the Novita model name.
    """

    connection: NovitaConnection | None = None
    MODEL_PREFIX = "novita/"

    def __init__(self, **kwargs):
        """Initialize the Novita LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = NovitaConnection()
        super().__init__(**kwargs)
