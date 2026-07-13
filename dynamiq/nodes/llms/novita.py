from dynamiq.connections import Novita as NovitaConnection
from dynamiq.nodes.llms.base import BaseLLM


class Novita(BaseLLM):
    """Novita AI LLM node.

    This class provides an implementation for the Novita AI Language Model node.

    Attributes:
        connection (NovitaConnection | None): The connection to use for the Novita AI LLM.
        MODEL_PREFIX (str): The prefix for the Novita AI model name.
    """

    connection: NovitaConnection | None = None
    MODEL_PREFIX = "novita/"

    def __init__(self, **kwargs):
        """Initialize the Novita AI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = NovitaConnection()
        super().__init__(**kwargs)
