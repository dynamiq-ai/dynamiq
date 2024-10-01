from dynamiq.connections import Cerebras as CerebrasConnection
from dynamiq.nodes.llms.base import BaseLLM


class Cerebras(BaseLLM):
    """Cerebras LLM node.

    This class provides an implementation for the Cerebras Language Model node.

    Attributes:
        connection (CerebrasConnection): The connection to use for the Cerebras LLM.
        MODEL_PREFIX (str): The prefix for the Cerebras model name.
    """
    connection: CerebrasConnection
    MODEL_PREFIX = "cerebras/"

    def __init__(self, **kwargs):
        """Initialize the Cerebras LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CerebrasConnection()
        super().__init__(**kwargs)
