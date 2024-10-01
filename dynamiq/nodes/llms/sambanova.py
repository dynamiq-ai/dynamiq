from dynamiq.connections import SambaNova as SambaNovaConnection
from dynamiq.nodes.llms.base import BaseLLM


class SambaNova(BaseLLM):
    """SambaNova LLM node.

    This class provides an implementation for the SambaNova Language Model node.

    Attributes:
        connection (SambaNovaConnection): The connection to use for the SambaNova LLM.
        MODEL_PREFIX (str): The prefix for the SambaNova model name.
    """
    connection: SambaNovaConnection
    MODEL_PREFIX = "sambanova/"

    def __init__(self, **kwargs):
        """Initialize the SambaNova LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = SambaNovaConnection()
        super().__init__(**kwargs)
