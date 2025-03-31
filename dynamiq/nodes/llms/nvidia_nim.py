from dynamiq.connections import NvidiaNIM as NvidiaNIMConnection
from dynamiq.nodes.llms.base import BaseLLM


class NvidiaNIM(BaseLLM):
    """Nvidia NIM LLM node.

    This class provides an implementation for the Nvidia NIM Language Model node.

    Attributes:
        connection (Nvidia_NIM_Connection | None): The connection to use for the Nvidia NIM LLM.
        MODEL_PREFIX (str): The prefix for the Nvidia NIM model name.
    """

    connection: NvidiaNIMConnection | None = None
    MODEL_PREFIX = "nvidia_nim/"

    def __init__(self, **kwargs):
        """Initialize the Nvidia NIM LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = NvidiaNIMConnection()
        super().__init__(**kwargs)
