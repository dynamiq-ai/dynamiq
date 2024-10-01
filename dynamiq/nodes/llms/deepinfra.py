from dynamiq.connections import DeepInfra as DeepInfraConnection
from dynamiq.nodes.llms.base import BaseLLM


class DeepInfra(BaseLLM):
    """DeepInfra LLM node.

    This class provides an implementation for the DeepInfra Language Model node.

    Attributes:
        connection (DeepInfraConnection): The connection to use for the DeepInfra LLM.
        MODEL_PREFIX (str): The prefix for the DeepInfra model name.
    """
    connection: DeepInfraConnection
    MODEL_PREFIX = "deepinfra/"

    def __init__(self, **kwargs):
        """Initialize the DeepInfra LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = DeepInfraConnection()
        super().__init__(**kwargs)
