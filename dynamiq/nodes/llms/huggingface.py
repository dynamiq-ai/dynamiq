from dynamiq.connections import HuggingFace as HuggingFaceConnection
from dynamiq.nodes.llms.base import BaseLLM


class HuggingFace(BaseLLM):
    """HuggingFace LLM node.

    This class provides an implementation for the HuggingFace Language Model node.

    Attributes:
        connection (HuggingFaceConnection | None): The connection to use for the HuggingFace LLM.
        MODEL_PREFIX (str): The prefix for the HuggingFace model name.
    """
    connection: HuggingFaceConnection | None = None
    MODEL_PREFIX = "huggingface/"

    def __init__(self, **kwargs):
        """Initialize the HuggingFace LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFaceConnection()
        super().__init__(**kwargs)
