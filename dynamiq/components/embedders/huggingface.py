from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import HuggingFace as HuggingFaceConnection


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Initializes the HuggingFaceEmbedder component with given configuration.

    Attributes:
        connection (HuggingFaceConnection): The connection to the  HuggingFace API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "huggingface/BAAI/bge-large-zh"
    """
    connection: HuggingFaceConnection
    model: str = "huggingface/BAAI/bge-large-zh"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFaceConnection()
        super().__init__(**kwargs)
