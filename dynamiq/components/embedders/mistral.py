from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Mistral as MistralConnection


class MistralEmbedder(BaseEmbedder):
    """
    Initializes the MistralEmbedder component with given configuration.

    Attributes:
        connection (MistralConnection): The connection to the  Mistral API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "mistral/mistral-embed"
    """
    connection: MistralConnection
    model: str = "mistral/mistral-embed"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralConnection()
        super().__init__(**kwargs)
