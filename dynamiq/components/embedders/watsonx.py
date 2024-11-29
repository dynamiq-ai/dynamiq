from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import WatsonX as WatsonXConnection


class WatsonXEmbedder(BaseEmbedder):
    """
    Initializes the WatsonXEmbedder component with given configuration.

    Attributes:
        connection (WatsonXConnection): The connection to the  WatsonX API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "watsonx/ibm/slate-30m-english-rtrvr"
    """
    connection: WatsonXConnection
    model: str = "watsonx/ibm/slate-30m-english-rtrvr"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = WatsonXConnection()
        super().__init__(**kwargs)
