from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Cohere as CohereConnection


class CohereEmbedder(BaseEmbedder):
    """
    Initializes the CohereEmbedder component with given configuration.

    Attributes:
        connection (CohereConnection): The connection to the  Cohere API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "cohere/embed-english-v2.0"
        input_type (str): Specifies the type of input you're giving to the model. Defaults to "search_query"
    """
    connection: CohereConnection
    model: str = "cohere/embed-english-v2.0"
    input_type: str = "search_query"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    @property
    def embed_params(self) -> dict:
        params = super().embed_params
        params["input_type"] = self.input_type
        if self.truncate:
            params["truncate"] = self.truncate

        return params
