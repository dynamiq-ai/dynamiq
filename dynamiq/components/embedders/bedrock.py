from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import AWS as BedrockConnection


class BedrockEmbedder(BaseEmbedder):
    """
    Initializes the BedrockEmbedder component with given configuration.

    Attributes:
        connection (BedrockConnection): The connection to the  Bedrock API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "amazon.titan-embed-text-v1".
    """
    connection: BedrockConnection
    model: str = "amazon.titan-embed-text-v1"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = BedrockConnection()
        super().__init__(**kwargs)

    @property
    def embed_params(self) -> dict:
        params = super().embed_params
        if "cohere" in self.model:
            params["input_type"] = self.input_type
            if self.truncate:
                params["truncate"] = self.truncate

        return params
