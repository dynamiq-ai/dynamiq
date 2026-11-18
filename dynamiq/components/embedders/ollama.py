from typing import ClassVar

from pydantic import field_validator

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Ollama as OllamaConnection


class OllamaEmbedder(BaseEmbedder):
    """
    Initializes the OllamaEmbedder component with given configuration.

    Attributes:
        connection (OllamaConnection): The connection to the Ollama server. A new connection to
            http://localhost:11434 is created if none is provided.
        model (str): The model name to use for embedding. The "ollama/" prefix is added if it is missing.
            Defaults to "ollama/nomic-embed-text".
    """

    MODEL_PREFIX: ClassVar[str] = "ollama/"
    connection: OllamaConnection
    model: str = "ollama/nomic-embed-text"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OllamaConnection()
        super().__init__(**kwargs)

    @field_validator("model")
    @classmethod
    def set_model(cls, value: str) -> str:
        """Add the "ollama/" prefix that litellm needs to route the call to Ollama."""
        if not value.startswith(cls.MODEL_PREFIX):
            value = f"{cls.MODEL_PREFIX}{value}"
        return value

    @property
    def embed_params(self) -> dict:
        """
        Returns the embedding parameters for the Ollama server.

        The client of an Ollama connection is the `requests` module, which litellm does not use for Ollama
        embeddings, so the server URL is always sent as `api_base` instead of the client.

        Returns:
            dict: A dictionary containing the parameters for the embedding call.
        """
        return self.connection.conn_params
