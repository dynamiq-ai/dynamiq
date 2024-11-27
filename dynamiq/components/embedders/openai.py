from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import OpenAI as OpenAIConnection


class OpenAIEmbedder(BaseEmbedder):
    """
    Provides functionality to compute embeddings for documents using OpenAI's models.

    This class leverages the OpenAI API to generate embeddings for given text documents. It's designed to work
    with instances of the Document class from the dynamiq package. The embeddings generated can be used for tasks
    such as similarity search, clustering, and more.

    Attributes:
        connection (OpenAIConnection): The connection to the  OpenAI API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to "text-embedding-3-small"


    Example:
        >>> from dynamiq.types import Document
        >>> from dynamiq.components.embedders.openai import OpenAIEmbedder
        >>>
        >>> doc = Document(content="I love pizza!")
        >>>
        >>> document_embedder = OpenAIEmbedder()
        >>>
        >>> result = document_embedder.run([doc])
        >>> print(result["documents"][0].embedding)
        [0.017020374536514282, -0.023255806416273117, ...]

    Note:
        An OpenAI API key must be provided either via environment variables or when creating an instance of
        OpenAIDocumentEmbedder through the OpenAIConnection.
    """

    connection: OpenAIConnection | None = None
    model: str = "text-embedding-3-small"

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @property
    def embed_params(self) -> dict:
        params = super().embed_params
        if self.dimensions:
            params["dimensions"] = self.dimensions

        return params
