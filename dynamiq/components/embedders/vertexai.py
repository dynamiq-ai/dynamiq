from typing import Any

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import VertexAI as VertexAIConnection


class VertexAIEmbedder(BaseEmbedder):
    """
    An embedder for generating embeddings using Vertex AI.

    This component manages a connection to Vertex AI and provides
    methods to generate embeddings for text or documents.

    Attributes:
        connection (VertexAIConnection): The Vertex AI connection instance.
            If not provided, a new connection is created.
        model (str): The Vertex AI embedding model name.
            Defaults to "vertex_ai/text-embedding-005".
        input_type (str): Specifies the embedding task type.
            Can be "search_query", "search_document", "classification", or "clustering".
    """

    connection: VertexAIConnection
    model: str = "vertex_ai/text-embedding-005"
    input_type: str = "search_query"

    def __init__(
        self,
        *,
        connection: VertexAIConnection | None = None,
        model: str | None = None,
        input_type: str | None = None,
        **kwargs: Any
    ):
        """
        Initialize the VertexAIEmbedder.

        Args:
            connection: An existing Vertex AI connection.
            model: Override the default embedding model.
            input_type: Override the default embedding task type.
            **kwargs: Additional BaseEmbedder keyword arguments.
        """
        if connection is None:
            connection = VertexAIConnection()
        super().__init__(connection=connection, **kwargs)

        if model:
            self.model = model
        if input_type:
            self.input_type = input_type

    @property
    def embed_params(self) -> dict:
        """
        Build the parameters for the embedding request.

        Returns:
            A dictionary of parameters including task type,
            truncate length, and embedding dimensions.
        """
        # Copy base parameters to avoid side effects
        params = super().embed_params.copy()

        # Map our input types to Vertex AI task enums
        task_mapping = {
            "search_query": "RETRIEVAL_QUERY",
            "search_document": "RETRIEVAL_DOCUMENT",
            "classification": "CLASSIFICATION",
            "clustering": "CLUSTERING",
        }
        params["task_type"] = task_mapping.get(self.input_type, self.input_type)

        if self.truncate is not None:
            params["truncate"] = self.truncate
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        return params
