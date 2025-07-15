from typing import Any

from dynamiq.components.embedders.vertexai import VertexAIEmbedder as VertexAIEmbedderComponent
from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class VertexAIDocumentEmbedder(DocumentEmbedder):
    """
    Computes embeddings for documents using Vertex AI.

    This class extends DocumentEmbedder to generate document embeddings
    via the Vertex AI embedding API.

    Attributes:
        name (str): The name of this node.
        connection (VertexAIConnection): The Vertex AI connection instance.
        model (str): The Vertex AI embedding model name.
        input_type (str): The embedding task type (defaults to "search_document").
        document_embedder (VertexAIEmbedderComponent): The embedder component.
    """

    name: str = "VertexAIDocumentEmbedder"
    connection: VertexAIConnection
    model: str = "vertex_ai/text-embedding-005"
    input_type: str = "search_document"
    document_embedder: VertexAIEmbedderComponent | None = None

    def __init__(
        self,
        *,
        connection: VertexAIConnection | None = None,
        model: str | None = None,
        input_type: str | None = None,
        **kwargs: Any
    ):
        """
        Initialize the document embedder.

        Args:
            connection: Optional existing Vertex AI connection.
            model: Optional override for the embedding model.
            input_type: Optional override for the embedding task type.
            **kwargs: Additional keyword args for the base node.
        """
        if connection is None:
            connection = VertexAIConnection()
        super().__init__(connection=connection, **kwargs)

        if model:
            self.model = model
        if input_type:
            self.input_type = input_type

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize or reuse the Vertex AI embedder component.

        Args:
            connection_manager: Optional connection manager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

        if self.document_embedder is None:
            self.document_embedder = VertexAIEmbedderComponent(
                connection=self.connection, model=self.model, input_type=self.input_type
            )


class VertexAITextEmbedder(TextEmbedder):
    """
    Computes embeddings for text strings using Vertex AI.

    This class extends TextEmbedder to generate text embeddings
    via the Vertex AI embedding API.

    Attributes:
        name (str): The name of this node.
        connection (VertexAIConnection): The Vertex AI connection instance.
        model (str): The Vertex AI embedding model name.
        input_type (str): The embedding task type (defaults to "search_query").
        text_embedder (VertexAIEmbedderComponent): The embedder component.
    """

    name: str = "VertexAITextEmbedder"
    connection: VertexAIConnection
    model: str = "vertex_ai/text-embedding-005"
    input_type: str = "search_query"
    text_embedder: VertexAIEmbedderComponent | None = None

    def __init__(
        self,
        *,
        connection: VertexAIConnection | None = None,
        model: str | None = None,
        input_type: str | None = None,
        **kwargs: Any
    ):
        """
        Initialize the text embedder.

        Args:
            connection: Optional existing Vertex AI connection.
            model: Optional override for the embedding model.
            input_type: Optional override for the embedding task type.
            **kwargs: Additional keyword args for the base node.
        """
        if connection is None:
            connection = VertexAIConnection()
        super().__init__(connection=connection, **kwargs)

        if model:
            self.model = model
        if input_type:
            self.input_type = input_type

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize or reuse the Vertex AI embedder component.

        Args:
            connection_manager: Optional connection manager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

        if self.text_embedder is None:
            self.text_embedder = VertexAIEmbedderComponent(
                connection=self.connection, model=self.model, input_type=self.input_type
            )
