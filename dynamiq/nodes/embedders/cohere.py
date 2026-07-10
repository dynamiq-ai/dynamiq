from typing import Literal

from pydantic import Field

from dynamiq.components.embedders.cohere import CohereEmbedder as CohereEmbedderComponent
from dynamiq.connections import Cohere as CohereConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class CohereDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using Cohere models.

    This class extends ConnectionNode to create embeddings for documents using Cohere API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (CohereConnection | None): The connection to the Cohere API.
        model (str): The model name to use for embedding.
        document_embedder (CohereDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[CohereConnection]): The connection to the Cohere API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to `embed-v4.0`.
    """

    name: str = "cohere-document-embedder"
    connection: CohereConnection | None = None
    model: str = "embed-v4.0"
    input_type: Literal["search_document", "search_query", "classification", "clustering"] = Field(
        default="search_document",
        description="Cohere embedding task. Use search_document for chunks stored in a RAG index.",
    )
    batch_size: int = Field(default=32, ge=1, le=96)
    meta_fields_to_embed: list[str] = Field(default_factory=list)
    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description="Cohere provider truncation policy. NONE fails instead of silently dropping content.",
    )
    dimensions: Literal[256, 512, 1024, 1536] | None = Field(
        default=None,
        description="Embed v4 output dimensions. Must match the query embedder and vector index.",
    )
    document_embedder: CohereEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the CohereDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new CohereConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the CohereDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = CohereEmbedderComponent(
                connection=self.connection,
                model=self.model,
                client=self.client,
                input_type=self.input_type,
                batch_size=self.batch_size,
                meta_fields_to_embed=self.meta_fields_to_embed,
                truncate=self.truncate,
                dimensions=self.dimensions,
            )


class CohereTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified Cohere models.

    This class extends ConnectionNode to provide text embedding functionality using litellm embedding.

    Args:
        connection (Optional[CohereConnection]): An existing connection to Cohere API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Cohere model for text embeddings. Defaults to `embed-v4.0`.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (CohereConnection | None): The connection to Cohere API.
        model (str): The Cohere model identifier for text embeddings.
        text_embedder (CohereTextEmbedderComponent): The component for text embedding.

    """

    name: str = "cohere-text-embedder"
    connection: CohereConnection | None = None
    model: str = "embed-v4.0"
    input_type: Literal["search_query", "search_document", "classification", "clustering"] = Field(
        default="search_query",
        description="Cohere embedding task. Use search_query for RAG retrieval queries.",
    )
    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description="Cohere provider truncation policy. NONE fails instead of silently dropping query text.",
    )
    dimensions: Literal[256, 512, 1024, 1536] | None = Field(
        default=None,
        description="Embed v4 output dimensions. Must match document embeddings and the vector index.",
    )
    text_embedder: CohereEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the CohereTextEmbedder.

        If neither client nor connection is provided in kwargs, a new CohereConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the CohereTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = CohereEmbedderComponent(
                connection=self.connection,
                model=self.model,
                client=self.client,
                input_type=self.input_type,
                truncate=self.truncate,
                dimensions=self.dimensions,
            )
