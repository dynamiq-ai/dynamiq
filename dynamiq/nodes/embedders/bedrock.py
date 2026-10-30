from typing import Literal

from pydantic import Field

from dynamiq.components.embedders.bedrock import BedrockEmbedder as BedrockEmbedderComponent
from dynamiq.connections import AWS as BedrockConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class BedrockDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using Bedrock models.

    This class extends ConnectionNode to create embeddings for documents using Bedrock API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (BedrockConnection | None): The connection to the Bedrock API.
        model (str): The model name to use for embedding.
        document_embedder (BedrockDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[BedrockConnection]): The connection to the Bedrock API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'amazon.titan-embed-text-v1'.
    """

    name: str = "bedrock-document-embedder"
    connection: BedrockConnection | None = None
    model: str = "amazon.titan-embed-text-v1"
    input_type: Literal["search_document", "search_query", "classification", "clustering"] = Field(
        default="search_document",
        description="Cohere embedding task, sent only for Cohere models. "
        "Use search_document for chunks stored in a RAG index.",
    )
    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description="Cohere truncation policy, sent only for Cohere models. NONE fails instead of dropping content.",
    )
    dimensions: Literal[256, 512, 1024, 1536] | None = Field(
        default=None,
        description="Output dimensions, sent only for Cohere models that support it (embed-v4). "
        "Must match the other embedder and the vector index.",
    )
    document_embedder: BedrockEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the BedrockDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new BedrockConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = BedrockConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the BedrockDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = BedrockEmbedderComponent(
                connection=self.connection,
                model=self.model,
                client=self.client,
                input_type=self.input_type,
                truncate=self.truncate,
                dimensions=self.dimensions,
            )


class BedrockTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified Cohere models.

    This class extends ConnectionNode to provide text embedding functionality using Bedrock API.

    Args:
        connection (Optional[BedrockConnection]): An existing connection to Bedrock API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Bedrock model for text embeddings. Defaults to
            'amazon.titan-embed-text-v1'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (BedrockConnection | None): The connection to Bedrock API.
        model (str): The Bedrock model identifier for text embeddings.
        text_embedder (BedrockTextEmbedderComponent): The component for text embedding.

    """

    name: str = "bedrock-text-embedder"
    connection: BedrockConnection | None = None
    model: str = "amazon.titan-embed-text-v1"
    input_type: Literal["search_document", "search_query", "classification", "clustering"] = Field(
        default="search_query",
        description="Cohere embedding task, sent only for Cohere models. Use search_query for RAG retrieval queries.",
    )
    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description="Cohere truncation policy, sent only for Cohere models. NONE fails instead of dropping content.",
    )
    dimensions: Literal[256, 512, 1024, 1536] | None = Field(
        default=None,
        description="Output dimensions, sent only for Cohere models that support it (embed-v4). "
        "Must match the other embedder and the vector index.",
    )
    text_embedder: BedrockEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the BedrockTextEmbedder.

        If neither client nor connection is provided in kwargs, a new BedrockConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = BedrockConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the BedrockTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = BedrockEmbedderComponent(
                connection=self.connection,
                model=self.model,
                client=self.client,
                input_type=self.input_type,
                truncate=self.truncate,
                dimensions=self.dimensions,
            )
