from dynamiq.components.embedders.openai import OpenAIEmbedder as OpenAIEmbedderComponent
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class OpenAIDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using OpenAI's models.

    This class extends ConnectionNode to create embeddings for documents using OpenAI's API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OpenAIConnection | None): The connection to the OpenAI API.
        client (OpenAIClient | None): The OpenAI client instance.
        model (str): The model name to use for embedding.
        dimensions (int | None): The number of dimensions for the output embeddings.
        document_embedder (OpenAIDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[OpenAIConnection]): The connection to the OpenAI API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'text-embedding-3-small'.
        dimensions (Optional[int]): The number of dimensions for the output embeddings. Supported
            only by 'text-embedding-3' and later models. Defaults to None.
    """

    name: str = "OpenAIDocumentEmbedder"
    connection: OpenAIConnection | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    document_embedder: OpenAIEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the OpenAIDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new OpenAIConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the OpenAIDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = OpenAIEmbedderComponent(
                connection=self.connection,
                model=self.model,
                dimensions=self.dimensions,
                client=self.client,
            )


class OpenAITextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified OpenAI models.

    This class extends ConnectionNode to provide text embedding functionality using OpenAI's API.

    Args:
        connection (Optional[OpenAIConnection]): An existing connection to OpenAI's API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the OpenAI model for text embeddings. Defaults to
            'text-embedding-3-small'.
        dimensions (Optional[int]): Desired dimensionality of output embeddings. Defaults to None,
            using the model's default output dimensionality.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OpenAIConnection | None): The connection to OpenAI's API.
        client (OpenAIClient | None): The OpenAI client instance.
        model (str): The OpenAI model identifier for text embeddings.
        dimensions (int | None): The desired dimensionality of output embeddings.
        text_embedder (OpenAITextEmbedderComponent): The component for text embedding.

    Notes:
        The `dimensions` parameter is model-dependent and may not be supported by all models.
    """

    name: str = "OpenAITextEmbedder"
    connection: OpenAIConnection | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    text_embedder: OpenAIEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the OpenAITextEmbedder.

        If neither client nor connection is provided in kwargs, a new OpenAIConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the OpenAITextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = OpenAIEmbedderComponent(
                connection=self.connection,
                model=self.model,
                dimensions=self.dimensions,
                client=self.client,
            )
