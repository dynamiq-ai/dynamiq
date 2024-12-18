from dynamiq.components.embedders.mistral import MistralEmbedder as MistralEmbedderComponent
from dynamiq.connections import Mistral as MistralConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class MistralDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using Mistral models.

    This class extends ConnectionNode to create embeddings for documents using litellm embedding.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (MistralConnection | None): The connection to the Mistral API.
        model (str): The model name to use for embedding.
        document_embedder (MistralDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[MistralConnection]): The connection to the Mistral API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'mistral/mistral-embed'.
            only by 'text-embedding-3' and later models. Defaults to None.
    """

    name: str = "MistralDocumentEmbedder"
    connection: MistralConnection | None = None
    model: str = "mistral/mistral-embed"
    document_embedder: MistralEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the MistralDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new MistralConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the MistralDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = MistralEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )


class MistralTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified Mistral models.

    This class extends ConnectionNode to provide text embedding functionality using Mistral API.

    Args:
        connection (Optional[MistralConnection]): An existing connection to Mistral API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Mistral model for text embeddings. Defaults to
            'mistral/mistral-embed'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (MistralConnection | None): The connection to Mistral's API.
        model (str): The Mistral model identifier for text embeddings.
        text_embedder (MistralTextEmbedderComponent): The component for text embedding.

    """

    name: str = "MistralTextEmbedder"
    connection: MistralConnection | None = None
    model: str = "mistral/mistral-embed"
    text_embedder: MistralEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the MistralTextEmbedder.

        If neither client nor connection is provided in kwargs, a new MistralConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MistralConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the MistralTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = MistralEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )
