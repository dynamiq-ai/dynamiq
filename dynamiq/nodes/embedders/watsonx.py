from dynamiq.components.embedders.watsonx import WatsonXEmbedder as WatsonXEmbedderComponent
from dynamiq.connections import WatsonX as WatsonXConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class WatsonXDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using WatsonX models.

    This class extends ConnectionNode to create embeddings for documents using litellm embedding.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (WatsonXConnection | None): The connection to the WatsonX API.
        model (str): The model name to use for embedding.
        document_embedder (WatsonXDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[WatsonXConnection]): The connection to the WatsonX API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'watsonx/ibm/slate-30m-english-rtrvr'.
    """

    name: str = "WatsonXDocumentEmbedder"
    connection: WatsonXConnection | None = None
    model: str = "watsonx/ibm/slate-30m-english-rtrvr"
    document_embedder: WatsonXEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the WatsonXDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new WatsonXConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = WatsonXConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the WatsonXDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = WatsonXEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )


class WatsonXTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified WatsonX models.

    This class extends ConnectionNode to provide text embedding functionality using WatsonX API.

    Args:
        connection (Optional[WatsonXConnection]): An existing connection to WatsonX API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the WatsonX model for text embeddings. Defaults to
            'watsonx/ibm/slate-30m-english-rtrvr'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (WatsonXConnection | None): The connection to WatsonX's API.
        model (str): The WatsonX model identifier for text embeddings.
        text_embedder (WatsonXTextEmbedderComponent): The component for text embedding.

    """

    name: str = "WatsonXTextEmbedder"
    connection: WatsonXConnection | None = None
    model: str = "watsonx/ibm/slate-30m-english-rtrvr"
    text_embedder: WatsonXEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the WatsonXTextEmbedder.

        If neither client nor connection is provided in kwargs, a new WatsonXConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = WatsonXConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the WatsonXTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = WatsonXEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )
