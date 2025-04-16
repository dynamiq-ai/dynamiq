from dynamiq.components.embedders.gemini import GeminiEmbedder as GeminiEmbedderComponent
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class GeminiDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using Gemini models.

    This class extends DocumentEmbedder to create embeddings for documents using Gemini API.

    Attributes:
        name (str): The name of the node.
        connection (GeminiConnection | None): The connection to the Gemini API.
        model (str): The model name to use for embedding.
        document_embedder (GeminiEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[GeminiConnection]): The connection to the Gemini API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'gemini/gemini-embedding-exp-03-07'.
        input_type (str): Specifies the type of embedding task. Defaults to "search_document".
    """

    name: str = "GeminiDocumentEmbedder"
    connection: GeminiConnection | None = None
    model: str = "gemini/gemini-embedding-exp-03-07"
    input_type: str = "search_document"
    document_embedder: GeminiEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the GeminiDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new GeminiConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the GeminiDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = GeminiEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client, input_type=self.input_type
            )


class GeminiTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using specified Gemini models.

    This class extends TextEmbedder to provide text embedding functionality using Gemini API.

    Args:
        connection (Optional[GeminiConnection]): An existing connection to Gemini API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Gemini model for text embeddings. Defaults to
            'gemini/gemini-embedding-exp-03-07'.
        input_type (str): Specifies the type of embedding task. Defaults to "search_query".

    Attributes:
        name (str): The name of the node.
        connection (GeminiConnection | None): The connection to Gemini API.
        model (str): The Gemini model identifier for text embeddings.
        text_embedder (GeminiEmbedderComponent): The component for text embedding.
    """

    name: str = "GeminiTextEmbedder"
    connection: GeminiConnection | None = None
    model: str = "gemini/gemini-embedding-exp-03-07"
    input_type: str = "search_query"
    text_embedder: GeminiEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the GeminiTextEmbedder.

        If neither client nor connection is provided in kwargs, a new GeminiConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the GeminiTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = GeminiEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client, input_type=self.input_type
            )
