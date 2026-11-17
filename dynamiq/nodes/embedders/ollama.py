from dynamiq.components.embedders.ollama import OllamaEmbedder as OllamaEmbedderComponent
from dynamiq.connections import Ollama as OllamaConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder, TextEmbedder


class OllamaDocumentEmbedder(DocumentEmbedder):
    """
    Provides functionality to compute embeddings for documents using models served by Ollama.

    This class extends DocumentEmbedder to create embeddings for documents using litellm embedding.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OllamaConnection | None): The connection to the Ollama server.
        model (str): The model name to use for embedding.
        document_embedder (OllamaEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[OllamaConnection]): The connection to the Ollama server. A new connection
            to http://localhost:11434 is created if none is provided.
        model (str): The model name to use for embedding, with or without the "ollama/" prefix.
            Defaults to 'ollama/nomic-embed-text'.
    """

    name: str = "ollama-document-embedder"
    connection: OllamaConnection | None = None
    model: str = "ollama/nomic-embed-text"
    document_embedder: OllamaEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """
        Initializes the OllamaDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new OllamaConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OllamaConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initializes the components of the OllamaDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = OllamaEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )


class OllamaTextEmbedder(TextEmbedder):
    """
    A component designed to embed strings using models served by Ollama.

    This class extends TextEmbedder to provide text embedding functionality using an Ollama server.

    Args:
        connection (Optional[OllamaConnection]): An existing connection to the Ollama server. A new
            connection to http://localhost:11434 is created if none is provided.
        model (str): The identifier of the Ollama model for text embeddings, with or without the
            "ollama/" prefix. Defaults to 'ollama/nomic-embed-text'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OllamaConnection | None): The connection to the Ollama server.
        model (str): The Ollama model identifier for text embeddings.
        text_embedder (OllamaEmbedderComponent): The component for text embedding.
    """

    name: str = "ollama-text-embedder"
    connection: OllamaConnection | None = None
    model: str = "ollama/nomic-embed-text"
    text_embedder: OllamaEmbedderComponent | None = None

    def __init__(self, **kwargs):
        """Initialize the OllamaTextEmbedder.

        If neither client nor connection is provided in kwargs, a new OllamaConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OllamaConnection()
        super().__init__(**kwargs)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the OllamaTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = OllamaEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )
