from dynamiq.components.retrievers.chroma import ChromaDocumentRetriever as ChromaDocumentRetrieverComponent
from dynamiq.connections import Chroma
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.storages.vector import ChromaVectorStore


class ChromaDocumentRetriever(Retriever):
    """
    Document Retriever using Chroma.

    This class implements a document retriever that uses Chroma as the underlying vector store.
    It extends the VectorStoreNode class and provides functionality to retrieve documents
    based on vector similarity.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group the node belongs to.
        name (str): The name of the node.
        vector_store (ChromaVectorStore | None): The ChromaVectorStore instance.
        filters (dict[str, Any] | None): Filters to apply when retrieving documents.
        top_k (int): The maximum number of documents to retrieve.
        document_retriever (ChromaDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Keyword arguments for initializing the node.
    """

    name: str = "ChromaDocumentRetriever"
    connection: Chroma | None = None
    vector_store: ChromaVectorStore | None = None
    document_retriever: ChromaDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the ChromaDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Chroma connection will be created.

        Args:
            **kwargs: Keyword arguments for initializing the node.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Chroma()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return ChromaVectorStore

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the ChromaDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = ChromaDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )
