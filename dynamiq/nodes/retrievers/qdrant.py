from dynamiq.components.retrievers.qdrant import QdrantDocumentRetriever as QdrantDocumentRetrieverComponent
from dynamiq.connections import Qdrant
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.storages.vector import QdrantVectorStore


class QdrantDocumentRetriever(Retriever):
    """Document Retriever using Qdrant.

    This class implements a document retriever that uses Qdrant as the vector store backend.

    Args:
        vector_store (QdrantVectorStore, optional): An instance of QdrantVectorStore to interface
            with Qdrant vectors.
        filters (dict[str, Any], optional): Filters to apply for retrieving specific documents.
            Defaults to None.
        top_k (int, optional): The maximum number of documents to return. Defaults to 10.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (QdrantVectorStore | None): The QdrantVectorStore instance.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        document_retriever (QdrantDocumentRetrieverComponent): The document retriever component.
    """

    name: str = "QdrantDocumentRetriever"
    connection: Qdrant | None = None
    vector_store: QdrantVectorStore | None = None
    document_retriever: QdrantDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the QdrantDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Qdrant connection will be created.

        Args:
            **kwargs: Keyword arguments to initialize the retriever.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Qdrant()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return QdrantVectorStore

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the retriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = QdrantDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )
