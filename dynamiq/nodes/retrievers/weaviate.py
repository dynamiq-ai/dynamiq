from dynamiq.components.retrievers.weaviate import WeaviateDocumentRetriever as WeaviateDocumentRetrieverComponent
from dynamiq.connections import Weaviate
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.storages.vector import WeaviateVectorStore


class WeaviateDocumentRetriever(Retriever):
    """Document Retriever using Weaviate.

    This class implements a document retriever that uses Weaviate as the vector store backend.

    Args:
        vector_store (WeaviateVectorStore, optional): An instance of WeaviateVectorStore to interface
            with Weaviate vectors.
        filters (dict[str, Any], optional): Filters to apply for retrieving specific documents.
            Defaults to None.
        top_k (int, optional): The maximum number of documents to return. Defaults to 10.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (WeaviateVectorStore | None): The WeaviateVectorStore instance.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        document_retriever (WeaviateDocumentRetrieverComponent): The document retriever component.
    """

    name: str = "WeaviateDocumentRetriever"
    connection: Weaviate | None = None
    vector_store: WeaviateVectorStore | None = None
    document_retriever: WeaviateDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the WeaviateDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Weaviate connection will be created.

        Args:
            **kwargs: Keyword arguments to initialize the retriever.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Weaviate()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return WeaviateVectorStore

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
            self.document_retriever = WeaviateDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )
