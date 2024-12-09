from dynamiq.components.retrievers.milvus import MilvusDocumentRetriever as MilvusDocumentRetrieverComponent
from dynamiq.connections import Milvus
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.storages.vector import MilvusVectorStore


class MilvusDocumentRetriever(Retriever):
    """
    Document Retriever using Milvus.

    This class implements a document retriever that uses Milvus as the underlying vector store.
    It extends the VectorStoreNode class and provides functionality to retrieve documents
    based on vector similarity.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group the node belongs to.
        name (str): The name of the node.
        vector_store (MilvusVectorStore | None): The MilvusVectorStore instance.
        filters (dict[str, Any] | None): Filters to apply when retrieving documents.
        top_k (int): The maximum number of documents to retrieve.
        document_retriever (MilvusDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Keyword arguments for initializing the node.
    """

    name: str = "MilvusDocumentRetriever"
    connection: Milvus | None = None
    vector_store: MilvusVectorStore | None = None
    document_retriever: MilvusDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the MilvusDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Milvus connection will be created.

        Args:
            **kwargs: Keyword arguments for initializing the node.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Milvus()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return MilvusVectorStore

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the MilvusDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = MilvusDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )
