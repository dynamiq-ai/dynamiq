from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.nodes.writers.base import Writer
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore, QdrantWriterVectorStoreParams


class QdrantDocumentWriter(Writer, QdrantWriterVectorStoreParams):
    """
    Document Writer Node using Qdrant Vector Store.

    This class represents a node for writing documents to a Weaviate Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Qdrant | None): The Qdrant connection.
        vector_store (QdrantVectorStore | None): The Qdrant Vector Store instance.
    """

    name: str = "QdrantDocumentWriter"
    connection: QdrantConnection | None = None
    vector_store: QdrantVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the QdrantDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Qdrant connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = QdrantConnection()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return QdrantVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(QdrantWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }
