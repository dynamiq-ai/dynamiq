from dynamiq.connections import Weaviate
from dynamiq.nodes.writers.base import Writer
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams
from dynamiq.storages.vector.weaviate.weaviate import WeaviateVectorStore


class WeaviateDocumentWriter(Writer, BaseWriterVectorStoreParams):
    """
    Document Writer Node using Weaviate Vector Store.

    This class represents a node for writing documents to a Weaviate Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Weaviate | None): The Weaviate connection.
        vector_store (WeaviateVectorStore | None): The Weaviate Vector Store instance.
    """

    name: str = "WeaviateDocumentWriter"
    connection: Weaviate | None = None
    vector_store: WeaviateVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the WeaviateDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Weaviate connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Weaviate()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return WeaviateVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(BaseWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }
