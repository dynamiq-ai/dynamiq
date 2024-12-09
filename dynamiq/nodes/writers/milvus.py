from dynamiq.connections import Milvus
from dynamiq.nodes.writers.base import Writer
from dynamiq.storages.vector import MilvusVectorStore
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams


class MilvusDocumentWriter(Writer, BaseWriterVectorStoreParams):
    """
    Document Writer Node using Milvus Vector Store.

    This class represents a node for writing documents to a Milvus Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Chroma | None): The connection to the Chroma Vector Store.
        vector_store (ChromaVectorStore | None): The Chroma Vector Store instance.
    """

    name: str = "MilvusDocumentWriter"
    connection: Milvus | None = None
    vector_store: MilvusVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the MilvusDocumentWriter.

        If no vector_store or connection is provided in kwargs, a default Milvus connection will be created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Milvus()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return MilvusVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(BaseWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }
