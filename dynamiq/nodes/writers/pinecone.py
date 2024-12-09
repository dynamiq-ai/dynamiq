from pydantic import model_validator
from dynamiq.connections import Pinecone
from dynamiq.nodes.writers.base import Writer
from dynamiq.storages.vector import PineconeVectorStore
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType, PineconeWriterVectorStoreParams


class PineconeDocumentWriter(Writer, PineconeWriterVectorStoreParams):
    """
    Document Writer Node using Pinecone Vector Store.

    This class represents a node for writing documents to a Pinecone Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Pinecone | None): The Pinecone connection object.
        vector_store (PineconeVectorStore | None): The Pinecone Vector Store object.
    """

    name: str = "PineconeDocumentWriter"
    connection: Pinecone | None = None
    vector_store: PineconeVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the PineconeDocumentWriter.

        If no vector_store or connection is provided in kwargs, a default Pinecone connection will be created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Pinecone()
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def check_required_params(self) -> "PineconeDocumentWriter":
        """
        Validate required parameters

        Returns:
            self: The updated instance.
        """
        if self.vector_store is None:
            if self.create_if_not_exist and self.index_type is None:
                raise ValueError("Index type 'pod' or 'serverless' must be specified when creating an index")

            if self.index_type == PineconeIndexType.POD and (self.environment is None or self.pod_type is None):
                raise ValueError("'environment' and 'pod_type' must be specified for 'pod' index")

            if self.index_type == PineconeIndexType.SERVERLESS and (self.cloud is None or self.region is None):
                raise ValueError("'cloud' and 'region' must be specified for 'serverless' index")

        return self

    @property
    def vector_store_cls(self):
        return PineconeVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(PineconeWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }
