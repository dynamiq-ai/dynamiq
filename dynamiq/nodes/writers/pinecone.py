from typing import Any, Literal

from dynamiq.connections import Pinecone
from dynamiq.nodes.node import NodeGroup, VectorStoreNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PineconeVectorStore
from dynamiq.storages.vector.pinecone.pinecone import PineconeWriterVectorStoreParams
from dynamiq.utils.logger import logger


class PineconeDocumentWriter(VectorStoreNode, PineconeWriterVectorStoreParams):
    """
    Document Writer Node using Pinecone Vector Store.

    This class represents a node for writing documents to a Pinecone Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Pinecone | None): The Pinecone connection object.
        vector_store (PineconeVectorStore | None): The Pinecone Vector Store object.
    """

    group: Literal[NodeGroup.WRITERS] = NodeGroup.WRITERS
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

    @property
    def vector_store_cls(self):
        return PineconeVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(PineconeWriterVectorStoreParams.model_fields)) | {"client": self.client}

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Execute the document writing process.

        This method writes the input documents to the Pinecone Vector Store.

        Args:
            input_data (dict[str, Any]): A dictionary containing the input data.
                Expected to have a 'documents' key with the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the number of upserted documents.

        Raises:
            Any exceptions raised by the vector store's write_documents method.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data["documents"]

        upserted_count = self.vector_store.write_documents(documents)
        logger.debug(f"Upserted {upserted_count} documents to Pinecone Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
