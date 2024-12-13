from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore, QdrantWriterVectorStoreParams
from dynamiq.utils.logger import logger


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

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        This method writes the input documents to the Qdrant Vector Store.

        Args:
            input_data (WriterInputSchema): Input data containing the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the count of upserted documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents
        content_key = input_data.content_key

        upserted_count = self.vector_store.write_documents(documents, content_key=content_key)
        logger.debug(f"Upserted {upserted_count} documents to Qdrant Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
