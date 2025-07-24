from dynamiq.connections import Weaviate
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.storages.vector.weaviate import WeaviateWriterVectorStoreParams
from dynamiq.utils.logger import logger


class WeaviateDocumentWriter(Writer, WeaviateWriterVectorStoreParams):
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
        return self.model_dump(include=set(WeaviateWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        This method writes the input documents to the Weaviate Vector Store.

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
        logger.debug(f"Upserted {upserted_count} documents to Weaviate Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
