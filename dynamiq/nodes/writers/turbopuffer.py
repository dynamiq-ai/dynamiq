from dynamiq.connections import Turbopuffer
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import TurbopufferVectorStore
from dynamiq.storages.vector.turbopuffer import TurbopufferWriterVectorStoreParams
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class TurbopufferDocumentWriter(Writer, TurbopufferWriterVectorStoreParams):
    """
    Document Writer Node using Turbopuffer Vector Store.

    This class represents a node for writing documents to a Turbopuffer namespace, named by index_name.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Turbopuffer | None): The Turbopuffer connection.
        vector_store (TurbopufferVectorStore | None): The Turbopuffer Vector Store instance.
    """

    name: str = "turbopuffer-document-writer"
    connection: Turbopuffer | None = None
    vector_store: TurbopufferVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the TurbopufferDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Turbopuffer connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Turbopuffer()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return TurbopufferVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(TurbopufferWriterVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        Args:
            input_data (WriterInputSchema): Input data containing the documents to be written.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the count of upserted documents.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents
        content_key = input_data.content_key

        upserted_count = self.vector_store.write_documents(documents, content_key=content_key)
        logger.debug(f"Upserted {upserted_count} documents to Turbopuffer Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
