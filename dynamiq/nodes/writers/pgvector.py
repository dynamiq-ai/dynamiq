from dynamiq.connections import PostgreSQL
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PGVectorStore
from dynamiq.storages.vector.pgvector.pgvector import PGVectorStoreWriterParams
from dynamiq.utils.logger import logger


class PGVectorDocumentWriter(Writer, PGVectorStoreWriterParams):
    """
    Document Writer Node using PGVector Vector Store.

    This class represents a node for writing documents to a PGVector Vector Store.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (PostgreSQL | None): The PostgreSQL connection.
        vector_store (PGVectorStore | None): The PGVector Vector Store instance.
    """

    name: str = "PGVectorDocumentWriter"
    connection: PostgreSQL | str | None = None
    vector_store: PGVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the PGVectorDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default PostgreSQL connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = PostgreSQL()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return PGVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(PGVectorStoreWriterParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing operation.

        This method writes the input documents to the PGVector Vector Store.

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
        embedding_key = input_data.embedding_key

        upserted_count = self.vector_store.write_documents(
            documents, content_key=content_key, embedding_key=embedding_key
        )
        logger.debug(f"Upserted {upserted_count} documents to PGVector Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
