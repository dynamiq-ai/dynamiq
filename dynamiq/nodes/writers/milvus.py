from dynamiq.connections import Milvus
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import MilvusVectorStore
from dynamiq.storages.vector.milvus.milvus import MilvusVectorStoreParams
from dynamiq.utils.logger import logger


class MilvusDocumentWriter(Writer, MilvusVectorStoreParams):
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
        return self.model_dump(include=set(MilvusVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(self, input_data: WriterInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document writing process.

        This method writes the input documents to the Milvus Vector Store.

        Args:
            input_data (WriterInputSchema): An instance containing the input data.
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

        documents = input_data.documents
        content_key = input_data.content_key
        embedding_key = input_data.embedding_key

        # Write documents to Milvus
        upserted_count = self.vector_store.write_documents(
            documents, content_key=content_key, embedding_key=embedding_key
        )
        logger.debug(f"Upserted {upserted_count} documents to Milvus Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
