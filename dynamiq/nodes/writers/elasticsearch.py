from dynamiq.connections import Elasticsearch
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.writers.base import Writer, WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.storages.vector.elasticsearch.elasticsearch import ElasticsearchVectorStoreWriterParams
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.utils.logger import logger


class ElasticsearchDocumentWriter(Writer, ElasticsearchVectorStoreWriterParams):
    """
    Document Writer Node using Elasticsearch Vector Store.

    This class represents a node for writing documents to an Elasticsearch Vector Store.
    It supports vector search, BM25 text search, and hybrid search capabilities.

    Attributes:
        group (Literal[NodeGroup.WRITERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (Optional[Elasticsearch]): The Elasticsearch connection.
        vector_store (Optional[ElasticsearchVectorStore]): The Elasticsearch Vector Store instance.
    """

    name: str = "ElasticsearchDocumentWriter"
    connection: Elasticsearch | str | None = None
    vector_store: ElasticsearchVectorStore | None = None

    def __init__(self, **kwargs):
        """
        Initialize the ElasticsearchDocumentWriter.

        If neither vector_store nor connection is provided in kwargs, a default Elasticsearch connection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Elasticsearch()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return ElasticsearchVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(ElasticsearchVectorStoreWriterParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def execute(
        self,
        input_data: WriterInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, int]:
        """
        Execute the document writing operation.

        This method writes the input documents to the Elasticsearch Vector Store.
        It supports:
        - Vector embeddings for similarity search
        - Text content for BM25 search
        - Metadata for filtering and custom ranking
        - Custom mappings and analyzers
        - Index settings and templates
        - Duplicate handling with configurable policy
        - Batch operations with configurable size

        Args:
            input_data (WriterInputSchema): Input data containing the documents to be written.
            config (Optional[RunnableConfig]): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, int]: A dictionary containing the count of written documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        upserted_count = self.vector_store.write_documents(
            documents=input_data.documents,
            policy=DuplicatePolicy.FAIL,
            batch_size=None,
            content_key=input_data.content_key,
            embedding_key=input_data.embedding_key,
        )
        logger.debug(f"Upserted {upserted_count} documents to Elasticsearch Vector Store.")

        return {
            "upserted_count": upserted_count,
        }
