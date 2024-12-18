from typing import Any

from dynamiq.components.retrievers.milvus import MilvusDocumentRetriever as MilvusDocumentRetrieverComponent
from dynamiq.connections import Milvus
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.retrievers.base import Retriever, RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import MilvusVectorStore
from dynamiq.storages.vector.milvus.milvus import MilvusVectorStoreParams


class MilvusDocumentRetriever(Retriever, MilvusVectorStoreParams):
    """
    Document Retriever using Milvus.

    This class implements a document retriever that uses Milvus as the underlying vector store.
    It extends the VectorStoreNode class and provides functionality to retrieve documents
    based on vector similarity.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group the node belongs to.
        name (str): The name of the node.
        vector_store (MilvusVectorStore | None): The MilvusVectorStore instance.
        filters (dict[str, Any] | None): Filters to apply when retrieving documents.
        top_k (int): The maximum number of documents to retrieve.
        document_retriever (MilvusDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Keyword arguments for initializing the node.
    """

    name: str = "MilvusDocumentRetriever"
    connection: Milvus | None = None
    vector_store: MilvusVectorStore | None = None
    document_retriever: MilvusDocumentRetrieverComponent | None = None

    def __init__(self, **kwargs):
        """
        Initialize the MilvusDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Milvus connection will be created.

        Args:
            **kwargs: Keyword arguments for initializing the node.
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

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the MilvusDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = MilvusDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )

    def execute(self, input_data: RetrieverInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the document retrieval process.

        This method takes an input embedding, retrieves similar documents using the
        document retriever component, and returns the retrieved documents.

        Args:
            input_data (RetrieverInputSchema): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query_embedding = input_data.embedding
        content_key = input_data.content_key
        embedding_key = input_data.embedding_key
        filters = input_data.filters or self.filters
        top_k = input_data.top_k or self.top_k

        output = self.document_retriever.run(
            query_embedding, filters=filters, top_k=top_k, content_key=content_key, embedding_key=embedding_key
        )

        return {
            "documents": output["documents"],
        }
