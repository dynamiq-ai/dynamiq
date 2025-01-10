from typing import Any

from dynamiq.components.retrievers.pgvector import PGVectorDocumentRetriever as PGVectorDocumentRetrieverComponent
from dynamiq.connections import PostgreSQL
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.retrievers.base import Retriever, RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PGVectorStore
from dynamiq.storages.vector.pgvector.pgvector import PGVectorStoreParams, PGVectorStoreRetrieverParams


class PGVectorDocumentRetriever(Retriever, PGVectorStoreRetrieverParams):
    """
    Document Retriever using PGVector.

    This class implements a document retriever that uses PGVector as the underlying vector store.
    It extends the VectorStoreNode class and provides functionality to retrieve documents
    based on vector similarity.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group the node belongs to.
        name (str): The name of the node.
        vector_store (PGVectorStore | None): The PGVectorStore instance.
        filters (dict[str, Any] | None): Filters to apply when retrieving documents.
        top_k (int): The maximum number of documents to retrieve.
        document_retriever (PGVectorDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Keyword arguments for initializing the node.
    """

    name: str = "PGVectorDocumentRetriever"
    connection: PostgreSQL | None = None
    vector_store: PGVectorStore | None = None
    document_retriever: PGVectorDocumentRetrieverComponent | None = None

    def __init__(self, **kwargs):
        """
        Initialize the PGVectorDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default PostgreSQL connection will be created.

        Args:
            **kwargs: Keyword arguments for initializing the node.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = PostgreSQL()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return PGVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(PGVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def init_components(self, connection_manager: ConnectionManager = ConnectionManager()):
        """
        Initialize the components of the PGVectorDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = PGVectorDocumentRetrieverComponent(
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

        alpha = input_data.alpha or self.alpha
        query = input_data.query

        output = self.document_retriever.run(
            query_embedding,
            filters=filters,
            top_k=top_k,
            content_key=content_key,
            embedding_key=embedding_key,
            query=query,
            alpha=alpha,
        )

        return {
            "documents": output["documents"],
        }
