from typing import Any

from pydantic import BaseModel, Field

from dynamiq.components.retrievers.elasticsearch import (
    ElasticsearchDocumentRetriever as ElasticsearchDocumentRetrieverComponent,
)
from dynamiq.connections import Elasticsearch
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.storages.vector.elasticsearch.elasticsearch import ElasticsearchVectorStoreParams


class ElasticsearchRetrieverInputSchema(BaseModel):
    """Input schema for Elasticsearch retriever."""

    query: list[float] = Field(..., description="Vector query for similarity search")
    filters: dict[str, Any] = Field(default={}, description="Filters to apply for retrieving specific documents")
    top_k: int = Field(default=0, description="Number of documents to retrieve")
    exclude_document_embeddings: bool = Field(default=True, description="Whether to exclude embeddings in response")
    scale_scores: bool = Field(default=False, description="Whether to scale scores to 0-1 range")
    content_key: str = Field(default="content", description="Key to use for content in response")
    embedding_key: str = Field(default="embedding", description="Key to use for embedding in response")


class ElasticsearchDocumentRetriever(Retriever, ElasticsearchVectorStoreParams):
    """Document Retriever using Elasticsearch for vector similarity search.

    This class implements a document retriever that uses Elasticsearch as the underlying store
    for vector similarity search with optional metadata filtering.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group the node belongs to.
        name (str): The name of the node.
        vector_store (ElasticsearchVectorStore | None): The ElasticsearchVectorStore instance.
        filters (dict[str, Any] | None): Filters to apply when retrieving documents.
        top_k (int): The maximum number of documents to retrieve.
        document_retriever (ElasticsearchDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Keyword arguments for initializing the node.
    """

    name: str = "ElasticsearchDocumentRetriever"
    connection: Elasticsearch | None = None
    vector_store: ElasticsearchVectorStore | None = None
    document_retriever: ElasticsearchDocumentRetrieverComponent | None = None
    input_schema = ElasticsearchRetrieverInputSchema

    def __init__(self, **kwargs):
        """Initialize the ElasticsearchDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs,
        a default Elasticsearch connection will be created.

        Args:
            **kwargs: Keyword arguments for initializing the node.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Elasticsearch()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return ElasticsearchVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(ElasticsearchVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def init_components(self, connection_manager: ConnectionManager = ConnectionManager()):
        """Initialize the components of the ElasticsearchDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = ElasticsearchDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )

    def execute(
        self, input_data: ElasticsearchRetrieverInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the document retrieval process.

        This method takes input data containing the vector query and search parameters,
        retrieves relevant documents using vector similarity search,
        and returns the retrieved documents.

        Args:
            input_data (ElasticsearchRetrieverInputSchema): The input data containing:
                - query: Vector query for similarity search
                - filters: Optional metadata filters
                - top_k: Number of documents to retrieve
                - exclude_document_embeddings: Whether to exclude embeddings
                - scale_scores: Whether to scale scores to 0-1 range
            config (RunnableConfig, optional): The configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        output = self.document_retriever.run(
            query=input_data.query,
            filters=input_data.filters or self.filters,
            top_k=input_data.top_k or self.top_k,
            exclude_document_embeddings=input_data.exclude_document_embeddings,
            scale_scores=input_data.scale_scores,
            content_key=input_data.content_key,
            embedding_key=input_data.embedding_key,
        )

        return {
            "documents": output["documents"],
        }
