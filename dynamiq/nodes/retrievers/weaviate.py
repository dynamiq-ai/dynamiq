from typing import Any, Literal

from dynamiq.components.retrievers.weaviate import (
    WeaviateDocumentRetriever as WeaviateDocumentRetrieverComponent,
)
from dynamiq.connections import Weaviate
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import NodeGroup, VectorStoreNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore


class WeaviateDocumentRetriever(VectorStoreNode):
    """Document Retriever using Weaviate.

    This class implements a document retriever that uses Weaviate as the vector store backend.

    Args:
        vector_store (WeaviateVectorStore, optional): An instance of WeaviateVectorStore to interface
            with Weaviate vectors.
        filters (dict[str, Any], optional): Filters to apply for retrieving specific documents.
            Defaults to None.
        top_k (int, optional): The maximum number of documents to return. Defaults to 10.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (WeaviateVectorStore | None): The WeaviateVectorStore instance.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        document_retriever (WeaviateDocumentRetrieverComponent): The document retriever component.
    """

    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    name: str = "WeaviateDocumentRetriever"
    connection: Weaviate | None = None
    vector_store: WeaviateVectorStore | None = None
    filters: dict[str, Any] | None = None
    top_k: int = 10
    document_retriever: WeaviateDocumentRetrieverComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the WeaviateDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Weaviate connection will be created.

        Args:
            **kwargs: Keyword arguments to initialize the retriever.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Weaviate()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return WeaviateVectorStore

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the retriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = WeaviateDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the document retrieval process.

        This method retrieves documents based on the input embedding.

        Args:
            input_data (dict[str, Any]): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query_embedding = input_data["embedding"]
        filters = input_data.get("filters") or self.filters
        top_k = input_data.get("top_k") or self.top_k

        output = self.document_retriever.run(query_embedding, filters=filters, top_k=top_k)

        return {
            "documents": output["documents"],
        }
