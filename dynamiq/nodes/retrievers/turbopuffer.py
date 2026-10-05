from typing import Any

from dynamiq.components.retrievers.turbopuffer import (
    TurbopufferDocumentRetriever as TurbopufferDocumentRetrieverComponent,
)
from dynamiq.connections import Turbopuffer
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.retrievers.base import Retriever, RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import TurbopufferVectorStore
from dynamiq.storages.vector.turbopuffer import TurbopufferRetrieverVectorStoreParams
from dynamiq.types.cancellation import check_cancellation


class TurbopufferDocumentRetriever(Retriever, TurbopufferRetrieverVectorStoreParams):
    """Document Retriever using Turbopuffer.

    This class implements a document retriever that uses a Turbopuffer namespace, named by index_name,
    as the vector store backend.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (TurbopufferVectorStore | None): The TurbopufferVectorStore instance.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        alpha (float): The weight of vector similarity in hybrid retrieval.
        document_retriever (TurbopufferDocumentRetrieverComponent): The document retriever component.
    """

    name: str = "turbopuffer-document-retriever"
    connection: Turbopuffer | None = None
    vector_store: TurbopufferVectorStore | None = None
    document_retriever: TurbopufferDocumentRetrieverComponent | None = None

    def __init__(self, **kwargs):
        """
        Initialize the TurbopufferDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Turbopuffer connection will be
        created.

        Args:
            **kwargs: Keyword arguments to initialize the retriever.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Turbopuffer()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return TurbopufferVectorStore

    @property
    def vector_store_params(self):
        params = self.model_dump(include=set(TurbopufferRetrieverVectorStoreParams.model_fields))
        params.pop("max_vector_distance", None)
        params.update(
            {
                "connection": self.connection,
                "client": self.client,
            }
        )
        return params

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the retriever.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = TurbopufferDocumentRetrieverComponent(
                vector_store=self.vector_store,
                filters=self.filters,
                top_k=self.top_k,
                similarity_threshold=self.similarity_threshold,
                max_vector_distance=self.max_vector_distance,
            )

    def execute(self, input_data: RetrieverInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the document retrieval process.

        Args:
            input_data (RetrieverInputSchema): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        similarity_threshold = (
            input_data.similarity_threshold
            if input_data.similarity_threshold is not None
            else self.similarity_threshold
        )
        max_vector_distance = (
            input_data.max_vector_distance if input_data.max_vector_distance is not None else self.max_vector_distance
        )

        output = self.document_retriever.run(
            input_data.embedding,
            filters=input_data.filters or self.filters,
            top_k=input_data.top_k or self.top_k,
            content_key=input_data.content_key,
            query=input_data.query,
            alpha=input_data.alpha if input_data.alpha is not None else self.alpha,
            similarity_threshold=similarity_threshold,
            max_vector_distance=max_vector_distance,
        )

        return {
            "documents": output["documents"],
        }
