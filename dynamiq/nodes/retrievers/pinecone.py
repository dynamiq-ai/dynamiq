from dynamiq.components.retrievers.pinecone import PineconeDocumentRetriever as PineconeDocumentRetrieverComponent
from dynamiq.connections import Pinecone
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.retrievers.base import Retriever, RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PineconeVectorStore
from dynamiq.storages.vector.pinecone.pinecone import PineconeVectorStoreParams


class PineconeDocumentRetriever(Retriever, PineconeVectorStoreParams):
    """Document Retriever using Pinecone.

    This class implements a document retriever that uses Pinecone as the vector store backend.

    Attributes:
        group (Literal[NodeGroup.RETRIEVERS]): The group of the node.
        name (str): The name of the node.
        vector_store (PineconeVectorStore | None): The Pinecone vector store.
        filters (dict[str, Any] | None): Filters to apply for retrieving specific documents.
        top_k (int): The maximum number of documents to return.
        document_retriever (PineconeDocumentRetrieverComponent): The document retriever component.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """

    name: str = "PineconeDocumentRetriever"
    connection: Pinecone | None = None
    vector_store: PineconeVectorStore | None = None
    document_retriever: PineconeDocumentRetrieverComponent | None = None

    def __init__(self, **kwargs):
        """
        Initialize the PineconeDocumentRetriever.

        If neither vector_store nor connection is provided in kwargs, a default Pinecone connection will be created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("vector_store") is None and kwargs.get("connection") is None:
            kwargs["connection"] = Pinecone()
        super().__init__(**kwargs)

    @property
    def vector_store_cls(self):
        return PineconeVectorStore

    @property
    def vector_store_params(self):
        return self.model_dump(include=set(PineconeVectorStoreParams.model_fields)) | {
            "connection": self.connection,
            "client": self.client,
        }

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize the components of the PineconeDocumentRetriever.

        This method sets up the document retriever component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use.
                Defaults to a new ConnectionManager instance.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_retriever is None:
            self.document_retriever = PineconeDocumentRetrieverComponent(
                vector_store=self.vector_store, filters=self.filters, top_k=self.top_k
            )

    def execute(self, input_data: RetrieverInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Execute the document retrieval process.

        This method retrieves documents based on the input embedding.

        Args:
            input_data (RetrieverInputSchema): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query_embedding = input_data.embedding
        content_key = input_data.content_key
        filters = input_data.filters or self.filters
        top_k = input_data.top_k or self.top_k

        output = self.document_retriever.run(query_embedding, filters=filters, top_k=top_k, content_key=content_key)

        return {
            "documents": output["documents"],
        }
