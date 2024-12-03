from typing import Any

from dynamiq.storages.vector.weaviate import WeaviateVectorStore
from dynamiq.utils.logger import logger


class WeaviateDocumentRetriever:
    """
    Document Retriever using Weaviate
    """

    def __init__(
        self,
        *,
        vector_store: WeaviateVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a Weaviate vector store with optional filtering.
        Args:
            vector_store (WeaviateVectorStore): An instance of WeaviateVectorStore to interface with Weaviate vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.
        Raises:
            ValueError: If the `vector_store` is not an instance of `WeaviateVectorStore`.
        This initializer checks if the `vector_store` provided is an instance of the expected `WeaviateVectorStore`
        class, sets up filtering conditions if any, and defines how many top results to retrieve in document queries.
        """
        if not isinstance(vector_store, WeaviateVectorStore):
            msg = "document_store must be an instance of WeaviateVectorStore"
            raise ValueError(msg)

        self.vector_store = vector_store
        self.filters = filters or {}
        self.top_k = top_k

    def run(
        self,
        query_embedding: list[float],
        exclude_document_embeddings: bool = True,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
    ):
        """
        Retrieves documents from the WeaviateDocumentStore that are similar to the provided query embedding.
        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query_embedding.
        """
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            exclude_document_embeddings=exclude_document_embeddings,
            content_key=content_key,
        )
        logger.debug(f"Retrieved {len(docs)} documents from Weaviate Vector Store.")

        return {"documents": docs}

    def close(self):
        """
        Closes the WeaviateDocumentRetriever component.
        """
        self.vector_store.close()
