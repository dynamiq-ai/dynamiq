from typing import Any

from dynamiq.storages.vector.qdrant import QdrantVectorStore
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class QdrantDocumentRetriever:
    """
    Document Retriever using Qdrant
    """

    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a Qdrant vector store with optional filtering.
        Args:
            vector_store (QdrantVectorStore): An instance of QdrantVectorStore to interface with Qdrant vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.
        Raises:
            ValueError: If the `vector_store` is not an instance of `QdrantVectorStore`.
        This initializer checks if the `vector_store` provided is an instance of the expected `QdrantVectorStore`
        class, sets up filtering conditions if any, and defines how many top results to retrieve in document queries.
        """
        if not isinstance(vector_store, QdrantVectorStore):
            msg = "document_store must be an instance of QdrantVectorStore"
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
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from the QdrantDocumentStore that are similar to the provided query embedding.
        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (Optional[int], optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]], optional): Filters to apply
                for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query_embedding.
        """
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store._query_by_embedding(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            return_embedding=not exclude_document_embeddings,
            content_key=content_key,
        )
        logger.debug(f"Retrieved {len(docs)} documents from Qdrant Vector Store.")

        return {"documents": docs}
