from typing import Any

from dynamiq.storages.vector import PGVectorStore
from dynamiq.utils.logger import logger


class PGVectorDocumentRetriever:
    """
    Document Retriever using PGVector.
    """

    def __init__(
        self,
        *,
        vector_store: PGVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ):
        """
        Initializes a component for retrieving documents from a PGVector vector store with optional filtering.

        Args:
            vector_store (PGVectorStore): An instance of PGVectorStore to interface with PGVector vectors.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.

        Raises:
            ValueError: If the `vector_store` is not an instance of `PGVectorStore`.

        This initializer checks if the `vector_store` provided is an instance of the expected `PGVectorStore`
        class, sets up filtering conditions if any, and defines how many top results to retrieve in document queries.
        """
        if not isinstance(vector_store, PGVectorStore):
            msg = "document_store must be an instance of PGVectorStore"
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
        embedding_key: str | None = None,
        query: str | None = None,
        alpha: float | None = 0.5,
    ):
        """
        Retrieves documents from the PGVectorStore that are similar to the provided query embedding.

        Args:
            query_embedding (List[float]): The embedding vector of the query for which similar documents are to be
            retrieved.
            exclude_document_embeddings (bool, optional): Specifies whether to exclude the embeddings of the retrieved
            documents from the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): The field used to store content in the storage. Defaults to None.
            embedding_key (Optional[str]): The field used to store vector in the storage. Defaults to None.
            query(Optional[str]): The query string to search for (when using keyword search). Defaults to None.
            alpha (Optional[float]): The alpha value for hybrid retrieval. Defaults to 0.5.

            When using hybrid retrieval, the alpha value determines the weight of the keyword search score in the
            final ranking. A value of 0.0 means only keyword search score will be used, and a value of 1.0 means only
            vector similarity score will be considered.

        Returns:
            List[Document]: A list of Document instances sorted by their relevance to the query_embedding.
        """

        top_k = top_k or self.top_k
        filters = filters or self.filters

        if query:
            docs = self.vector_store._hybrid_retrieval(
                query_embedding=query_embedding,
                query=query,
                filters=filters,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                content_key=content_key,
                embedding_key=embedding_key,
                alpha=alpha,
            )
        else:
            docs = self.vector_store._embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                content_key=content_key,
                embedding_key=embedding_key,
            )

        logger.debug(f"Retrieved {len(docs)} documents from pgvector Vector Store.")

        return {"documents": docs}
