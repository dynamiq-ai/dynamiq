from typing import Any

from dynamiq.components.retrievers.utils import filter_documents_by_threshold
from dynamiq.storages.vector.turbopuffer import TurbopufferVectorStore
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class TurbopufferDocumentRetriever:
    """
    Document Retriever using Turbopuffer
    """

    def __init__(
        self,
        *,
        vector_store: TurbopufferVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        similarity_threshold: float | None = None,
        max_vector_distance: float | None = None,
    ):
        """
        Initializes a component for retrieving documents from a Turbopuffer vector store.

        Args:
            vector_store (TurbopufferVectorStore): The vector store to retrieve documents from.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): The maximum number of documents to return. Defaults to 10.
            similarity_threshold (float | None): The minimum score a document needs. Defaults to None.
            max_vector_distance (float | None): The maximum cosine distance of a vector match. Defaults to None.

        Raises:
            ValueError: If the `vector_store` is not an instance of `TurbopufferVectorStore`.
        """
        if not isinstance(vector_store, TurbopufferVectorStore):
            raise ValueError("document_store must be an instance of TurbopufferVectorStore")

        self.vector_store = vector_store
        self.filters = filters or {}
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_vector_distance = max_vector_distance

    def run(
        self,
        query_embedding: list[float],
        exclude_document_embeddings: bool = True,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
        query: str | None = None,
        alpha: float | None = None,
        similarity_threshold: float | None = None,
        max_vector_distance: float | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieves documents from Turbopuffer that are similar to the query.

        With a query string, retrieval is hybrid and documents are scored by fused relative scores in
        [0, 1]. Without one, documents are scored by cosine similarity mapped to [0, 1]. A similarity
        threshold above 1 is read as a maximum cosine distance, as with Weaviate.

        Args:
            query_embedding (list[float]): The embedding vector of the query.
            exclude_document_embeddings (bool, optional): Whether to leave the embeddings of the
                retrieved documents out of the output.
            top_k (int, optional): The maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents.
            content_key (Optional[str]): The attribute used to store content.
            query (Optional[str]): The query string for keyword search. Defaults to None.
            alpha (Optional[float]): The weight of vector similarity in hybrid retrieval, from 0.0
                (keyword only) to 1.0 (vector only). Defaults to the store's alpha.
            similarity_threshold (float | None): The minimum score a document needs.
            max_vector_distance (float | None): The maximum cosine distance of a vector match.

        Returns:
            dict[str, list[Document]]: The retrieved documents, most relevant first.
        """
        top_k = top_k or self.top_k
        filters = filters or self.filters

        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        vector_distance = max_vector_distance if max_vector_distance is not None else self.max_vector_distance

        if query:
            docs = self.vector_store._hybrid_retrieval(
                query_embedding=query_embedding,
                query=query,
                filters=filters,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                alpha=alpha,
                content_key=content_key,
                max_vector_distance=vector_distance,
            )
            docs = filter_documents_by_threshold(docs, threshold, higher_is_better=True)
        else:
            min_score = threshold if threshold is not None and threshold <= 1 else None
            max_distance = threshold if threshold is not None and threshold > 1 else vector_distance
            docs = self.vector_store._embedding_retrieval(
                query_embedding=query_embedding,
                filters=filters,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                content_key=content_key,
                min_score=min_score,
                max_distance=max_distance,
            )

        logger.debug(f"Retrieved {len(docs)} documents from Turbopuffer Vector Store.")

        return {"documents": docs}

    def close(self):
        """
        Closes the TurbopufferDocumentRetriever component.
        """
        self.vector_store.close()
