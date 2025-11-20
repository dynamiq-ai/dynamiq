from typing import Any

from dynamiq.components.retrievers.utils import filter_documents_by_threshold
from dynamiq.storages.vector import OpenSearchVectorStore
from dynamiq.storages.vector.opensearch.opensearch import OpenSearchSimilarityMetric
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class OpenSearchDocumentRetriever:
    """Document Retriever using OpenSearch."""

    def __init__(
        self,
        *,
        vector_store: OpenSearchVectorStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        similarity_threshold: float | None = None,
    ):
        """
        Initialize OpenSearchDocumentRetriever.

        Args:
            vector_store (OpenSearchVectorStore): An instance of OpenSearchVectorStore.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            top_k (int): Maximum number of documents to return. Defaults to 10.

        Raises:
            ValueError: If `vector_store` is not an instance of OpenSearchVectorStore.
        """
        if not isinstance(vector_store, OpenSearchVectorStore):
            raise ValueError("vector_store must be an instance of OpenSearchVectorStore")

        self.vector_store = vector_store
        self.filters = filters or {}
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def _higher_is_better(self) -> bool:
        return self.vector_store.similarity in {
            OpenSearchSimilarityMetric.COSINE,
            OpenSearchSimilarityMetric.INNER_PRODUCT,
        }

    def run(
        self,
        query_embedding: list[float],
        exclude_document_embeddings: bool = True,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
        scale_scores: bool = False,
        similarity_threshold: float | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from OpenSearchVectorStore.

        Args:
            query_embedding (list[float]): Vector query for similarity search.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.
            top_k (Optional[int]): Maximum number of documents to return. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters to apply for retrieving specific documents. Defaults to None.
            content_key (Optional[str]): Field used to store content. Defaults to None.
            embedding_key (Optional[str]): Field used to store vector. Defaults to None.
            scale_scores (bool): Whether to scale scores to the 0-1 range. Defaults to False.

        Returns:
            dict[str, list[Document]]: A dictionary containing a list of retrieved documents.

        Raises:
            ValueError: If the query format is invalid.
        """
        top_k = top_k or self.top_k
        filters = filters or self.filters

        docs = self.vector_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            exclude_document_embeddings=exclude_document_embeddings,
            scale_scores=scale_scores,
            content_key=content_key,
            embedding_key=embedding_key,
        )

        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        docs = filter_documents_by_threshold(docs, threshold, higher_is_better=self._higher_is_better())

        logger.debug(f"Retrieved {len(docs)} documents from OpenSearch Vector Store")
        return {"documents": docs}
