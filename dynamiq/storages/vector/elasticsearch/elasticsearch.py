from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from elasticsearch import NotFoundError

from dynamiq.connections import Elasticsearch
from dynamiq.storages.vector.base import BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.elasticsearch.filters import _normalize_filters
from dynamiq.storages.vector.exceptions import VectorStoreException
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.utils import create_file_id_filter
from dynamiq.types import Document
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch as ElasticsearchClient


class ElasticsearchSimilarityMetric(str, Enum):
    """Supported similarity metrics for Elasticsearch."""

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    L2 = "l2_norm"
    MAX_INNER_PRODUCT = "max_inner_product"


class ElasticsearchVectorStoreParams(BaseVectorStoreParams):
    """Parameters for Elasticsearch vector store.

    Attributes:
        index_name (str): Name of the index. Defaults to "default".
        dimension (int): Dimension of the vectors. Defaults to 1536.
        similarity (str): Similarity metric to use. Defaults to "cosine".
        content_key (str): Key for content field. Defaults to "content".
        embedding_key (str): Key for embedding field. Defaults to "embedding".
    """

    dimension: int = 1536
    similarity: ElasticsearchSimilarityMetric = ElasticsearchSimilarityMetric.COSINE
    embedding_key: str = "embedding"
    write_batch_size: int = 100
    scroll_size: int = 1000


class ElasticsearchVectorStoreWriterParams(ElasticsearchVectorStoreParams, BaseWriterVectorStoreParams):
    """Parameters for Elasticsearch vector store writer."""

    pass


class ElasticsearchVectorStore:
    """Vector store using Elasticsearch for dense vector search."""

    def __init__(
        self,
        connection: Elasticsearch | None = None,
        client: Optional["ElasticsearchClient"] | None = None,
        index_name: str = "default",
        dimension: int = 1536,
        similarity: ElasticsearchSimilarityMetric = ElasticsearchSimilarityMetric.COSINE,
        create_if_not_exist: bool = False,
        content_key: str = "content",
        embedding_key: str = "embedding",
        write_batch_size: int = 100,
        scroll_size: int = 1000,
        index_settings: dict | None = None,
        mapping_settings: dict | None = None,
    ):
        """Initialize ElasticsearchVectorStore.

        Args:
            connection: Elasticsearch connection
            client: Elasticsearch client
            index_name: Name of the index
            dimension: Dimension of vectors
            similarity: Similarity metric
            create_if_not_exist: Create index if not exists
            content_key: Key for content field
            embedding_key: Key for embedding field
            write_batch_size: Batch size for write operations
            scroll_size: Batch size for scroll operations
            index_settings: Custom index settings
            mapping_settings: Custom mapping settings
        """
        if client is None:
            if connection is None:
                connection = Elasticsearch()
            self.client = connection.connect()
        else:
            self.client = client

        self.index_name = index_name
        self.dimension = dimension
        self.similarity = similarity
        self.content_key = content_key
        self.embedding_key = embedding_key
        self.write_batch_size = write_batch_size
        self.scroll_size = scroll_size
        self.index_settings = index_settings or {}
        self.mapping_settings = mapping_settings or {}

        if create_if_not_exist:
            self._create_index_if_not_exists()

        logger.debug(f"ElasticsearchVectorStore initialized with index: {self.index_name}")

    def _create_index_if_not_exists(self) -> None:
        """Create the index if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            # Base mapping
            mapping = {
                "mappings": {
                    "properties": {
                        self.content_key: {"type": "text"},
                        "metadata": {"type": "object"},
                        self.embedding_key: {
                            "type": "dense_vector",
                            "dims": self.dimension,
                            "index": True,
                            "similarity": self.similarity,
                        },
                    }
                }
            }

            # Add custom mapping settings if provided
            if self.mapping_settings:
                mapping["mappings"].update(self.mapping_settings)

            # Add index settings if provided
            if self.index_settings:
                mapping["settings"] = self.index_settings

            self.client.indices.create(index=self.index_name, body=mapping)

    def _handle_duplicate_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
    ) -> list[Document]:
        """Handle duplicate documents based on policy.

        Args:
            documents: List of documents to check
            policy: Policy for handling duplicates

        Returns:
            List of documents after applying policy

        Raises:
            VectorStoreException: If duplicates found with FAIL policy
        """
        if policy == DuplicatePolicy.OVERWRITE:
            return documents

        # Get unique documents
        unique_docs = {}
        for doc in documents:
            if doc.id in unique_docs:
                logger.warning(f"Duplicate document ID found: {doc.id}")
            unique_docs[doc.id] = doc

        if policy == DuplicatePolicy.SKIP:
            # Check which documents already exist
            existing_ids = set()
            for doc_id in unique_docs.keys():
                try:
                    self.client.get(index=self.index_name, id=doc_id)
                    existing_ids.add(doc_id)
                except NotFoundError:
                    pass

            # Remove existing documents
            return [doc for doc in documents if doc.id not in existing_ids]

        elif policy == DuplicatePolicy.FAIL:
            # Check for existing documents
            existing_ids = set()
            for doc_id in unique_docs.keys():
                try:
                    self.client.get(index=self.index_name, id=doc_id)
                    existing_ids.add(doc_id)
                except NotFoundError:
                    pass

            if existing_ids:
                raise VectorStoreException(f"Documents with IDs {existing_ids} already exist")

        return list(unique_docs.values())

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
        batch_size: int | None = None,
        content_key: str | None = None,
    ) -> int:
        """Write documents to Elasticsearch.

        Args:
            documents: Documents to write
            policy: Policy for handling duplicates
            batch_size: Size of batches for bulk operations
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            Number of documents written

        Raises:
            ValueError: If documents are invalid
            VectorStoreException: If duplicates found with FAIL policy
        """
        if not documents:
            return 0

        if not isinstance(documents[0], Document):
            raise ValueError("Documents must be of type Document")

        # Handle duplicates
        documents = self._handle_duplicate_documents(documents, policy)
        if not documents:
            return 0

        # Process in batches
        batch_size = batch_size or self.write_batch_size
        content_key = content_key or self.content_key
        total_written = 0

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            operations = []
            for doc in batch:
                operations.extend(
                    [
                        {"index": {"_index": self.index_name, "_id": doc.id}},
                        {
                            content_key: doc.content,
                            "metadata": doc.metadata,
                            self.embedding_key: doc.embedding,
                        },
                    ]
                )

            if operations:
                self.client.bulk(operations=operations, refresh=True)
                total_written += len(batch)

        return total_written

    def _scale_score(self, score: float, similarity: ElasticsearchSimilarityMetric) -> float:
        """Scale the score based on similarity metric.

        Args:
            score: Raw score from Elasticsearch
            similarity: Similarity metric used

        Returns:
            Scaled score between 0 and 1
        """
        if similarity == ElasticsearchSimilarityMetric.COSINE:
            # Elasticsearch cosine scores are between -1 and 1
            return (score + 1) / 2
        elif similarity == ElasticsearchSimilarityMetric.DOT_PRODUCT:
            # Normalize dot product using sigmoid
            return float(1 / (1 + np.exp(-score / 100)))
        else:  # L2
            # L2 distance is inverse - smaller is better
            # Convert to similarity score
            return 1 / (1 + score)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        exclude_document_embeddings: bool = True,
        filters: dict[str, Any] | None = None,
        scale_scores: bool = False,
        content_key: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents by vector similarity.

        Args:
            query_embedding (List[float]): Query vector.
            top_k (int): Number of results. Defaults to 10.
            exclude_document_embeddings (bool): Exclude embeddings in response. Defaults to True.
            filters (dict[str, Any] | None): Metadata filters. Defaults to None.
            scale_scores (bool): Whether to scale scores to 0-1 range. Defaults to False.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List[Document]: Retrieved documents.

        Raises:
            ValueError: If query_embedding is invalid.
        """
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        if len(query_embedding) != self.dimension:
            raise ValueError(f"query_embedding must have dimension {self.dimension}")

        # Build the query
        query = {
            "knn": {
                "field": self.embedding_key,
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 2,
            }
        }

        if filters:
            # Normalise filters to Elasticsearch format
            filters = _normalize_filters(filters)
            query["knn"]["filter"] = {"bool": filters}

        # Execute search
        response = self.client.search(
            index=self.index_name,
            query=query,
            size=top_k,
            _source_excludes=([self.embedding_key] if exclude_document_embeddings else None),
        )
        content_key = content_key or self.content_key

        # Convert results to Documents with optional score scaling
        documents = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            if scale_scores:
                score = self._scale_score(score, self.similarity)

            if content_key not in hit["_source"]:
                continue

            doc = Document(
                id=hit["_id"],
                content=hit["_source"][content_key],
                metadata=hit["_source"]["metadata"],
                score=score,
            )
            if not exclude_document_embeddings:
                doc.embedding = hit["_source"][self.embedding_key]
            documents.append(doc)

        return documents

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """Delete documents from the store.

        Args:
            document_ids (List[str] | None): IDs to delete. Defaults to None.
            delete_all (bool): Delete all documents. Defaults to False.
        """
        if delete_all:
            self.client.delete_by_query(index=self.index_name, query={"match_all": {}}, refresh=True)
        elif document_ids:
            operations = []
            for doc_id in document_ids:
                operations.append({"delete": {"_index": self.index_name, "_id": doc_id}})
            if operations:
                self.client.bulk(operations=operations, refresh=True)
        else:
            logger.warning("No document IDs provided. No documents will be deleted.")

    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """Delete documents matching filters.

        Args:
            filters (dict[str, Any]): Metadata filters.
        """
        if not filters:
            logger.warning("No filters provided. No documents will be deleted.")
            return

        filters = _normalize_filters(filters)
        bool_query = {"bool": filters}

        self.client.delete_by_query(index=self.index_name, query=bool_query, refresh=True)

    def delete_documents_by_file_id(self, file_id: str):
        """
        Delete documents from the Pinecone vector store by file ID.
            file_id should be located in the metadata of the document.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def list_documents(
        self, include_embeddings: bool = False, content_key: str | None = None, scale_scores: bool = False
    ) -> list[Document]:
        """
        List documents in the Pinecone vector store.

        Args:
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.
            content_key (Optional[str]): The field used to store content in the storage.
            scale_scores (bool): Whether to scale scores to 0-1 range. Defaults to False.

        Returns:
            list[Document]: List of Document objects retrieved.
        """

        response = self.client.search(
            index=self.index_name,
            query={"match_all": {}},
            _source_excludes=([self.embedding_key] if not include_embeddings else None),
        )
        content_key = content_key or self.content_key

        # Convert results to Documents with optional score scaling
        all_documents = []
        for hit in response["hits"]["hits"]:
            score = hit["_score"]
            if scale_scores:
                score = self._scale_score(score, self.similarity)

            if content_key not in hit["_source"]:
                continue

            doc = Document(
                id=hit["_id"],
                content=hit["_source"][content_key],
                metadata=hit["_source"]["metadata"],
                score=score,
            )
            if include_embeddings:
                doc.embedding = hit["_source"][self.embedding_key]
            all_documents.append(doc)

        return all_documents

    def count_documents(self) -> int:
        """
        Count the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """
        response = self.client.count(
            index=self.index_name,
            query={"match_all": {}},
            _source_excludes=([self.embedding_key]),
        )
        return response.get("count", 0)

    def close(self) -> None:
        """Close the client connection."""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
