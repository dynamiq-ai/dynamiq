from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import bulk

from dynamiq.connections import AWSOpenSearch
from dynamiq.nodes.dry_run import DryRunMixin
from dynamiq.storages.vector.base import BaseVectorStore, BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreException
from dynamiq.storages.vector.opensearch.filters import _normalize_filters
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.types import Document
from dynamiq.types.dry_run import DryRunConfig
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from opensearchpy import OpenSearch as OpenSearchClient


class OpenSearchSimilarityMetric(str, Enum):
    """Supported similarity metrics for OpenSearch."""

    COSINE = "cosinesimil"
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    INNER_PRODUCT = "innerproduct"
    HAMMING = "hamming"
    HAMMING_BIT = "hammingbit"


class OpenSearchMethod(str, Enum):
    """Supported method names for OpenSearch."""

    HNSW = "hnsw"
    IVF = "ivf"


class OpenSearchEngine(str, Enum):
    """Supported engine names for OpenSearch."""

    FAISS = "faiss"
    LUCENE = "lucene"


class OpenSearchVectorStoreParams(BaseVectorStoreParams):
    """Parameters for OpenSearch vector store.

    Attributes:
        index_name (str): Name of the index. Defaults to "default".
        content_key (str): Key for content field. Defaults to "content".
        dimension (int): Dimension of the vectors. Defaults to 1536.
        similarity (str): Similarity metric to use. Defaults to "cosine".
        embedding_key (str): Key for embedding field. Defaults to "embedding".
        batch_size (int): Batch size for writing operations. Defaults to 100.
    """

    similarity: OpenSearchSimilarityMetric = OpenSearchSimilarityMetric.COSINE
    embedding_key: str = "embedding"
    batch_size: int = 100


class OpenSearchVectorStoreWriterParams(OpenSearchVectorStoreParams, BaseWriterVectorStoreParams):
    """Parameters for OpenSearch vector store writer."""

    dimension: int = 1536


class OpenSearchVectorStore(BaseVectorStore, DryRunMixin):
    """Vector store using OpenSearch for dense vector search."""

    def __init__(
        self,
        connection: AWSOpenSearch | None = None,
        client: Optional["OpenSearchClient"] | None = None,
        index_name: str = "default",
        dimension: int = 1536,
        similarity: OpenSearchSimilarityMetric = OpenSearchSimilarityMetric.COSINE,
        create_if_not_exist: bool = False,
        content_key: str = "content",
        embedding_key: str = "embedding",
        batch_size: int = 100,
        index_settings: dict | None = None,
        mapping_settings: dict | None = None,
        dry_run_config: DryRunConfig | None = None,
    ):
        """
        Initialize OpenSearchVectorStore.

        Args:
            connection (Optional[AWSOpenSearch]): AWS OpenSearch connection.
            client (Optional[OpenSearchClient]): OpenSearch client. Defaults to None.
            index_name (str): Name of the index. Defaults to "default".
            dimension (int): Dimension of vectors. Defaults to 1536.
            similarity (OpenSearchSimilarityMetric): Similarity metric.
                        Defaults to OpenSearchSimilarityMetric.COSINE.
            create_if_not_exist (bool): Whether to create the index if it does not exist. Defaults to False.
            content_key (str): Key for content field. Defaults to "content".
            embedding_key (str): Key for embedding field. Defaults to "embedding".
            batch_size (int): Batch size for write operations. Defaults to 100.
            index_settings (Optional[dict]): Custom index settings. Defaults to None.
            mapping_settings (Optional[dict]): Custom mapping settings. Defaults to None.
            dry_run_config (Optional[DryRunConfig]): Configuration for dry run mode. Defaults to None.
        """
        super().__init__(dry_run_config=dry_run_config)

        if client is None:
            if connection is None:
                connection = AWSOpenSearch()
            self.client = connection.connect()
        else:
            self.client = client

        self.index_name = index_name
        self.dimension = dimension
        self.similarity = similarity
        self.content_key = content_key
        self.embedding_key = embedding_key
        self.batch_size = batch_size
        self.index_settings = index_settings or {}
        self.mapping_settings = mapping_settings or {}

        if not self.client.indices.exists(index=self.index_name):
            if create_if_not_exist:
                logger.info(f"Index {self.index_name} does not exist. Creating a new index.")
                self._create_index_if_not_exists()
                self._track_collection(self.index_name)
            else:
                raise ValueError(
                    f"Index {self.index_name} does not exist. Set 'create_if_not_exist' to True to create it."
                )
        else:
            logger.info(f"Collection {self.index_name} already exists. Skipping creation.")

        logger.debug(f"OpenSearchVectorStore initialized with index: {self.index_name}")

    def _create_index_if_not_exists(self) -> None:
        """Create the index if it doesn't exist."""

        base_settings = {"index": {"knn": True}}
        settings = base_settings.copy()

        # Add custom index settings if provided
        if self.index_settings:
            for key, value in self.index_settings.items():
                if key in settings and isinstance(settings[key], dict) and isinstance(value, dict):
                    settings[key].update(value)
                else:
                    settings[key] = value

        mapping = {
            "settings": settings,
            "mappings": {
                "properties": {
                    self.content_key: {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "metadata": {"type": "object"},
                    self.embedding_key: {
                        "type": "knn_vector",
                        "dimension": self.dimension,
                        "method": {
                            "engine": OpenSearchEngine.FAISS,
                            "name": OpenSearchMethod.HNSW,
                            "parameters": {"ef_construction": 128, "m": 24},
                            "space_type": self.similarity,
                        },
                    },
                }
            },
        }

        # Add custom mapping settings if provided
        if self.mapping_settings:
            merged_mappings = mapping["mappings"].copy()
            for key, value in self.mapping_settings.items():
                if (
                    key == "properties"
                    and key in merged_mappings
                    and isinstance(merged_mappings[key], dict)
                    and isinstance(value, dict)
                ):
                    merged_mappings[key].update(value)
                else:
                    merged_mappings[key] = value
            mapping["mappings"] = merged_mappings

        self.client.indices.create(index=self.index_name, body=mapping)

    def delete_collection(self, collection_name: str | None = None) -> None:
        """
        Delete the collection in the database.

        Args:
            collection_name (str | None): Name of the collection to delete.
        """
        try:
            collection_to_delete = collection_name or self.index_name
            self.client.indices.delete(index=collection_to_delete)
            logger.info(f"Deleted collection '{collection_to_delete}'.")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_to_delete}': {e}")
            raise

    def _handle_duplicate_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL
    ) -> list[Document]:
        """
        Handle duplicate documents based on policy.

        Args:
            documents (list[Document]): List of documents to check for duplicates.
            policy (DuplicatePolicy): Policy for handling duplicates. Defaults to DuplicatePolicy.FAIL.

        Returns:
            list[Document]: List of documents after applying the specified policy.

        Raises:
            VectorStoreException: If duplicates are found and the policy is set to FAIL.
        """
        if policy == DuplicatePolicy.OVERWRITE:
            return documents

        # Get unique documents
        unique_docs = {}
        for doc in documents:
            if doc.id in unique_docs:
                logger.warning(f"Duplicate document ID found: {doc.id}")
            unique_docs[doc.id] = doc

        if policy == DuplicatePolicy.NONE:
            return list(unique_docs.values())

        existing_ids = set()
        for doc_id in unique_docs.keys():
            try:
                self.retrieve_document_by_file_id(file_id=doc_id)
                existing_ids.add(doc_id)
            except NotFoundError:
                pass

        if policy == DuplicatePolicy.FAIL and existing_ids:
            raise VectorStoreException(f"Documents with IDs {existing_ids} already exist")

        filtered_docs = [doc for doc_id, doc in unique_docs.items() if doc_id not in existing_ids]
        return filtered_docs

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
        batch_size: int | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> int:
        """
        Write documents to OpenSearch.

        Args:
            documents (list[Document]): List of documents to write.
            policy (DuplicatePolicy): Policy for handling duplicate documents. Defaults to DuplicatePolicy.FAIL.
            batch_size (Optional[int]): Size of batches for bulk operations. Defaults to None.
            content_key (Optional[str]): The field used to store content. Defaults to None.
            embedding_key (Optional[str]): The field used to store embeddings. Defaults to None.

        Returns:
            int: Number of documents successfully written.

        Raises:
            ValueError: If the provided documents are invalid.
            VectorStoreException: If duplicates are found when using the FAIL policy.
        """
        if not documents:
            return 0

        if not isinstance(documents[0], Document):
            raise ValueError("Documents must be of type Document")

        # Handle duplicates
        documents = self._handle_duplicate_documents(documents, policy)
        if not documents:
            return 0

        batch_size = batch_size or self.batch_size
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        total_written = 0
        actions = []
        for doc in documents:
            action = {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": doc.id,
                "_source": {content_key: doc.content, "metadata": doc.metadata},
            }
            # Only include embedding field if present and not None
            if getattr(doc, "embedding", None) is not None:
                action["_source"][embedding_key] = doc.embedding

            actions.append(action)

        for i in range(0, len(actions), batch_size):
            chunk = actions[i : i + batch_size]

            success_count, _ = bulk(self.client, chunk, refresh=True)
            total_written += success_count

            self._track_documents([action["_id"] for action in chunk])

        return total_written

    def _scale_score(self, score: float, similarity: OpenSearchSimilarityMetric) -> float:
        """
        Scale the score based on the similarity metric.

        Args:
            score (float): Raw score from OpenSearch.
            similarity (OpenSearchSimilarityMetric): Similarity metric used.

        Returns:
            float: Scaled score between 0 and 1, depending on the similarity metric used.
        """
        if similarity == OpenSearchSimilarityMetric.COSINE:
            # Normalize range [0, 2] to [0, 1]
            return score / 2
        elif similarity == OpenSearchSimilarityMetric.INNER_PRODUCT:
            # Normalize using sigmoid function as inner product scores can be any range
            return float(1 / (1 + np.exp(-score / 100)))
        else:  # L1, L2, LINF, HAMMING, HAMMING_BIT
            # L1, L2, LINF, HAMMING, HAMMING_BIT distance is inverse - smaller is better
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
        embedding_key: str | None = None,
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
            embedding_key (Optional[str]): The field used to store embeddings in the storage.

        Returns:
            List[Document]: Retrieved documents.

        Raises:
            ValueError: If query_embedding is invalid.
        """
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        embedding_key = embedding_key or self.embedding_key
        content_key = content_key or self.content_key

        knn_query = {embedding_key: {"vector": query_embedding, "k": top_k}}

        if filters:
            normalized_filters = _normalize_filters(filters)
            knn_query[embedding_key]["filter"] = {"bool": normalized_filters}

        body = {
            "size": top_k,
            "query": {"knn": knn_query},
            "_source": {"excludes": [embedding_key] if exclude_document_embeddings else []},
        }

        response = self.client.search(index=self.index_name, body=body)

        documents = []
        for hit in response["hits"]["hits"]:
            score = hit.get("_score", None)
            if score is None:
                continue

            if scale_scores:
                score = self._scale_score(score, self.similarity)

            source = hit.get("_source", {})
            if content_key not in source:
                continue

            doc = Document(id=hit["_id"], content=source[content_key], metadata=source.get("metadata", {}), score=score)
            if not exclude_document_embeddings and embedding_key in source:
                doc.embedding = source[embedding_key]
            documents.append(doc)

        return documents

    def retrieve_document_by_file_id(
        self,
        file_id: str,
        include_embeddings: bool = False,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ):
        embedding_key = embedding_key or self.embedding_key
        content_key = content_key or self.content_key

        response = self.client.get(
            index=self.index_name,
            id=file_id,
            _source_excludes=([embedding_key] if not include_embeddings else None),
        )

        # Convert result to Document
        doc = Document(
            id=response["_id"],
            content=response["_source"][content_key],
            metadata=response["_source"]["metadata"],
        )
        if include_embeddings:
            doc.embedding = response["_source"][embedding_key]

        return doc

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the store.

        Args:
            document_ids (Optional[List[str]]): IDs to delete. Defaults to None.
            delete_all (bool): Delete all documents. Defaults to False.
        """
        if delete_all:
            self.client.delete_by_query(index=self.index_name, body={"query": {"match_all": {}}}, refresh=True)

        elif document_ids:
            operations = [{"_op_type": "delete", "_index": self.index_name, "_id": doc_id} for doc_id in document_ids]
            bulk(self.client, operations, refresh=True)

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

        body = {"query": bool_query}

        response = self.client.delete_by_query(index=self.index_name, body=body, refresh=True)

        deleted_count = response.get("deleted", 0)
        logger.info(f"Deleted {deleted_count} documents matching filters.")

    def list_documents(
        self,
        top_k: int | None = 100,
        include_embeddings: bool = False,
        content_key: str | None = None,
        embedding_key: str | None = None,
        scale_scores: bool = False,
    ) -> list[Document]:
        """
        List documents in the OpenSearch vector store.

        Args:
            top_k (Optional[int]): Maximal number of documents to retrieve. Defaults to 100.
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.
            content_key (Optional[str]): The field used to store content in the storage.
            embedding_key (Optional[str]): The field used to store embeddings in the storage.
            scale_scores (bool): Whether to scale scores to 0-1 range. Defaults to False.

        Returns:
            list[Document]: List of Document objects retrieved.
        """
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        # Build search query
        search_body = {
            "size": top_k,
            "query": {"match_all": {}},
        }

        if not include_embeddings:
            search_body["_source"] = {"excludes": [embedding_key]}

        response = self.client.search(index=self.index_name, body=search_body)

        # Convert hits to Document objects
        all_documents = []
        for hit in response["hits"]["hits"]:
            score = hit.get("_score", None)
            if scale_scores and score is not None:
                score = self._scale_score(score, self.similarity)

            if content_key not in hit["_source"]:
                continue

            doc = Document(
                id=hit["_id"],
                content=hit["_source"][content_key],
                metadata=hit["_source"].get("metadata", {}),
                score=score,
            )
            if include_embeddings and embedding_key in hit["_source"]:
                doc.embedding = hit["_source"][embedding_key]

            all_documents.append(doc)

        return all_documents

    def count_documents(self) -> int:
        """
        Count the number of documents in the OpenSearch index.

        Returns:
            int: The number of documents in the store.
        """
        response = self.client.count(index=self.index_name, body={"query": {"match_all": {}}})
        return response.get("count", 0)

    def get_field_statistics(self, field: str) -> dict[str, Any]:
        """
        Get statistics for a numeric field.

        Args:
            field (str): Full field name (must be numeric)

        Returns:
            Dictionary with min, max, avg, sum
        """
        response = self.client.search(
            index=self.index_name,
            body={"size": 0, "aggs": {"stats": {"stats": {"field": field}}}},
        )
        return response["aggregations"]["stats"]

    def update_document_by_file_id(
        self,
        file_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> None:
        """Update an existing document.

        Args:
            file_id (str): Document ID
            content (Optional[str]): Update content
            metadata (Optional[dict[str, Any]]): Update field metadata or add new fields
            embedding (Optional[list[float]]): New embedding vector
            content_key (Optional[str]): Key for content field.
            embedding_key (Optional[str]): The field used to store embeddings in the storage.
        """
        update_fields = {}
        if content is not None:
            update_fields[content_key or self.content_key] = content
        if metadata is not None:
            update_fields["metadata"] = metadata
        if embedding is not None:
            update_fields[embedding_key or self.embedding_key] = embedding

        if update_fields:
            self.client.update(index=self.index_name, id=file_id, body={"doc": update_fields}, refresh=True)

    def update_documents_batch(
        self,
        documents: list[Document],
        batch_size: int | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> int:
        """
        Update multiple documents in batches.

        Args:
            documents (list[Document]): List of documents to update.
            batch_size (Optional[int]): Size of batches for bulk operations.
            content_key (Optional[str]): Key for content field.
            embedding_key (Optional[str]): The field used to store embeddings in the storage.

        Returns:
            int: Number of documents successfully updated.

        """
        batch_size = batch_size or self.batch_size
        total_updated = 0
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        def generate_actions(docs):
            actions = []
            for doc in docs:
                update_doc = {
                    content_key: doc.content,
                    "metadata": doc.metadata,
                }
                if getattr(doc, "embedding", None) is not None:
                    update_doc[embedding_key] = doc.embedding

                action = {
                    "_op_type": "update",
                    "_index": self.index_name,
                    "_id": doc.id,
                    "doc": update_doc,
                }
                actions.append(action)
            return actions

        for i in range(0, len(documents), batch_size):
            sub_set_docs = documents[i : i + batch_size]
            batch_actions = generate_actions(sub_set_docs)
            success_count, _ = bulk(self.client, batch_actions, refresh=True)
            total_updated += success_count
        return total_updated

    def create_alias(
        self,
        alias_name: str,
        index_names: list[str] | None = None,
    ) -> None:
        """
        Create an alias for one or more indices.

        Args:
            alias_name (str): Name of the alias.
            index_names (Optional[list[str]]): List of indices to include in the alias. Defaults to None.
        """
        index_names = index_names or [self.index_name]
        actions = []
        for index in index_names:
            actions.append({"add": {"index": index, "alias": alias_name}})
        self.client.indices.update_aliases({"actions": actions})

    def close(self) -> None:
        """Close the client connection."""
        if hasattr(self, "client"):
            self.client.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
