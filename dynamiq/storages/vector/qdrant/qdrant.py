import enum
import json
import logging
import re
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generator, Optional, Pattern, TypeVar, Union

import numpy as np
import qdrant_client
from qdrant_client import grpc
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.nodes.dry_run import DryRunMixin
from dynamiq.storages.vector.base import BaseVectorStore, BaseWriterVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreDuplicateDocumentException as DuplicateDocumentError
from dynamiq.storages.vector.exceptions import VectorStoreException as DocumentStoreError
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.qdrant.converters import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    convert_dynamiq_documents_to_qdrant_points,
    convert_id,
    convert_qdrant_point_to_dynamiq_document,
)
from dynamiq.storages.vector.qdrant.filters import convert_filters_to_qdrant
from dynamiq.types import Document
from dynamiq.types.dry_run import DryRunConfig

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

T = TypeVar("T")


class QdrantStoreError(DocumentStoreError):
    pass


class SparseEmbedding:
    """Class representing a sparse embedding."""

    indices: list[int]
    values: list[float]


FilterType = dict[str, Union[dict[str, Any], list[Any], str, int, float, bool]]


def get_batches_from_generator(iterable, n):
    """Batch elements of an iterable into fixed-length chunks or blocks.

    Args:
        iterable: The iterable to batch.
        n: The size of each batch.

    Yields:
        Batches of the iterable.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))


class QdrantSimilarityMetric(str, enum.Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    L2 = "l2"


class QdrantWriterVectorStoreParams(BaseWriterVectorStoreParams):
    dimension: int = 1536
    metric: QdrantSimilarityMetric = QdrantSimilarityMetric.COSINE


class QdrantVectorStore(BaseVectorStore, DryRunMixin):
    """QdrantVectorStore a Document Store for Qdrant.

    Usage example:

    ```python
    from dynamiq.types import Document
    from dynamiq.storages.vector.qdrant import QdrantVectorStore

    document_store = QdrantVectorStore(
            url="https://xxxxxx-xxxxx-xxxxx-xxxx-xxxxxxxxx.us-east.aws.cloud.qdrant.io:6333",
        api_key="<your-api-key>",
    )

    document_store.count_documents()
    ```

    Attributes:
        DISTANCE_BY_SIMILARITY (ClassVar[dict[QdrantSimilarityMetric, str]]): Mapping of metrics to distances.
    """

    DISTANCE_BY_SIMILARITY: ClassVar[dict[QdrantSimilarityMetric, str]] = {
        QdrantSimilarityMetric.COSINE: rest.Distance.COSINE,
        QdrantSimilarityMetric.DOT_PRODUCT: rest.Distance.DOT,
        QdrantSimilarityMetric.L2: rest.Distance.EUCLID,
    }
    _MISSING_INDEX_PATTERN: ClassVar[Pattern[str]] = re.compile(
        r'Index required but not found for "(?P<field>[^"]+)" of one of the following types: \[(?P<types>[^\]]+)\]',
        re.IGNORECASE,
    )
    _PAYLOAD_SCHEMA_BY_NAME: ClassVar[dict[str, rest.PayloadSchemaType]] = {
        schema.value.lower(): schema for schema in rest.PayloadSchemaType
    }

    def __init__(
        self,
        connection: QdrantConnection | None = None,
        client: Optional["QdrantClient"] = None,
        location: str | None = None,
        url: str | None = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        prefix: str | None = None,
        timeout: int | None = None,
        host: str | None = None,
        path: str | None = None,
        force_disable_check_same_thread: bool = False,
        index_name: str = "Document",
        dimension: int = 1536,
        on_disk: bool = False,
        use_sparse_embeddings: bool = False,
        sparse_idf: bool = False,
        metric: QdrantSimilarityMetric = QdrantSimilarityMetric.COSINE,
        return_embedding: bool = False,
        create_if_not_exist: bool = False,
        recreate_index: bool = False,
        shard_number: int | None = None,
        replication_factor: int | None = None,
        write_consistency_factor: int | None = None,
        on_disk_payload: bool | None = None,
        hnsw_config: dict | None = None,
        optimizers_config: dict | None = None,
        wal_config: dict | None = None,
        quantization_config: dict | None = None,
        init_from: dict | None = None,
        wait_result_from_api: bool = True,
        metadata: dict | None = None,
        write_batch_size: int = 100,
        scroll_size: int = 10_000,
        payload_fields_to_index: list[dict] | None = None,
        content_key: str = "content",
        dry_run_config: DryRunConfig | None = None,
    ):
        """Initializes the QdrantDocumentStore.

        Args:
            location: If `memory` - use in-memory Qdrant instance. If `str` - use it as a URL parameter. If `None` - use
                default values for host and port.
            url: Either host or str of `Optional[scheme], host, Optional[port], Optional[prefix]`.
            port: Port of the REST API interface.
            grpc_port: Port of the gRPC interface.
            prefer_grpc: If `True` - use gRPC interface whenever possible in custom methods.
            https: If `True` - use HTTPS(SSL) protocol.
            api_key: API key for authentication in Qdrant Cloud.
            prefix: If not `None` - add prefix to the REST URL path. Example: service/v1 will result in
                http://localhost:6333/service/v1/{qdrant-endpoint} for REST API.
            timeout: Timeout for REST and gRPC API requests.
            host: Host name of Qdrant service. If `url` and `host` are `None`, set to `localhost`.
            path: Persistence path for QdrantLocal.
            force_disable_check_same_thread: For QdrantLocal, force disable check_same_thread. Only use this if you can
                guarantee that you can resolve the thread safety outside QdrantClient.
            index_name: Name of the index.
            dimension: Dimension of the embeddings.
            on_disk: Whether to store the collection on disk.
            use_sparse_embedding: If set to `True`, enables support for sparse embeddings.
            sparse_idf: If set to `True`, computes the Inverse Document Frequency (IDF) when using sparse embeddings. It
                is required to use techniques like BM42. It is ignored if `use_sparse_embeddings` is `False`.
            metric: The similarity metric to use.
            return_embedding: Whether to return embeddings in the search results.
            recreate_index: Whether to recreate the index.
            shard_number: Number of shards in the collection.
            replication_factor: Replication factor for the collection. Defines how many copies of each shard will be
                created. Effective only in distributed mode.
            write_consistency_factor: Write consistency factor for the collection. Minimum value is 1. Defines how many
                replicas should apply to the operation for it to be considered successful. Increasing this number makes
                the collection more resilient to inconsistencies but will cause failures if not enough replicas are
                available. Effective only in distributed mode.
            on_disk_payload: If `True`, the point's payload will not be stored in memory and will be read from the disk
                every time it is requested. This setting saves RAM by slightly increasing response time. Note: indexed
                payload values remain in RAM.
            hnsw_config: Params for HNSW index.
            optimizers_config: Params for optimizer.
            wal_config: Params for Write-Ahead-Log.
            quantization_config: Params for quantization. If `None`, quantization will be disabled.
            init_from: Use data stored in another collection to initialize this collection.
            wait_result_from_api: Whether to wait for the result from the API after each request.
            metadata: Additional metadata to include with the documents.
            write_batch_size: The batch size for writing documents.
            scroll_size: The scroll size for reading documents.
            payload_fields_to_index: List of payload fields to index.
            content_key (Optional[str]): The field used to store content in the storage.
            dry_run_config (Optional[DryRunConfig]): Configuration for dry run mode. Defaults to None.
        """
        super().__init__(dry_run_config=dry_run_config)

        self._client = client
        if self._client is None:
            connection = connection or QdrantConnection()
            self._client = connection.connect()

        # Store the Qdrant client specific attributes
        self.location = location
        self.url = url
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.api_key = api_key
        self.prefix = prefix
        self.timeout = timeout
        self.host = host
        self.path = path
        self.force_disable_check_same_thread = force_disable_check_same_thread
        self.metadata = metadata or {}
        self.api_key = api_key

        # Store the Qdrant collection specific attributes
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from
        self.wait_result_from_api = wait_result_from_api
        self.create_if_not_exist = create_if_not_exist
        self.recreate_index = recreate_index
        self.payload_fields_to_index = payload_fields_to_index
        self.use_sparse_embeddings = use_sparse_embeddings
        self.sparse_idf = use_sparse_embeddings and sparse_idf
        self.dimension = dimension
        self.on_disk = on_disk
        self.metric = metric
        self.index_name = index_name
        self.return_embedding = return_embedding
        self.write_batch_size = write_batch_size
        self.scroll_size = scroll_size
        self.content_key = content_key
        self._indexed_payload_fields: set[str] = set()
        self._auto_index_attempted: set[str] = set()

    @property
    def client(self):
        if not self._client:
            self._client = qdrant_client.QdrantClient(
                location=self.location,
                url=self.url,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                https=self.https,
                api_key=self.api_key.resolve_value() if self.api_key else None,
                prefix=self.prefix,
                timeout=self.timeout,
                host=self.host,
                path=self.path,
                metadata=self.metadata,
                force_disable_check_same_thread=self.force_disable_check_same_thread,
            )
            # Make sure the collection is properly set up
            self._set_up_collection(
                collection_name=self.index_name,
                embedding_dim=self.dimension,
                create_if_not_exist=self.create_if_not_exist,
                recreate_collection=self.recreate_index,
                similarity=self.metric,
                use_sparse_embeddings=self.use_sparse_embeddings,
                sparse_idf=self.sparse_idf,
                on_disk=self.on_disk,
                payload_fields_to_index=self.payload_fields_to_index,
            )
        return self._client

    def _collection_exists(self, collection_name: str) -> bool:
        """Safely determine if the collection is present in Qdrant."""

        client = self._client or self.client

        try:
            return client.collection_exists(collection_name)
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                try:
                    client.get_collection(collection_name)
                    return True
                except UnexpectedResponse as inner_exc:
                    if getattr(inner_exc, "status_code", None) == 404:
                        return False
                    raise
            raise
        except ValueError:
            return False

    def count_documents(self) -> int:
        """Returns the number of documents present in the Document Store.

        Returns:
            The number of documents in the Document Store.
        """
        try:
            response = self.client.count(
                collection_name=self.index_name,
            )
            return response.count
        except (UnexpectedResponse, ValueError):
            # Qdrant local raises ValueError if the collection is not found, but
            # with the remote server UnexpectedResponse is raised. Until that's unified,
            # we need to catch both.
            return 0

    def filter_documents(
        self,
        filters: dict[str, Any] | rest.Filter | None = None,
    ) -> list[Document]:
        """Returns the documents that match the provided filters.

        For a detailed specification of the filters, refer to the
        [documentation](https://docs.dynamiq.deepset.ai/docs/metadata-filtering)

        Args:
            filters: The filters to apply to the document list.

        Returns:
            A list of documents that match the given filters.
        """
        if filters and not isinstance(filters, dict) and not isinstance(filters, rest.Filter):
            msg = "Filter must be a dictionary or an instance of `qdrant_client.http.models.Filter`"
            raise ValueError(msg)

        if filters and not isinstance(filters, rest.Filter) and "operator" not in filters:
            raise ValueError("Filter must contain an 'operator' key")

        return list(
            self.get_documents_generator(
                filters,
            )
        )

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
        content_key: str | None = None,
    ) -> int:
        """Writes documents to Qdrant using the specified policy.

        The QdrantDocumentStore can handle duplicate documents based on the given policy. The available policies are:
        - `FAIL`: The operation will raise an error if any document already exists.
        - `OVERWRITE`: Existing documents will be overwritten with the new ones.
        - `SKIP`: Existing documents will be skipped, and only new documents will be added.

        Args:
            documents: A list of Document objects to write to Qdrant.
            policy: The policy for handling duplicate documents.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            The number of documents written to the document store.
        """
        if not self._collection_exists(self.index_name):
            if self.create_if_not_exist:
                logger.info(f"Collection {self.index_name} doesn't exist. Creating...")
                self._set_up_collection(
                    collection_name=self.index_name,
                    embedding_dim=self.dimension,
                    create_if_not_exist=True,
                    recreate_collection=self.recreate_index,
                    similarity=self.metric,
                    use_sparse_embeddings=self.use_sparse_embeddings,
                    sparse_idf=self.sparse_idf,
                    on_disk=self.on_disk,
                )
            else:
                raise QdrantStoreError(f"Collection {self.index_name} doesn't exist")
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)

        if len(documents) == 0:
            logger.warning("Calling QdrantDocumentStore.write_documents() with empty list")
            return 0

        document_objects = self._handle_duplicate_documents(
            documents=documents,
            index=self.index_name,
            policy=policy,
        )

        batched_documents = get_batches_from_generator(document_objects, self.write_batch_size)
        for document_batch in batched_documents:
            batch = convert_dynamiq_documents_to_qdrant_points(
                document_batch,
                use_sparse_embeddings=self.use_sparse_embeddings,
                content_key=content_key or self.content_key,
            )

            self.client.upsert(
                collection_name=self.index_name,
                points=batch,
                wait=self.wait_result_from_api,
            )

        self._track_documents([doc.id for doc in documents])

        return len(document_objects)

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """Deletes documents that match the provided `document_ids` from the document store.

        Args:
            document_ids: The document ids to delete.
            delete_all (bool): If True, delete all documents. Defaults to False.
        """
        if delete_all:
            self.client.delete_collection(collection_name=self.index_name)
        elif document_ids:
            ids = [convert_id(_id) for _id in document_ids]
            try:
                self.client.delete(
                    collection_name=self.index_name,
                    points_selector=ids,
                    wait=self.wait_result_from_api,
                )
            except KeyError:
                logger.warning(
                    "Called QdrantDocumentStore.delete_documents() on a non-existing ID",
                )
        else:
            raise ValueError("Either `document_ids` or `delete_all` must be provided.")

    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Delete documents from the DocumentStore based on the provided filters.

        Args:
            filters (dict[str, Any]): The filters to apply to the document list.
        """
        if filters:
            documents = self.filter_documents(filters=filters)
            document_ids = [doc.id for doc in documents]
            self.delete_documents(document_ids=document_ids)
        else:
            raise ValueError("No filters provided to delete documents.")

    def delete_collection(self, collection_name: str | None = None):
        """
        Delete a Qdrant collection.

        Args:
            collection_name (str | None): Name of the collection to delete.
        """
        try:
            collection_to_delete = collection_name or self.index_name
            self.client.delete_collection(collection_name=collection_to_delete)
            logger.info(f"Deleted collection '{collection_to_delete}'.")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_to_delete}': {e}")
            raise

    def get_documents_generator(
        self,
        filters: dict[str, Any] | rest.Filter | None = None,
        include_embeddings: bool = False,
        content_key: str | None = None,
    ) -> Generator[Document, None, None]:
        """Returns a generator that yields documents from Qdrant based on the provided filters.

        Args:
            filters: Filters applied to the retrieved documents.
            include_embeddings: Whether to include the embeddings of the retrieved documents.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            A generator that yields documents retrieved from Qdrant.
        """

        index = self.index_name
        qdrant_filters = convert_filters_to_qdrant(filters)

        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = self._execute_with_payload_index_retry(
                lambda: self.client.scroll(
                    collection_name=index,
                    scroll_filter=qdrant_filters,
                    limit=self.scroll_size,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=include_embeddings,
                )
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId) and next_offset.num == 0 and next_offset.uuid == ""
            )

            for record in records:
                yield convert_qdrant_point_to_dynamiq_document(
                    record,
                    use_sparse_embeddings=self.use_sparse_embeddings,
                    content_key=content_key or self.content_key,
                )

    def list_documents(self, include_embeddings: bool = False, content_key: str | None = None) -> list[Document]:
        """Returns a list of all documents in the Document Store.

        Args:
            include_embeddings: Whether to include the embeddings of the retrieved documents.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            A list of all documents in the Document Store.
        """
        return list(
            self.get_documents_generator(
                include_embeddings=include_embeddings, content_key=content_key or self.content_key
            )
        )

    def get_documents_by_id(
        self, ids: list[str], index: str | None = None, content_key: str | None = None
    ) -> list[Document]:
        """Retrieves documents from Qdrant by their IDs.

        Args:
            ids: A list of document IDs to retrieve.
            index: The name of the index to retrieve documents from.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            A list of documents.
        """
        index = index or self.index_name

        documents: list[Document] = []

        ids = [convert_id(_id) for _id in ids]
        records = self.client.retrieve(
            collection_name=index,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )

        for record in records:
            documents.append(
                convert_qdrant_point_to_dynamiq_document(
                    record,
                    use_sparse_embeddings=self.use_sparse_embeddings,
                    content_key=content_key or self.content_key,
                )
            )
        return documents

    def _query_by_sparse(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: dict[str, Any] | rest.Filter | None = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
        score_threshold: float | None = None,
        content_key: str | None = None,
    ) -> list[Document]:
        """Queries Qdrant using a sparse embedding and returns the most relevant documents.

        Args:
            query_sparse_embedding: Sparse embedding of the query.
            filters: Filters applied to the retrieved documents.
            top_k: Maximum number of documents to return.
            scale_score: Whether to scale the scores of the retrieved documents.
            return_embedding: Whether to return the embeddings of the retrieved documents.
            score_threshold: A minimal score threshold for the result. Score of the returned result might be higher or
                smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only
                higher scores will be returned.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List of documents that are most similar to `query_sparse_embedding`.

        Raises:
            QdrantStoreError: If the Document Store was initialized with `use_sparse_embeddings=False`.
        """

        if not self.use_sparse_embeddings:
            message = (
                "You are trying to query using sparse embeddings, but the Document Store "
                "was initialized with `use_sparse_embeddings=False`. "
            )
            raise QdrantStoreError(message)

        qdrant_filters = convert_filters_to_qdrant(filters)
        query_indices = query_sparse_embedding.indices
        query_values = query_sparse_embedding.values
        response = self._execute_with_payload_index_retry(
            lambda: self.client.query_points(
                collection_name=self.index_name,
                query=rest.SparseVector(
                    indices=query_indices,
                    values=query_values,
                ),
                using=SPARSE_VECTORS_NAME,
                query_filter=qdrant_filters,
                limit=top_k,
                with_vectors=return_embedding,
                score_threshold=score_threshold,
            )
        )
        points = response.points
        results = [
            convert_qdrant_point_to_dynamiq_document(
                point, use_sparse_embeddings=self.use_sparse_embeddings, content_key=content_key or self.content_key
            )
            for point in points
        ]
        if scale_score:
            for document in results:
                score = document.score
                score = float(1 / (1 + np.exp(-score / 100)))
                document.score = score
        return results

    def _query_by_embedding(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | rest.Filter | None = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
        score_threshold: float | None = None,
        content_key: str | None = None,
    ) -> list[Document]:
        """Queries Qdrant using a dense embedding and returns the most relevant documents.

        Args:
            query_embedding: Dense embedding of the query.
            filters: Filters applied to the retrieved documents.
            top_k: Maximum number of documents to return.
            scale_score: Whether to scale the scores of the retrieved documents.
            return_embedding: Whether to return the embeddings of the retrieved documents.
            score_threshold: A minimal score threshold for the result. Score of the returned result might be higher or
                smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only
                higher scores will be returned.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List of documents that are most similar to `query_embedding`.
        """
        qdrant_filters = convert_filters_to_qdrant(filters)

        response = self._execute_with_payload_index_retry(
            lambda: self.client.query_points(
                collection_name=self.index_name,
                query=query_embedding,
                using=DENSE_VECTORS_NAME if self.use_sparse_embeddings else None,
                query_filter=qdrant_filters,
                limit=top_k,
                with_vectors=return_embedding,
                score_threshold=score_threshold,
            )
        )
        points = response.points
        results = [
            convert_qdrant_point_to_dynamiq_document(
                point, use_sparse_embeddings=self.use_sparse_embeddings, content_key=content_key or self.content_key
            )
            for point in points
        ]
        if scale_score:
            for document in results:
                score = document.score
                if str(self.metric).lower() == "cosine":
                    score = (score + 1) / 2
                else:
                    score = float(1 / (1 + np.exp(-score / 100)))
                document.score = score
        return results

    def _query_hybrid(
        self,
        query_embedding: list[float],
        query_sparse_embedding: SparseEmbedding,
        filters: dict[str, Any] | rest.Filter | None = None,
        top_k: int = 10,
        return_embedding: bool = False,
        score_threshold: float | None = None,
        content_key: str | None = None,
    ) -> list[Document]:
        """Retrieves documents based on dense and sparse embeddings and fuses the results using Reciprocal Rank Fusion.

        This method is not part of the public interface of `QdrantDocumentStore` and shouldn't be used directly. Use the
        `QdrantHybridRetriever` instead.

        Args:
            query_embedding: Dense embedding of the query.
            query_sparse_embedding: Sparse embedding of the query.
            filters: Filters applied to the retrieved documents.
            top_k: Maximum number of documents to return.
            return_embedding: Whether to return the embeddings of the retrieved documents.
            score_threshold: A minimal score threshold for the result. Score of the returned result might be higher or
                smaller than the threshold depending on the Distance function used. E.g. for cosine similarity only
                higher scores will be returned.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            List of Document that are most similar to `query_embedding` and `query_sparse_embedding`.

        Raises:
            QdrantStoreError: If the Document Store was initialized with `use_sparse_embeddings=False`.
        """

        # This implementation is based on the code from the Python Qdrant client:
        # https://github.com/qdrant/qdrant-client/blob/8e3ea58f781e4110d11c0a6985b5e6bb66b85d33/qdrant_client/qdrant_fastembed.py#L519
        if not self.use_sparse_embeddings:
            message = (
                "You are trying to query using sparse embeddings, but the Document Store "
                "was initialized with `use_sparse_embeddings=False`. "
            )
            raise QdrantStoreError(message)

        qdrant_filters = convert_filters_to_qdrant(filters)

        try:
            response = self._execute_with_payload_index_retry(
                lambda: self.client.query_points(
                    collection_name=self.index_name,
                    prefetch=[
                        rest.Prefetch(
                            query=rest.SparseVector(
                                indices=query_sparse_embedding.indices,
                                values=query_sparse_embedding.values,
                            ),
                            using=SPARSE_VECTORS_NAME,
                            filter=qdrant_filters,
                        ),
                        rest.Prefetch(
                            query=query_embedding,
                            using=DENSE_VECTORS_NAME,
                            filter=qdrant_filters,
                        ),
                    ],
                    query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=return_embedding,
                )
            )
            points = response.points
        except Exception as e:
            msg = "Error during hybrid search"
            raise QdrantStoreError(msg) from e

        results = [
            convert_qdrant_point_to_dynamiq_document(
                point, use_sparse_embeddings=True, content_key=content_key or self.content_key
            )
            for point in points
        ]

        return results

    def get_distance(self, similarity: str | QdrantSimilarityMetric) -> rest.Distance:
        """Retrieves the distance metric for the specified similarity measure.

        Args:
            similarity: The similarity measure to retrieve the distance.

        Returns:
            The corresponding rest.Distance object.

        Raises:
            QdrantStoreError: If the provided similarity measure is not supported.
        """
        if isinstance(similarity, str):
            try:
                similarity = QdrantSimilarityMetric(similarity.lower())
            except ValueError as exc:
                msg = (
                    f"Provided similarity '{similarity}' is not supported by Qdrant document store. "
                    f"Please choose one of the options: {', '.join(metric.value for metric in QdrantSimilarityMetric)}"
                )
                raise QdrantStoreError(msg) from exc

        try:
            return self.DISTANCE_BY_SIMILARITY[similarity]
        except KeyError as ke:
            msg = (
                f"Provided similarity '{similarity}' is not supported by Qdrant "
                f"document store. Please choose one of the options: "
                f"{', '.join(metric.value for metric in QdrantSimilarityMetric)}"
            )
            raise QdrantStoreError(msg) from ke

    def _create_payload_index(self, collection_name: str, payload_fields_to_index: list[dict] | None = None):
        """Create payload index for the collection if payload_fields_to_index is provided."""

        if not payload_fields_to_index:
            return

        for payload_index in payload_fields_to_index:
            field_name = payload_index["field_name"]
            field_schema = payload_index["field_schema"]

            if field_name in self._indexed_payload_fields:
                continue

            try:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=True,
                )
            except UnexpectedResponse as exc:
                status_code = getattr(exc, "status_code", None)
                content = exc.content.decode("utf-8", "ignore") if getattr(exc, "content", None) else ""
                if status_code in {400, 409} and "already exists" in content.lower():
                    logger.debug(
                        "Payload index for field %s already exists in collection %s", field_name, collection_name
                    )
                else:
                    raise

            self._indexed_payload_fields.add(field_name)
            self._auto_index_attempted.add(field_name)

    def _execute_with_payload_index_retry(self, operation: Callable[[], T]) -> T:
        """Execute and auto-create missing payload indexes, retrying a few times if needed."""

        attempts = 0
        last_exc: UnexpectedResponse | None = None

        while True:
            try:
                return operation()
            except UnexpectedResponse as exc:
                last_exc = exc
                attempts += 1

                if not self._try_create_missing_payload_index(exc):
                    raise

                if attempts > 5:
                    raise last_exc

    def _extract_error_text(self, exc: UnexpectedResponse) -> str:
        """Extract the error text from a Qdrant UnexpectedResponse."""

        if not getattr(exc, "content", None):
            return ""

        payload = exc.content.decode("utf-8", "ignore")

        try:
            data = json.loads(payload)
        except Exception:
            return payload

        if isinstance(data, dict):
            status = data.get("status")
            if isinstance(status, dict):
                error = status.get("error")
                if isinstance(error, str):
                    return error
        return payload

    def _try_create_missing_payload_index(self, exc: UnexpectedResponse) -> bool:
        """Inspect an error and create the required payload index if the server requests it."""

        if getattr(exc, "status_code", None) != 400 or not getattr(exc, "content", None):
            return False

        message = self._extract_error_text(exc).replace('"', '"')
        logger.debug("Qdrant query error: %s", message)

        match = self._MISSING_INDEX_PATTERN.search(message)
        if not match:
            logger.debug("No payload index hint found in error message")
            return False

        field_name = match.group("field")
        if field_name in self._auto_index_attempted:
            return False

        type_candidates = [candidate.strip().strip("\"' ") for candidate in match.group("types").split(",")]
        schema = self._resolve_payload_schema(type_candidates)

        logger.info(
            "Creating payload index for missing field %s in collection %s (schema=%s)",
            field_name,
            self.index_name,
            schema.value,
        )

        self._auto_index_attempted.add(field_name)
        self._create_payload_index(
            collection_name=self.index_name,
            payload_fields_to_index=[{"field_name": field_name, "field_schema": schema}],
        )
        return True

    def _resolve_payload_schema(self, type_candidates: list[str]) -> rest.PayloadSchemaType:
        """Resolve the payload schema type, defaulting to KEYWORD when unknown."""

        for candidate in type_candidates:
            schema = self._PAYLOAD_SCHEMA_BY_NAME.get(candidate.lower())
            if schema:
                return schema
        return rest.PayloadSchemaType.KEYWORD

    def _set_up_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        create_if_not_exist: bool,
        recreate_collection: bool,
        similarity: str,
        use_sparse_embeddings: bool,
        sparse_idf: bool,
        on_disk: bool = False,
        payload_fields_to_index: list[dict] | None = None,
    ):
        """Sets up the Qdrant collection with the specified parameters.

        Args:
            collection_name: The name of the collection to set up.
            embedding_dim: The dimension of the embeddings.
            recreate_collection: Whether to recreate the collection if it already exists.
            similarity: The similarity measure to use.
            use_sparse_embeddings: Whether to use sparse embeddings.
            sparse_idf: Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required
                for BM42.
            on_disk: Whether to store the collection on disk.
            payload_fields_to_index: List of payload fields to index.

        Raises:
            QdrantStoreError: If the collection exists with incompatible settings.
            ValueError: If the collection exists with a different similarity measure or embedding dimension.
        """
        distance = self.get_distance(similarity)
        collection_exists = self._collection_exists(collection_name)

        should_create = (not collection_exists and create_if_not_exist) or recreate_collection

        if not collection_exists and not create_if_not_exist:
            msg = f"Collection '{collection_name}' does not exist in Qdrant."
            raise QdrantStoreError(msg)

        if should_create:
            logger.info(f"{'Creating' if not collection_exists else 'Recreating'} collection {collection_name}")
            self.recreate_collection(
                collection_name=collection_name,
                distance=distance,
                embedding_dim=embedding_dim,
                on_disk=on_disk,
                use_sparse_embeddings=use_sparse_embeddings,
                sparse_idf=sparse_idf,
            )
            self._track_collection(collection_name)
            if payload_fields_to_index:
                self._create_payload_index(collection_name, payload_fields_to_index)
            return

        collection_info = self.client.get_collection(collection_name)

        has_named_vectors = (
            isinstance(collection_info.config.params.vectors, dict)
            and DENSE_VECTORS_NAME in collection_info.config.params.vectors
        )

        if self.use_sparse_embeddings and not has_named_vectors:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it has been originally created without sparse embedding vectors. "
                f"If you want to use that collection, you can set `use_sparse_embeddings=False`. "
                f"To use sparse embeddings, you need to recreate the collection or migrate the existing one. "
                f"See `migrate_to_sparse_embeddings_support` function in "
                f"`dynamiq_integrations.document_stores.qdrant`."
            )
            raise QdrantStoreError(msg)

        elif not self.use_sparse_embeddings and has_named_vectors:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it has been originally created with sparse embedding vectors."
                f"If you want to use that collection, please set `use_sparse_embeddings=True`."
            )
            raise QdrantStoreError(msg)

        if self.use_sparse_embeddings:
            current_distance = collection_info.config.params.vectors[DENSE_VECTORS_NAME].distance
            current_vector_size = collection_info.config.params.vectors[DENSE_VECTORS_NAME].size
        else:
            current_distance = collection_info.config.params.vectors.distance
            current_vector_size = collection_info.config.params.vectors.size

        if current_distance != distance:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a similarity '{current_distance.name}'. "
                f"If you want to use that collection, but with a different "
                f"similarity, please set `recreate_collection=True` argument."
            )
            raise ValueError(msg)

        if current_vector_size != embedding_dim:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a vector size '{current_vector_size}'. "
                f"If you want to use that collection, but with a different "
                f"vector size, please set `recreate_collection=True` argument."
            )
            raise ValueError(msg)

    def recreate_collection(
        self,
        collection_name: str,
        distance,
        embedding_dim: int,
        on_disk: bool | None = None,
        use_sparse_embeddings: bool | None = None,
        sparse_idf: bool = False,
    ):
        """Recreates the Qdrant collection with the specified parameters.

        Args:
            collection_name: The name of the collection to recreate.
            distance: The distance metric to use for the collection.
            embedding_dim: The dimension of the embeddings.
            on_disk: Whether to store the collection on disk.
            use_sparse_embeddings: Whether to use sparse embeddings.
            sparse_idf: Whether to compute the Inverse Document Frequency (IDF) when using sparse embeddings. Required
                for BM42.
        """
        if on_disk is None:
            on_disk = self.on_disk

        if use_sparse_embeddings is None:
            use_sparse_embeddings = self.use_sparse_embeddings

        # dense vectors configuration
        vectors_config = rest.VectorParams(size=embedding_dim, on_disk=on_disk, distance=distance)

        # Reset cached payload index information when recreating a collection
        self._indexed_payload_fields.clear()
        self._auto_index_attempted.clear()

        if use_sparse_embeddings:
            # in this case, we need to define named vectors
            vectors_config = {DENSE_VECTORS_NAME: vectors_config}

            sparse_vectors_config = {
                SPARSE_VECTORS_NAME: rest.SparseVectorParams(
                    index=rest.SparseIndexParams(
                        on_disk=on_disk,
                    ),
                    modifier=rest.Modifier.IDF if sparse_idf else None,
                ),
            }

        if self._collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config if use_sparse_embeddings else None,
                shard_number=self.shard_number,
                replication_factor=self.replication_factor,
                write_consistency_factor=self.write_consistency_factor,
                on_disk_payload=self.on_disk_payload,
                hnsw_config=self.hnsw_config,
                optimizers_config=self.optimizers_config,
                wal_config=self.wal_config,
                quantization_config=self.quantization_config,
                init_from=self.init_from,
            )
        except UnexpectedResponse as exc:
            if getattr(exc, "status_code", None) == 404:
                msg = (
                    f"Failed to create collection '{collection_name}'. Qdrant returned 404 for the create API. "
                    f"Double-check the service URL or required prefix for your deployment."
                )
                raise QdrantStoreError(msg) from exc
            raise

    def _handle_duplicate_documents(
        self,
        documents: list[Document],
        index: str | None = None,
        policy: DuplicatePolicy = None,
    ):
        """Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        Args:
            documents: A list of Dynamiq Document objects.
            index: Name of the index.
            policy: The duplicate policy to use when writing documents.

        Returns:
            A list of Dynamiq Document objects.
        """

        index = index or self.index_name
        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index)
            ids_exist_in_db: list[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and policy == DuplicatePolicy.FAIL:
                msg = f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{index}'."
                raise DuplicateDocumentError(msg)

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    def _drop_duplicate_documents(self, documents: list[Document], index: str | None = None) -> list[Document]:
        """Drop duplicate documents based on same hash ID.

        Args:
            documents: A list of Dynamiq Document objects.
            index: Name of the index.

        Returns:
            A list of Dynamiq Document objects.
        """
        _hash_ids: set = set()
        _documents: list[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    index or self.index_name,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents
