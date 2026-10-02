import base64
import re
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np
from turbopuffer import BadRequestError, NotFoundError

from dynamiq.connections import Turbopuffer
from dynamiq.nodes.dry_run import DryRunMixin
from dynamiq.storages.vector.base import BaseVectorStore, BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreDuplicateDocumentException, VectorStoreException
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.utils import DEFAULT_SEARCHABLE_TEXT_METADATA_FIELDS
from dynamiq.types import Document
from dynamiq.types.dry_run import DryRunConfig
from dynamiq.utils.logger import logger

from .attributes import MAX_FILTERABLE_BYTES, STRING, coerce_value, infer_type, longest_string_bytes, normalize_value
from .filters import MATCH_ALL, MATCH_NONE, combine_and, convert_filters, to_turbopuffer_filter

if TYPE_CHECKING:
    from turbopuffer import Turbopuffer as TurbopufferClient
    from turbopuffer.types import Row


DISTANCE_METRIC = "cosine_distance"
VECTOR_ATTRIBUTE = "vector"
ID_ATTRIBUTE = "id"
DISTANCE_ATTRIBUTE = "$dist"
MAX_TOP_K = 10_000
MAX_ATTRIBUTES = 1024
MAX_ATTRIBUTE_NAME_BYTES = 128
MAX_ID_BYTES = 64
MAX_BM25_QUERY_LENGTH = 8192
DELETE_BATCH_SIZE = 10_000

# Pinned so a namespace keeps the same tokenizer when Turbopuffer changes its default. Stemming and
# stopword removal stay off to match how Weaviate scores keyword searches.
FULL_TEXT_SEARCH = {
    "tokenizer": "word_v4",
    "language": "english",
    "stemming": False,
    "remove_stopwords": False,
    "case_sensitive": False,
}


class TurbopufferWriterVectorStoreParams(BaseWriterVectorStoreParams):
    """Parameters for writing to a Turbopuffer namespace, named by index_name."""


class TurbopufferRetrieverVectorStoreParams(BaseVectorStoreParams):
    """Parameters for searching a Turbopuffer namespace, named by index_name."""

    alpha: float = 0.5
    max_vector_distance: float | None = None


class TurbopufferVectorStore(BaseVectorStore, DryRunMixin):
    """
    A Document Store for Turbopuffer.

    Each index is a Turbopuffer namespace. Turbopuffer creates a namespace on its first write, so
    reading a namespace that was never written returns no documents instead of failing.
    """

    PATTERN_NAMESPACE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

    def __init__(
        self,
        connection: Turbopuffer | None = None,
        client: Optional["TurbopufferClient"] = None,
        index_name: str = "default",
        create_if_not_exist: bool = False,
        content_key: str = "content",
        alpha: float = 0.5,
        batch_size: int = 500,
        dry_run_config: DryRunConfig | None = None,
    ):
        """
        Initialize a new instance of TurbopufferVectorStore.

        Args:
            connection (Turbopuffer | None): A Turbopuffer connection object. If None, a new one is
                created.
            client (Optional[TurbopufferClient]): A Turbopuffer client. If None, one is created from the
                connection.
            index_name (str): The namespace to use. Defaults to "default".
            create_if_not_exist (bool): Turbopuffer creates namespaces on the first write, so this only
                marks a namespace that does not exist yet for dry run cleanup.
            content_key (str): The attribute used to store content. Defaults to "content".
            alpha (float): The weight of vector similarity in hybrid retrieval, from 0.0 (keyword only)
                to 1.0 (vector only). Defaults to 0.5.
            batch_size (int): The number of documents sent in one write request. Defaults to 500.
            dry_run_config (Optional[DryRunConfig]): Configuration for dry run mode.

        Raises:
            ValueError: If the namespace name is not valid in Turbopuffer.
        """
        super().__init__(dry_run_config=dry_run_config)

        if not self.is_valid_namespace_name(index_name):
            raise ValueError(
                f"Namespace name '{index_name}' is invalid. It must match the pattern {self.PATTERN_NAMESPACE.pattern}"
            )

        self.client = client
        if self.client is None:
            if connection is None:
                connection = Turbopuffer()
            self.client = connection.connect()

        self.index_name = index_name
        self.content_key = content_key
        self.alpha = alpha
        self.batch_size = batch_size
        self._namespace = self.client.namespace(index_name)
        self._schema: dict[str, dict] | None = None

        if create_if_not_exist and not self._namespace.exists():
            self._track_collection(index_name)

    @classmethod
    def is_valid_namespace_name(cls, name: str) -> bool:
        return bool(cls.PATTERN_NAMESPACE.fullmatch(name or ""))

    def close(self):
        """Close the connection to Turbopuffer."""
        if self.client:
            self.client.close()

    # Schema

    def _load_schema(self, refresh: bool = False) -> dict[str, dict]:
        """Return the namespace schema, empty when the namespace does not exist yet."""
        if self._schema is None or refresh:
            try:
                schema = self._namespace.schema()
            except NotFoundError:
                schema = {}
            self._schema = {
                name: config.model_dump(exclude_none=True) if hasattr(config, "model_dump") else dict(config)
                for name, config in schema.items()
            }
        return self._schema

    @staticmethod
    def _is_filterable(config: dict[str, Any]) -> bool:
        if "filterable" in config:
            return bool(config["filterable"])
        return not config.get("full_text_search")

    def _full_text_fields(self, content_key: str | None = None) -> list[str]:
        """Return the attributes keyword search runs over, content first."""
        schema = self._load_schema()
        preferred = (content_key or self.content_key, *DEFAULT_SEARCHABLE_TEXT_METADATA_FIELDS)
        return [name for name in dict.fromkeys(preferred) if schema.get(name, {}).get("full_text_search")]

    # Writing

    def _encode_vector(self, embedding: list[float]) -> str:
        return base64.b64encode(np.asarray(embedding, dtype="<f4").tobytes()).decode("ascii")

    def _declare(
        self,
        name: str,
        value: Any,
        schema: dict[str, dict],
        declarations: dict[str, dict],
        content_key: str,
    ) -> tuple[Any, bool]:
        """Fit a value to the attribute's type, declaring the attribute when it is new.

        Returns the value to store and whether to store it.
        """
        config = declarations.get(name) or schema.get(name)

        if config is None:
            if len(schema) + len(declarations) >= MAX_ATTRIBUTES:
                logger.warning(
                    f"Namespace '{self.index_name}' reached {MAX_ATTRIBUTES} attributes, so '{name}' is not stored."
                )
                return None, False
            if name == content_key or name in DEFAULT_SEARCHABLE_TEXT_METADATA_FIELDS:
                config = {"type": STRING, "full_text_search": dict(FULL_TEXT_SEARCH)}
                if name != content_key:
                    config["filterable"] = True
            else:
                config = {"type": infer_type(value)}
            declarations[name] = config

        value = coerce_value(value, config["type"])
        if value is None:
            logger.warning(f"Value of '{name}' does not fit type '{config['type']}' and is not stored.")
            return None, False

        if self._is_filterable(config) and longest_string_bytes(value) > MAX_FILTERABLE_BYTES:
            # Turbopuffer rejects filterable values over 4 KiB, so the attribute stops being filterable.
            declarations[name] = {**config, "filterable": False}

        return value, True

    def _to_row(
        self,
        document: Document,
        content_key: str,
        schema: dict[str, dict],
        declarations: dict[str, dict],
    ) -> dict[str, Any]:
        """Convert a Document into a Turbopuffer row, declaring any new attributes."""
        if not isinstance(document, Document):
            raise ValueError(f"Expected a Document, got '{type(document)}' instead.")

        document_id = str(document.id)
        if len(document_id.encode("utf-8")) > MAX_ID_BYTES:
            raise ValueError(f"Document id '{document_id}' is longer than {MAX_ID_BYTES} bytes.")

        row: dict[str, Any] = {ID_ATTRIBUTE: document_id}
        if document.embedding is not None:
            row[VECTOR_ATTRIBUTE] = self._encode_vector(document.embedding)

        content, keep = self._declare(content_key, document.content or "", schema, declarations, content_key)
        if keep:
            row[content_key] = content

        for key, raw_value in (document.metadata or {}).items():
            if key in (ID_ATTRIBUTE, VECTOR_ATTRIBUTE, content_key) or key.startswith("$"):
                logger.warning(f"Metadata key '{key}' is reserved in Turbopuffer and is not stored.")
                continue
            if len(key.encode("utf-8")) > MAX_ATTRIBUTE_NAME_BYTES:
                logger.warning(f"Metadata key '{key[:32]}...' is longer than {MAX_ATTRIBUTE_NAME_BYTES} bytes.")
                continue

            value = normalize_value(raw_value)
            if value is None:
                continue

            value, keep = self._declare(key, value, schema, declarations, content_key)
            if keep:
                row[key] = value

        return row

    def _write_batch(self, documents: list[Document], content_key: str, policy: DuplicatePolicy) -> int:
        """Write one batch of documents, refreshing a stale schema once if the write is rejected."""
        for attempt in range(2):
            schema = self._load_schema(refresh=attempt > 0)
            declarations: dict[str, dict] = {}
            rows = [self._to_row(doc, content_key, schema, declarations) for doc in documents]

            params: dict[str, Any] = {"upsert_rows": rows}
            if any(VECTOR_ATTRIBUTE in row for row in rows):
                params["distance_metric"] = DISTANCE_METRIC
            if declarations:
                params["schema"] = declarations
            if policy == DuplicatePolicy.SKIP:
                # A document that does not exist yet has a null id, so existing documents are kept.
                params["upsert_condition"] = [ID_ATTRIBUTE, "Eq", None]

            try:
                response = self._namespace.write(**params)
            except BadRequestError as e:
                if attempt == 0:
                    # Another writer may have added attributes since the schema was read.
                    logger.debug(f"Turbopuffer rejected a write to '{self.index_name}', retrying: {e}")
                    continue
                raise VectorStoreException(f"Failed to write documents to Turbopuffer: {e}") from e

            schema.update(declarations)
            return response.rows_affected

        return 0

    def _existing_ids(self, ids: list[str]) -> list[str]:
        existing = []
        for start in range(0, len(ids), MAX_TOP_K):
            chunk = ids[start : start + MAX_TOP_K]
            rows = self._query_rows(
                rank_by=[ID_ATTRIBUTE, "asc"], top_k=len(chunk), filters=[ID_ATTRIBUTE, "In", chunk]
            )
            existing.extend(str(row.id) for row in rows)
        return existing

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        content_key: str | None = None,
    ) -> int:
        """
        Write documents to Turbopuffer.

        Metadata is stored as flat attributes. Dicts become JSON strings, values that do not fit an
        attribute's existing type are left out with a warning, and string attributes with a value
        over 4 KiB stop being filterable.

        Args:
            documents (list[Document]): The documents to write.
            policy (DuplicatePolicy): How to handle documents whose id already exists.
            content_key (Optional[str]): The attribute used to store content.

        Returns:
            int: The number of documents written.

        Raises:
            ValueError: If an input is not a Document or a document id is longer than 64 bytes.
            VectorStoreDuplicateDocumentException: If duplicates are found with the FAIL policy.
            VectorStoreException: If Turbopuffer rejects the write.
        """
        if not documents:
            return 0

        self._track_documents([doc.id for doc in documents])
        content_key = content_key or self.content_key

        # Turbopuffer rejects a write that holds the same id twice, so the last copy of each wins.
        unique = list({str(doc.id): doc for doc in documents}.values())

        if policy == DuplicatePolicy.FAIL:
            existing = self._existing_ids([str(doc.id) for doc in unique])
            if existing:
                raise VectorStoreDuplicateDocumentException(
                    f"IDs '{', '.join(existing)}' already exist in the document store."
                )

        written = 0
        for start in range(0, len(unique), self.batch_size):
            written += self._write_batch(unique[start : start + self.batch_size], content_key, policy)

        return written

    def replace_document_metadata(self, document_ids: str | list[str], metadata: dict[str, Any]) -> None:
        """
        Replace the metadata of one or more documents (full replacement).

        Args:
            document_ids (str | list[str]): The id, or list of ids, of the documents to update.
            metadata (dict[str, Any]): The new metadata that fully replaces the existing one.

        Raises:
            VectorStoreException: If any of the given ids does not exist.
        """
        ids = self._normalize_document_ids(document_ids)
        found = {doc.id: doc for doc in self.get_documents_by_id(ids, include_embeddings=True)}
        missing = [document_id for document_id in ids if document_id not in found]
        if missing:
            raise VectorStoreException(f"Documents {missing} not found in namespace '{self.index_name}'")

        documents = [
            Document(
                id=document_id,
                content=found[document_id].content,
                embedding=found[document_id].embedding,
                metadata=metadata,
            )
            for document_id in ids
        ]
        if documents:
            self.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

    # Reading

    def _attributes(self, include_embeddings: bool) -> dict[str, Any]:
        if include_embeddings:
            return {"include_attributes": True}
        return {"exclude_attributes": [VECTOR_ATTRIBUTE]}

    def _query_rows(self, **params: Any) -> list["Row"]:
        """Run a query, returning no rows when the namespace does not exist yet."""
        try:
            return self._namespace.query(**params).rows or []
        except NotFoundError:
            return []

    def _filter(self, filters: dict[str, Any] | None) -> Any:
        """Convert filters against the namespace schema."""
        if not filters:
            return MATCH_ALL
        return convert_filters(filters, self._load_schema())

    @staticmethod
    def _with_filter(params: dict[str, Any], converted: Any) -> dict[str, Any]:
        """Add a converted filter to query parameters, leaving it out when it matches everything."""
        if converted is not MATCH_ALL:
            params["filters"] = to_turbopuffer_filter(converted)
        return params

    def _to_document(self, row: "Row", content_key: str | None = None, score: float | None = None) -> Document:
        attributes = dict(row.model_extra or {})
        attributes.pop(DISTANCE_ATTRIBUTE, None)
        content = attributes.pop(content_key or self.content_key, None) or ""
        embedding = row.vector if isinstance(row.vector, list) else None
        return Document(id=str(row.id), content=content, metadata=attributes, embedding=embedding, score=score)

    @staticmethod
    def _distance(row: "Row") -> float:
        return float((row.model_extra or {}).get(DISTANCE_ATTRIBUTE, 0.0))

    @staticmethod
    def _top_k(top_k: int | None) -> int:
        return max(1, min(top_k or 10, MAX_TOP_K))

    def count_documents(self) -> int:
        """
        Count the documents in the namespace.

        Returns:
            int: The number of documents, 0 when the namespace does not exist.
        """
        try:
            response = self._namespace.query(aggregate_by={"count": ["Count"]})
        except NotFoundError:
            return 0
        return int((response.aggregations or {}).get("count", 0))

    def _iterate(
        self,
        filters: dict[str, Any] | None,
        include_embeddings: bool,
        content_key: str | None,
    ) -> Iterator[Document]:
        """Yield every matching document, paging through ids in order."""
        converted = self._filter(filters)
        if converted is MATCH_NONE:
            return

        last_id = None
        while True:
            cursor = [ID_ATTRIBUTE, "Gt", last_id] if last_id is not None else None
            params: dict[str, Any] = {"rank_by": [ID_ATTRIBUTE, "asc"], "top_k": MAX_TOP_K}
            params.update(self._attributes(include_embeddings))
            rows = self._query_rows(**self._with_filter(params, combine_and(converted, cursor)))
            for row in rows:
                yield self._to_document(row, content_key=content_key)
            if len(rows) < MAX_TOP_K:
                return
            last_id = rows[-1].id

    def filter_documents(self, filters: dict[str, Any] | None = None, content_key: str | None = None) -> list[Document]:
        """
        Return the documents that match the filters, with their embeddings.

        Args:
            filters (dict[str, Any] | None): The filters to apply. Without filters, every document is
                returned.
            content_key (Optional[str]): The attribute used to store content.

        Returns:
            list[Document]: The matching documents.
        """
        return list(self._iterate(filters, include_embeddings=True, content_key=content_key))

    def list_documents(self, include_embeddings: bool = False, content_key: str | None = None) -> list[Document]:
        """
        List every document in the namespace.

        Args:
            include_embeddings (bool): Whether to include document embeddings in the result.
            content_key (Optional[str]): The attribute used to store content.

        Returns:
            list[Document]: All documents in the namespace.
        """
        return list(self._iterate(None, include_embeddings=include_embeddings, content_key=content_key))

    def get_documents_by_id(
        self,
        ids: list[str],
        content_key: str | None = None,
        include_embeddings: bool = False,
    ) -> list[Document]:
        """
        Fetch documents by their exact ids (not a similarity search).

        Args:
            ids (list[str]): The document ids to fetch.
            content_key (Optional[str]): The attribute used to store content.
            include_embeddings (bool): Whether to include document embeddings in the result.

        Returns:
            list[Document]: The documents whose ids were found. Missing ids are skipped.
        """
        unique_ids = [str(i) for i in dict.fromkeys(ids or [])]
        documents = []
        for start in range(0, len(unique_ids), MAX_TOP_K):
            chunk = unique_ids[start : start + MAX_TOP_K]
            rows = self._query_rows(
                rank_by=[ID_ATTRIBUTE, "asc"],
                top_k=len(chunk),
                filters=[ID_ATTRIBUTE, "In", chunk],
                **self._attributes(include_embeddings),
            )
            documents.extend(self._to_document(row, content_key=content_key) for row in rows)
        return documents

    # Deleting

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the namespace.

        Args:
            document_ids (list[str], optional): The ids of the documents to delete.
            delete_all (bool): If True, delete the whole namespace.

        Raises:
            ValueError: If neither document_ids nor delete_all is provided.
        """
        if delete_all:
            self.delete_collection(self.index_name)
            return
        if not document_ids:
            raise ValueError("Either 'document_ids' or 'delete_all' must be set.")

        ids = [str(i) for i in dict.fromkeys(document_ids)]
        for start in range(0, len(ids), DELETE_BATCH_SIZE):
            try:
                self._namespace.write(deletes=ids[start : start + DELETE_BATCH_SIZE])
            except NotFoundError:
                return

    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Delete the documents that match the filters.

        Args:
            filters (dict[str, Any]): The filters selecting the documents to delete.

        Raises:
            ValueError: If no filters are provided.
        """
        if not filters:
            raise ValueError("No filters provided to delete documents.")

        # Deletes read the schema fresh, so a filter on a newly added attribute still matches.
        converted = convert_filters(filters, self._load_schema(refresh=True))
        if converted is MATCH_NONE:
            return

        try:
            self._namespace.write(delete_by_filter=to_turbopuffer_filter(converted))
        except NotFoundError:
            return

    def delete_collection(self, collection_name: str | None = None) -> None:
        """
        Delete a namespace with all of its documents. Deleting a namespace that does not exist succeeds.

        Args:
            collection_name (str | None): The namespace to delete. Defaults to this store's namespace.
        """
        name = collection_name or self.index_name
        try:
            self.client.namespace(name).delete_all()
        except NotFoundError:
            pass
        if name == self.index_name:
            self._schema = None

    # Retrieval

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exclude_document_embeddings: bool = True,
        content_key: str | None = None,
        min_score: float | None = None,
        max_distance: float | None = None,
    ) -> list[Document]:
        """
        Perform embedding-based retrieval on the documents.

        The score is the cosine similarity mapped to [0, 1] (1 - distance / 2), the same certainty
        Weaviate reports, so similarity thresholds carry over between the two stores.

        Args:
            query_embedding (list[float]): The query embedding.
            filters (dict[str, Any] | None): Filters to apply to the query.
            top_k (int | None): The number of top results to return.
            exclude_document_embeddings (bool): Whether to exclude document embeddings in the result.
            content_key (Optional[str]): The attribute used to store content.
            min_score (float | None): The minimum score a document needs.
            max_distance (float | None): The maximum cosine distance a document may have.

        Returns:
            list[Document]: The retrieved documents, most similar first.
        """
        converted = self._filter(filters)
        if converted is MATCH_NONE:
            return []

        params: dict[str, Any] = {
            "rank_by": [VECTOR_ATTRIBUTE, "ANN", list(query_embedding)],
            "top_k": self._top_k(top_k),
            **self._attributes(not exclude_document_embeddings),
        }
        documents = []
        for row in self._query_rows(**self._with_filter(params, converted)):
            distance = self._distance(row)
            if max_distance is not None and distance > max_distance:
                continue
            score = 1 - distance / 2
            if min_score is not None and score < min_score:
                continue
            documents.append(self._to_document(row, content_key=content_key, score=score))
        return documents

    def _bm25_rank_by(self, query: str, content_key: str | None) -> list | None:
        fields = self._full_text_fields(content_key)
        if not fields:
            return None
        query = query[:MAX_BM25_QUERY_LENGTH]
        if len(fields) == 1:
            return [fields[0], "BM25", query]
        return ["Sum", [[field, "BM25", query] for field in fields]]

    def _keyword_retrieval(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        content_key: str | None = None,
        exclude_document_embeddings: bool = True,
    ) -> list[Document]:
        """
        Perform BM25 retrieval over the content and the searchable metadata attributes.

        Args:
            query (str): The query string.
            filters (dict[str, Any] | None): Filters to apply to the query.
            top_k (int | None): The number of top results to return.
            content_key (Optional[str]): The attribute used to store content.
            exclude_document_embeddings (bool): Whether to exclude document embeddings in the result.

        Returns:
            list[Document]: The retrieved documents, highest BM25 score first. Empty for a blank query.
        """
        if not query or not query.strip():
            return []

        converted = self._filter(filters)
        if converted is MATCH_NONE:
            return []
        rank_by = self._bm25_rank_by(query, content_key)
        if rank_by is None:
            return []

        params: dict[str, Any] = {
            "rank_by": rank_by,
            "top_k": self._top_k(top_k),
            **self._attributes(not exclude_document_embeddings),
        }
        return [
            self._to_document(row, content_key=content_key, score=self._distance(row))
            for row in self._query_rows(**self._with_filter(params, converted))
        ]

    def _hybrid_retrieval(
        self,
        query_embedding: list[float],
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exclude_document_embeddings: bool = True,
        alpha: float | None = None,
        content_key: str | None = None,
        max_vector_distance: float | None = None,
    ) -> list[Document]:
        """
        Perform hybrid retrieval, combining vector and BM25 search.

        Both searches run in one request. Their scores are fused like Weaviate's relative score
        fusion: each result list is scaled to [0, 1], and the fused score is alpha times the vector
        score plus (1 - alpha) times the keyword score, so fused scores stay comparable to Weaviate's.

        Args:
            query_embedding (list[float]): The query embedding.
            query (str): The query string.
            filters (dict[str, Any] | None): Filters to apply to both searches.
            top_k (int | None): The number of top results to return.
            exclude_document_embeddings (bool): Whether to exclude document embeddings in the result.
            alpha (float | None): The weight of vector similarity, from 0.0 (keyword only) to 1.0
                (vector only). Defaults to the store's alpha.
            content_key (Optional[str]): The attribute used to store content.
            max_vector_distance (float | None): Vector results farther than this distance are dropped
                before fusion.

        Returns:
            list[Document]: The retrieved documents, highest fused score first.
        """
        alpha = self.alpha if alpha is None else alpha
        converted = self._filter(filters)
        if converted is MATCH_NONE:
            return []

        top_k = self._top_k(top_k)
        common = self._with_filter({"top_k": top_k, **self._attributes(not exclude_document_embeddings)}, converted)

        queries = [{"rank_by": [VECTOR_ATTRIBUTE, "ANN", list(query_embedding)], **common}]
        rank_by = self._bm25_rank_by(query, content_key) if query and query.strip() else None
        if rank_by is not None:
            queries.append({"rank_by": rank_by, **common})

        try:
            results = self._namespace.multi_query(queries=queries).results
        except NotFoundError:
            return []

        vector_rows = list(results[0].rows or [])
        if max_vector_distance is not None:
            vector_rows = [row for row in vector_rows if self._distance(row) <= max_vector_distance]
        keyword_rows = list(results[1].rows or []) if len(results) > 1 else []

        vector_scores = self._relative_scores(vector_rows, lower_is_better=True)
        keyword_scores = self._relative_scores(keyword_rows, lower_is_better=False)

        rows: dict[str, "Row"] = {}
        for row in [*vector_rows, *keyword_rows]:
            rows.setdefault(str(row.id), row)

        fused = {
            document_id: alpha * vector_scores.get(document_id, 0.0)
            + (1 - alpha) * keyword_scores.get(document_id, 0.0)
            for document_id in rows
        }
        ranked = sorted(rows, key=lambda document_id: fused[document_id], reverse=True)[:top_k]

        return [
            self._to_document(rows[document_id], content_key=content_key, score=fused[document_id])
            for document_id in ranked
        ]

    def _relative_scores(self, rows: list["Row"], lower_is_better: bool) -> dict[str, float]:
        """Scale a result list's scores to [0, 1], the best result scoring 1."""
        if not rows:
            return {}
        values = {str(row.id): self._distance(row) for row in rows}
        low, high = min(values.values()), max(values.values())
        if high == low:
            return {document_id: 1.0 for document_id in values}
        if lower_is_better:
            return {document_id: (high - value) / (high - low) for document_id, value in values.items()}
        return {document_id: (value - low) / (high - low) for document_id, value in values.items()}
