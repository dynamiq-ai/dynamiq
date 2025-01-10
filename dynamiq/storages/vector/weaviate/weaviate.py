import datetime
from typing import TYPE_CHECKING, Any, Optional

from weaviate.classes.query import HybridFusion
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateQueryError
from weaviate.util import generate_uuid5

from dynamiq.connections import Weaviate
from dynamiq.storages.vector.base import BaseVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreDuplicateDocumentException, VectorStoreException
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.utils import create_file_id_filter
from dynamiq.types import Document
from dynamiq.utils.logger import logger

from .filters import Filter, convert_filters

if TYPE_CHECKING:
    from weaviate import WeaviateClient
    from weaviate.collections.classes.data import DataObject

DOCUMENT_COLLECTION_PROPERTIES = [
    {"name": "_original_id", "dataType": ["text"]},
    {"name": "content", "dataType": ["text"]},
    {"name": "blob_data", "dataType": ["blob"]},
    {"name": "blob_mime_type", "dataType": ["text"]},
    {"name": "score", "dataType": ["number"]},
]

DEFAULT_QUERY_LIMIT = 9999


class WeaviteRetrieverVectorStoreParams(BaseVectorStoreParams):
    alpha: float = 0.5


class WeaviateVectorStore:
    """
    A Document Store for Weaviate.

    This class can be used with Weaviate Cloud Services or self-hosted instances.
    """

    def __init__(
        self,
        connection: Weaviate | None = None,
        client: Optional["WeaviateClient"] = None,
        index_name: str = "default",
        create_if_not_exist: bool = False,
        content_key: str = "content",
    ):
        """
        Initialize a new instance of WeaviateDocumentStore and connect to the Weaviate instance.

        Args:
            connection (Weaviate | None): A Weaviate connection object. If None, a new one is created.
            client (Optional[WeaviateClient]): A Weaviate client. If None, one is created from the connection.
            index_name (str): The name of the index to use. Defaults to "default".
            content_key (Optional[str]): The field used to store content in the storage.
        """
        self.client = client
        if self.client is None:
            if connection is None:
                connection = Weaviate()
            self.client = connection.connect()

        collection_settings = {
            "class": index_name,
            "invertedIndexConfig": {"indexNullState": True},
        }

        if not self.client.collections.exists(collection_settings["class"]):
            if create_if_not_exist:
                self.client.collections.create_from_dict(collection_settings)
            else:
                raise ValueError(
                    f"Collection '{collection_settings['class']}' does not exist."
                    " Set 'create_if_not_exist' to True to create it."
                )

        self._collection_settings = collection_settings
        self.content_key = content_key
        self._collection = self.client.collections.get(collection_settings["class"])

    def close(self):
        """Close the connection to Weaviate."""
        if self.client:
            self.client.close()

    def count_documents(self) -> int:
        """
        Count the number of documents in the DocumentStore.

        Returns:
            int: The number of documents in the store.
        """
        total = self._collection.aggregate.over_all(total_count=True).total_count
        return total if total else 0

    def _to_data_object(self, document: Document, content_key: str | None = None) -> dict[str, Any]:
        """
        Convert a Document to a Weaviate data object ready to be saved.

        Args:
            document (Document): The document to convert.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            dict[str, Any]: A dictionary representing the Weaviate data object.
        """
        data = document.to_dict()
        data[content_key or self.content_key] = data.pop("content", "")
        data["_original_id"] = data.pop("id")
        metadata = data.get("metadata", {})

        for key, val in metadata.items():
            data[key] = val

        del data["embedding"]
        del data["metadata"]

        return data

    def _to_document(
        self,
        data: "DataObject[dict[str, Any], None]",
        content_key: str | None = None,
    ) -> Document:
        """
        Convert a data object read from Weaviate into a Document.

        Args:
            data (DataObject[dict[str, Any], None]): The data object from Weaviate.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            Document: The converted Document object.
        """
        document_data = data.properties
        document_id = document_data.pop("_original_id")

        content = document_data.pop(content_key or self.content_key) or ""

        if isinstance(data.vector, list):
            document_data["embedding"] = data.vector
        elif isinstance(data.vector, dict):
            document_data["embedding"] = data.vector.get("default")
        else:
            document_data["embedding"] = None

        for key, value in document_data.items():
            if isinstance(value, datetime.datetime):
                document_data[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")

        if weaviate_meta := getattr(data, "metadata", None):
            if weaviate_meta.score is not None:
                document_data["score"] = weaviate_meta.score
            elif weaviate_meta.certainty is not None:
                document_data["score"] = weaviate_meta.certainty

        score = document_data.pop("score", None)
        embedding = document_data.pop("embedding", None)

        data = {
            "id": str(document_id),
            "content": content,
            "metadata": document_data,
            "score": score,
            "embedding": embedding,
        }

        logger.debug(f"Document loaded from Weaviate: {data}")

        return Document(**data)

    def _query(self) -> list[dict[str, Any]]:
        """
        Query all documents from Weaviate.

        Returns:
            list[dict[str, Any]]: A list of all documents in the store.

        Raises:
            VectorStoreException: If the query fails.
        """
        properties = [p.name for p in self._collection.config.get().properties]
        try:
            result = self._collection.iterator(
                include_vector=True, return_properties=properties
            )
        except WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise VectorStoreException(msg) from e
        return result

    def _query_with_filters(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Query documents from Weaviate with filters.

        Args:
            filters (dict[str, Any]): The filters to apply to the query.

        Returns:
            list[dict[str, Any]]: A list of documents matching the filters.

        Raises:
            VectorStoreException: If the query fails.
        """
        properties = [p.name for p in self._collection.config.get().properties]

        offset = 0
        partial_result = None
        result = []
        while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
            try:
                partial_result = self._collection.query.fetch_objects(
                    filters=convert_filters(filters),
                    include_vector=True,
                    limit=DEFAULT_QUERY_LIMIT,
                    offset=offset,
                    return_properties=properties,
                )
            except WeaviateQueryError as e:
                msg = f"Failed to query documents in Weaviate. Error: {e.message}"
                raise VectorStoreException(msg) from e
            result.extend(partial_result.objects)
            offset += DEFAULT_QUERY_LIMIT
        return result

    def filter_documents(self, filters: dict[str, Any] | None = None, content_key: str | None = None) -> list[Document]:
        """
        Filter documents based on the provided filters.

        Args:
            filters (dict[str, Any] | None): The filters to apply to the document list.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: A list of Documents that match the given filters.
        """
        if filters:
            result = self._query_with_filters(filters)
        else:
            result = self._query()
        return [self._to_document(doc, content_key=content_key) for doc in result]

    def list_documents(self, include_embeddings: bool = False, content_key: str | None = None) -> list[Document]:
        """
        List all documents in the DocumentStore.

        Args:
            include_embeddings (bool): Whether to include document embeddings in the result.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: A list of all documents in the store.
        """
        documents = []
        for item in self._collection.iterator(
            include_vector=include_embeddings
            # If using named vectors, you can specify ones to include e.g. ['title', 'body'], or True to include all
        ):
            document = self._to_document(item, content_key=content_key or self.content_key)
            documents.append(document)
        return documents

    def _batch_write(self, documents: list[Document], content_key: str | None = None) -> int:
        """
        Write documents to Weaviate in batches.

        Args:
            documents (list[Document]): The list of documents to write.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            int: The number of documents written.

        Raises:
            ValueError: If any of the input is not a Document.
            VectorStoreException: If the write operation fails.
        """
        with self.client.batch.dynamic() as batch:
            for doc in documents:
                if not isinstance(doc, Document):
                    msg = f"Expected a Document, got '{type(doc)}' instead."
                    raise ValueError(msg)

                batch.add_object(
                    properties=self._to_data_object(doc, content_key=content_key),
                    collection=self._collection.name,
                    uuid=generate_uuid5(doc.id),
                    vector=doc.embedding,
                )
        if failed_objects := self.client.batch.failed_objects:
            mapped_objects = {}
            for obj in failed_objects:
                properties = obj.object_.properties or {}
                id_ = properties.get("_original_id", obj.object_.uuid)
                mapped_objects[id_] = obj.data

            msg = "\n".join(
                [
                    f"Failed to write object with id '{id_}'. Error: '{message}'"
                    for id_, message in mapped_objects.items()
                ]
            )
            raise VectorStoreException(msg)

        return len(documents)

    def _write(self, documents: list[Document], policy: DuplicatePolicy, content_key: str | None = None) -> int:
        """
        Write documents to Weaviate using the specified policy.

        Args:
            documents (list[Document]): The list of documents to write.
            policy (DuplicatePolicy): The policy to use for handling duplicates.
            content_key (Optional[str]): The field used to store content in the storage.


        Returns:
            int: The number of documents written.

        Raises:
            ValueError: If any of the input is not a Document.
            VectorStoreDuplicateDocumentException: If duplicates are found with FAIL policy.
        """
        written = 0
        duplicate_errors_ids = []
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected a Document, got '{type(doc)}' instead."
                raise ValueError(msg)

            if policy == DuplicatePolicy.SKIP and self._collection.data.exists(uuid=generate_uuid5(doc.id)):
                continue

            try:
                self._collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=self._to_data_object(doc, content_key=content_key),
                    vector=doc.embedding,
                )

                written += 1
            except UnexpectedStatusCodeError:
                if policy == DuplicatePolicy.FAIL:
                    duplicate_errors_ids.append(doc.id)
        if duplicate_errors_ids:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise VectorStoreDuplicateDocumentException(msg)
        return written

    def write_documents(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE, content_key: str | None = None
    ) -> int:
        """
        Write documents to Weaviate using the specified policy.

        Args:
            documents (list[Document]): The list of documents to write.
            policy (DuplicatePolicy): The policy to use for handling duplicates.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            int: The number of documents written.
        """
        if policy in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            return self._batch_write(documents, content_key=content_key)

        return self._write(documents, policy)

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the DocumentStore.

        Args:
            document_ids (list[str], optional): The IDs of documents to delete.
            delete_all (bool): If True, delete all documents. Defaults to False.

        Raises:
            ValueError: If neither document_ids nor delete_all is provided.
        """
        if delete_all:
            weaviate_ids = [item.uuid for item in self._collection.iterator()]
        elif document_ids:
            weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        else:
            msg = "Either 'document_ids' or 'delete_all' must be set."
            raise ValueError(msg)
        self._collection.data.delete_many(
            where=Filter.by_id().contains_any(weaviate_ids)
        )

    def delete_documents_by_filters(self, filters: dict[str, Any]) -> None:
        """
        Delete documents from the DocumentStore based on the provided filters.

        Args:
            filters (dict[str, Any]): The filters to apply to the document list.
        """
        if filters:
            self._collection.data.delete_many(where=convert_filters(filters))
        else:
            raise ValueError("No filters provided to delete documents.")

    def delete_documents_by_file_id(self, file_id: str) -> None:
        """
        Delete documents from the DocumentStore based on the provided file_id.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def _keyword_retrieval(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """
        Perform BM25 retrieval on the documents.

        Args:
            query (str): The query string.
            filters (dict[str, Any] | None): Filters to apply to the query.
            top_k (int | None): The number of top results to return.

        Returns:
            list[Document]: A list of retrieved documents.
        """
        properties = [p.name for p in self._collection.config.get().properties]
        result = self._collection.query.bm25(
            query=query,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [self._to_document(doc) for doc in result.objects]

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exclude_document_embeddings=True,
        distance: float | None = None,
        certainty: float | None = None,
        content_key: str | None = None,
    ) -> list[Document]:
        """
        Perform embedding-based retrieval on the documents.

        Args:
            query_embedding (list[float]): The query embedding.
            filters (dict[str, Any] | None): Filters to apply to the query.
            top_k (int | None): The number of top results to return.
            exclude_document_embeddings (bool): Whether to exclude document embeddings in the result.
            distance (float | None): The maximum distance for retrieval.
            certainty (float | None): The minimum certainty for retrieval.
            content_key (Optional[str]): The field used to store content in the storage.

        Returns:
            list[Document]: A list of retrieved documents.

        Raises:
            ValueError: If both distance and certainty are provided.
        """
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        properties = [p.name for p in self._collection.config.get().properties]
        result = self._collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=not exclude_document_embeddings,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            return_properties=properties,
            return_metadata=["certainty"],
        )

        return [self._to_document(doc, content_key=content_key) for doc in result.objects]

    def _hybrid_retrieval(
        self,
        query_embedding: list[float],
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        exclude_document_embeddings=True,
        alpha: float = 0.5,
        fusion_type: HybridFusion = HybridFusion.RELATIVE_SCORE,
        content_key: str | None = None,
    ) -> list[Document]:
        """
        Perform hybrid retrieval on the documents.

        Args:
            query (str): The query string.
            filters (dict[str, Any] | None): Filters to apply to the query.
            top_k (int | None): The number of top results to return.

        Returns:
            list[Document]: A list of retrieved documents.
        """
        properties = [p.name for p in self._collection.config.get().properties]

        result = self._collection.query.hybrid(
            query=query,
            vector=query_embedding,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=not exclude_document_embeddings,
            query_properties=[content_key or self.content_key],
            return_properties=properties,
            return_metadata=["score"],
            alpha=alpha,
            fusion_type=fusion_type,
        )

        return [self._to_document(doc) for doc in result.objects]
