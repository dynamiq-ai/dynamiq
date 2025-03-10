import datetime
import re
from typing import TYPE_CHECKING, Any, Optional

from weaviate.classes.config import Configure
from weaviate.classes.query import HybridFusion
from weaviate.classes.tenants import Tenant, TenantActivityStatus
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateQueryError
from weaviate.util import generate_uuid5

from dynamiq.connections import Weaviate
from dynamiq.storages.vector.base import BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreDuplicateDocumentException, VectorStoreException
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.utils import create_file_id_filter
from dynamiq.types import Document
from dynamiq.utils.logger import logger

from .filters import Filter, convert_filters

if TYPE_CHECKING:
    from weaviate import WeaviateClient
    from weaviate.collections.classes.data import DataObject


DEFAULT_QUERY_LIMIT = 9999


class WeaviateWriterVectorStoreParams(BaseWriterVectorStoreParams):
    """Parameters for creating and managing Weaviate collections with multi-tenancy."""
    tenant_name: str | None = None


class WeaviateRetrieverVectorStoreParams(BaseVectorStoreParams):
    """Parameters for using existing Weaviate collections with tenant context."""
    alpha: float = 0.5
    tenant_name: str | None = None


class WeaviateVectorStore:
    """
    A Document Store for Weaviate.

    This class can be used with Weaviate Cloud Services or self-hosted instances.
    """

    PATTERN_COLLECTION_NAME = re.compile(r"^[A-Z][_0-9A-Za-z]*$")
    PATTERN_PROPERTY_NAME = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")

    @staticmethod
    def is_valid_collection_name(name: str) -> bool:
        return bool(WeaviateVectorStore.PATTERN_COLLECTION_NAME.fullmatch(name))

    @staticmethod
    def is_valid_property_name(name: str) -> bool:
        """
        Check if a property name is valid according to Weaviate naming rules.

        Args:
            name (str): The property name to check

        Returns:
            bool: True if the name is valid, False otherwise
        """
        return bool(WeaviateVectorStore.PATTERN_PROPERTY_NAME.fullmatch(name))

    @classmethod
    def _fix_and_validate_index_name(cls, index_name: str) -> str:
        """
        Fix the index name if it starts with a lowercase letter and then validate it.
        Logs a warning if the index name is corrected.
        """
        if index_name and index_name[0].islower():
            fixed_name = index_name[0].upper() + index_name[1:]
            logger.warning(
                f"Index name '{index_name}' starts with a lowercase letter. "
                f"Automatically updating it to '{fixed_name}'."
            )
            index_name = fixed_name
        if not cls.is_valid_collection_name(index_name):
            msg = (
                f"Collection name '{index_name}' is invalid. It must match the pattern "
                f"{cls.PATTERN_COLLECTION_NAME.pattern}"
            )
            raise ValueError(msg)
        return index_name

    def __init__(
        self,
        connection: Weaviate | None = None,
        client: Optional["WeaviateClient"] = None,
        index_name: str = "default",
        create_if_not_exist: bool = False,
        content_key: str = "content",
        tenant_name: str | None = None,
    ):
        """
        Initialize a new instance of WeaviateDocumentStore and connect to the
        Weaviate instance.

        Args:
            connection (Weaviate | None): A Weaviate connection object. If None, a
                new one is created.
            client (Optional[WeaviateClient]): A Weaviate client. If None, one is
                created from the connection.
            index_name (str): The name of the index to use. Defaults to "default".
            content_key (Optional[str]): The field used to store content in the
                storage.
            tenant_name (str | None): The name of the tenant to use for all operations.
                If provided, multi-tenancy will be enabled for the collection.
        """
        # Validate and normalize the index name
        index_name = self._fix_and_validate_index_name(index_name)
        collection_name = index_name

        # Initialize client
        self.client = client
        if self.client is None:
            if connection is None:
                connection = Weaviate()
            self.client = connection.connect()

        # Store multi-tenancy configuration
        self._multi_tenancy_enabled = tenant_name is not None
        self.content_key = content_key

        # Create collection if needed or validate existing collection
        if not self.client.collections.exists(collection_name):
            if create_if_not_exist:
                self._create_collection(collection_name, tenant_name)
            else:
                raise ValueError(
                    f"Collection '{collection_name}' does not exist. Set 'create_if_not_exist' to True to create it."
                )

        # Get the base collection
        base_collection = self.client.collections.get(collection_name)

        # Get the actual multi-tenancy configuration from the collection
        self._update_multi_tenancy_status(base_collection)

        # Set up the collection - either with tenant context or without
        self._setup_collection(base_collection, collection_name, tenant_name)

    def _create_collection(self, collection_name: str, tenant_name: str | None):
        """
        Create a new Weaviate collection with appropriate configuration.

        Args:
            collection_name: Name of the collection to create
            tenant_name: Optional tenant name to enable multi-tenancy
        """

        # Set up basic collection configuration
        collection_config = {"inverted_index_config": Configure.inverted_index(index_null_state=True)}

        # Add multi-tenancy configuration if tenant_name is provided
        if tenant_name is not None:
            mt_config = {"enabled": True}

            collection_config["multi_tenancy_config"] = Configure.multi_tenancy(**mt_config)

        # Create the collection
        self.client.collections.create(name=collection_name, **collection_config)
        logger.info(f"Created collection '{collection_name}'")

    def _update_multi_tenancy_status(self, collection):
        """
        Update the multi-tenancy status based on the actual collection configuration.

        Args:
            collection: The Weaviate collection
        """
        try:
            collection_config = collection.config.get()
            actual_multi_tenancy_enabled = False

            if hasattr(collection_config, "multi_tenancy_config") and collection_config.multi_tenancy_config:
                actual_multi_tenancy_enabled = collection_config.multi_tenancy_config.enabled

            # Update instance variable to reflect actual configuration
            self._multi_tenancy_enabled = actual_multi_tenancy_enabled

        except Exception as e:
            logger.warning(f"Failed to retrieve multi-tenancy configuration: {str(e)}")
            # Keep the inferred multi-tenancy setting as fallback

    def _setup_collection(self, base_collection, collection_name, tenant_name):
        """
        Set up the collection with or without tenant context.

        Args:
            base_collection: The base Weaviate collection
            collection_name: Name of the collection
            tenant_name: Optional tenant name to use
        """
        # No tenant specified - use the base collection
        if not tenant_name:
            self._collection = base_collection
            self._tenant_name = None
            return

        # Tenant specified but multi-tenancy is disabled in the collection
        if not self._multi_tenancy_enabled:
            raise ValueError(
                f"Collection '{collection_name}' has multi-tenancy disabled, "
                f"but tenant_name '{tenant_name}' was provided. "
                f"To use a tenant, create the collection with a tenant_name parameter."
            )

        # Tenant specified and multi-tenancy is enabled
        self._ensure_tenant_exists(base_collection, tenant_name)
        self._collection = base_collection.with_tenant(tenant_name)
        self._tenant_name = tenant_name

    def _ensure_tenant_exists(self, collection, tenant_name):
        """
        Check if the tenant exists and create it if it doesn't.

        Args:
            collection: The Weaviate collection
            tenant_name: Name of the tenant to check/create
        """
        try:
            # Use get_by_name method from Weaviate client if available
            tenant = None
            try:
                # Modern Weaviate client approach
                tenant = collection.tenants.get_by_name(tenant_name)
            except AttributeError:
                # Fallback for older clients - search in the list
                tenants = collection.tenants.get()

                # Handle different response formats
                if isinstance(tenants, dict) and tenant_name in tenants:
                    tenant = tenants[tenant_name]
                elif isinstance(tenants, list):
                    for t in tenants:
                        if (
                            (isinstance(t, dict) and t.get("name") == tenant_name)
                            or (hasattr(t, "name") and t.name == tenant_name)
                            or (str(t) == tenant_name)
                        ):
                            tenant = t
                            break

            # Create tenant if it doesn't exist
            if not tenant:
                logger.info(f"Creating new tenant '{tenant_name}' in collection")
                collection.tenants.create(tenants=[Tenant(name=tenant_name)])
                logger.info(f"Successfully created tenant '{tenant_name}'")
            else:
                logger.info(f"Tenant '{tenant_name}' already exists, no need to create")

        except Exception as e:
            logger.warning(
                f"Error while checking/creating tenant '{tenant_name}': {str(e)}\n"
                f"Error type: {type(e).__name__}\n"
                f"Will continue with the assumption that the tenant exists."
            )

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

        Raises:
            ValueError: If any property name is invalid according to Weaviate naming rules
        """
        data = document.to_dict()
        data[content_key or self.content_key] = data.pop("content", "")
        data["_original_id"] = data.pop("id")
        metadata = data.get("metadata", {})

        # Validate and add metadata properties
        for key, val in metadata.items():
            if not self.is_valid_property_name(key):
                raise ValueError(
                    f"Invalid property name: '{key}'. Property names must match the pattern: [_A-Za-z][_0-9A-Za-z]*"
                )
            data[key] = val

        # Ensure all property names in the data object are valid
        invalid_props = []
        for key in data:
            if key not in ["_original_id", "embedding", "metadata"] and not self.is_valid_property_name(key):
                invalid_props.append(key)

        if invalid_props:
            raise ValueError(
                f"Invalid property names: {invalid_props}. "
                "Property names must match the pattern: [_A-Za-z][_0-9A-Za-z]*"
            )

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

    def add_tenants(self, tenant_names: list[str]) -> None:
        """
        Add new tenants to the collection.

        Args:
            tenant_names (list[str]): List of tenant names to add.
        """
        if not self._multi_tenancy_enabled:
            raise ValueError("Multi-tenancy is not enabled for this collection")

        tenants = [Tenant(name=name) for name in tenant_names]
        self._collection.tenants.create(tenants=tenants)

    def remove_tenants(self, tenant_names: list[str]) -> None:
        """
        Remove tenants from the collection.

        Args:
            tenant_names (list[str]): List of tenant names to remove.
        """
        if not self._multi_tenancy_enabled:
            raise ValueError("Multi-tenancy is not enabled for this collection")

        self._collection.tenants.remove(tenant_names)

    def list_tenants(self) -> list[dict[str, Any]]:
        """
        List all tenants in the collection.

        Returns:
            list[dict[str, Any]]: List of tenant information.
        """
        if not self._multi_tenancy_enabled:
            raise ValueError("Multi-tenancy is not enabled for this collection")

        return self._collection.tenants.get()

    def get_tenant(self, tenant_name: str) -> dict[str, Any] | None:
        """
        Get information about a specific tenant.

        Args:
            tenant_name (str): Name of the tenant to get.

        Returns:
            dict[str, Any] | None: Tenant information or None if tenant doesn't exist.
        """
        if not self._multi_tenancy_enabled:
            raise ValueError("Multi-tenancy is not enabled for this collection")

        try:
            # Try using direct tenant lookup if available in the client version
            try:
                return self._collection.tenants.get_by_name(tenant_name)
            except AttributeError:
                # Fallback for older client versions
                tenants = self.list_tenants()

                # Search through the list of tenants
                for tenant in tenants:
                    if isinstance(tenant, dict) and tenant.get("name") == tenant_name:
                        return tenant
                    elif hasattr(tenant, "name") and tenant.name == tenant_name:
                        return tenant

                # Not found
                return None
        except Exception as e:
            logger.warning(f"Error getting tenant '{tenant_name}': {str(e)}")
            return None

    def update_tenant_status(self, tenant_name: str, status: TenantActivityStatus) -> None:
        """
        Update the activity status of a tenant.

        Args:
            tenant_name (str): Name of the tenant to update.
            status (TenantActivityStatus): New activity status (ACTIVE, INACTIVE, or OFFLOADED).

        Raises:
            ValueError: If multi-tenancy is not enabled or the tenant doesn't exist.
        """
        if not self._multi_tenancy_enabled:
            raise ValueError("Multi-tenancy is not enabled for this collection")

        # Check if tenant exists
        tenant = self.get_tenant(tenant_name)
        if not tenant:
            raise ValueError(f"Tenant '{tenant_name}' does not exist")

        try:
            self._collection.tenants.update(tenants=[Tenant(name=tenant_name, activity_status=status)])
            logger.info(f"Updated tenant '{tenant_name}' status to {status}")
        except Exception as e:
            logger.error(f"Failed to update tenant '{tenant_name}' status: {str(e)}")
            raise

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
        for item in self._collection.iterator(include_vector=include_embeddings):
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

                # Create the batch add parameters
                batch_params = {
                    "properties": self._to_data_object(doc, content_key=content_key),
                    "collection": self._collection.name,
                    "uuid": generate_uuid5(doc.id),
                    "vector": doc.embedding,
                }

                # Add tenant parameter if multi-tenancy is enabled and tenant is specified
                if self._multi_tenancy_enabled and self._tenant_name:
                    batch_params["tenant"] = self._tenant_name

                # Add the object with the appropriate parameters
                batch.add_object(**batch_params)

        if failed_objects := self.client.batch.failed_objects:
            mapped_objects = {}
            for obj in failed_objects:
                properties = obj.object_.properties or {}
                id_ = properties.get("_original_id", obj.object_.uuid)
                mapped_objects[id_] = obj.message if hasattr(obj, "message") else str(obj)

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
                # Create the insert parameters
                insert_params = {
                    "uuid": generate_uuid5(doc.id),
                    "properties": self._to_data_object(doc, content_key=content_key),
                    "vector": doc.embedding,
                }

                # Add tenant parameter if multi-tenancy is enabled and tenant is specified
                # Note: This shouldn't be necessary when using self._collection with tenant context,
                # but added for consistency with _batch_write
                if self._multi_tenancy_enabled and self._tenant_name:
                    insert_params["tenant"] = self._tenant_name

                self._collection.data.insert(**insert_params)
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

        return self._write(documents, policy, content_key=content_key)

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
