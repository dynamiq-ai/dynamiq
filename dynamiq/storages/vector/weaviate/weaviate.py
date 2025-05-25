import datetime
import re
from typing import TYPE_CHECKING, Any, Optional

from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import HybridFusion
from weaviate.classes.tenants import Tenant, TenantActivityStatus
from weaviate.exceptions import ObjectAlreadyExistsException, UnexpectedStatusCodeError, WeaviateQueryError
from weaviate.util import generate_uuid5

from dynamiq.connections import Weaviate
from dynamiq.storages.vector.base import BaseVectorStore, BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.dry_run import DryRunConfig, DryRunMode
from dynamiq.storages.vector.exceptions import VectorStoreDuplicateDocumentException, VectorStoreException
from dynamiq.storages.vector.policies import DuplicatePolicy
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


class WeaviateVectorStore(BaseVectorStore):
    """
    A Document Store for Weaviate.

    This class can be used with Weaviate Cloud Services or self-hosted instances.
    """

    PATTERN_COLLECTION_NAME = re.compile(r"^[A-Z][_0-9A-Za-z]*$")
    PATTERN_PROPERTY_NAME = re.compile(r"^[_A-Za-z][_0-9A-Za-z]*$")
    _PROPERTY_DATA_TYPES: dict[str, str] = {
        "content": DataType.TEXT,
        "message_content": DataType.TEXT,
        "message_role": DataType.TEXT,
        "message_timestamp": DataType.NUMBER,
        "message_id": DataType.TEXT,
        "user_id": DataType.TEXT,
        "session_id": DataType.TEXT,
    }

    def _get_property_type(self, property_name: str) -> str:
        """Gets the Weaviate data type for a known property, defaults to TEXT."""
        if property_name == self.content_key:
            return DataType.TEXT
        return self._PROPERTY_DATA_TYPES.get(property_name, DataType.TEXT)

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
        alpha: float = 0.5,
        dry_run_config: DryRunConfig | None = None,
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
            alpha (float): The alpha value used for hybrid retrieval operations. Controls
                the balance between keyword and vector search. Defaults to 0.5.
        """
        # Validate and normalize the index name
        index_name = self._fix_and_validate_index_name(index_name)
        collection_name = index_name

        self.client = client
        if self.client is None:
            if connection is None:
                connection = Weaviate()
            self.client = connection.connect()
        self._multi_tenancy_enabled = tenant_name is not None
        self.content_key = content_key
        self.alpha = alpha

        self.dry_run_config = dry_run_config
        self.original_index_name = index_name

        if dry_run_config:
            if dry_run_config.mode in [DryRunMode.TEMPORARY, DryRunMode.PERSISTENT, DryRunMode.INSPECTION]:
                collection_name = self._generate_test_collection_name(index_name, dry_run_config)

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

    def _generate_test_collection_name(self, original_name: str, config: DryRunConfig) -> str:
        """Generate a test collection name based on the dry run configuration.

        Args:
            original_name: The original collection name
            config: Dry run configuration

        Returns:
            str: Generated test collection name
        """
        if config.mode == DryRunMode.TEMPORARY:
            import uuid

            return f"{original_name}_dryrun_{uuid.uuid4().hex[:8]}"
        else:
            return f"{original_name}_{config.test_collection_suffix}"

    def _create_collection(
        self, collection_name: str, tenant_name: str | None, properties_to_define: list[str] | None = None
    ):
        """
        Create a new Weaviate collection with appropriate configuration and properties.

        Args:
            collection_name: Name of the collection to create
            tenant_name: Optional tenant name to enable multi-tenancy
            properties_to_define: List of property names to explicitly define in the schema.
        """
        collection_config_params = {
            "name": collection_name,
            "inverted_index_config": Configure.inverted_index(index_null_state=True),
            "vector_index_config": Configure.VectorIndex.hnsw(),
        }

        if tenant_name is not None:
            collection_config_params["multi_tenancy_config"] = Configure.multi_tenancy(enabled=True)

        properties = []
        all_props_to_define = set(properties_to_define or [])
        all_props_to_define.add(self.content_key)
        all_props_to_define.add("_original_id")

        for prop_name in all_props_to_define:
            if self.is_valid_property_name(prop_name):
                prop_type = self._get_property_type(prop_name)
                properties.append(Property(name=prop_name, data_type=prop_type))
                logger.debug(f"Prepared property definition: {prop_name} (Type: {prop_type})")
            else:
                logger.warning(f"Skipping definition for invalid property name: '{prop_name}'")

        if properties:
            collection_config_params["properties"] = properties
        try:
            self.client.collections.create(**collection_config_params)
            logger.info(f"Created collection '{collection_name}' with defined properties.")
        except ObjectAlreadyExistsException:
            logger.warning(f"Collection '{collection_name}' already exists. Skipping creation.")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise

    def ensure_properties_exist(self, properties_to_ensure: list[str]):
        """
        Checks if properties exist in the schema and adds them if they don't.

        Args:
            properties_to_ensure: A list of property names to check/add.
        """
        if not properties_to_ensure:
            return

        try:
            collection_config = self._collection.config.get()
            existing_properties = {prop.name for prop in collection_config.properties}

            for prop_name in properties_to_ensure:
                if prop_name not in existing_properties and self.is_valid_property_name(prop_name):
                    prop_type = self._get_property_type(prop_name)
                    try:
                        self._collection.config.add_property(Property(name=prop_name, data_type=prop_type))
                        logger.info(
                            f"Added missing property '{prop_name}' "
                            f"(Type: {prop_type}) to collection "
                            f"'{self._collection.name}' schema."
                        )
                    except Exception as add_err:
                        logger.error(f"Failed to add property '{prop_name}' to schema: {add_err}")
                elif prop_name in existing_properties:
                    logger.debug(f"Property '{prop_name}' already exists in schema.")
                elif not self.is_valid_property_name(prop_name):
                    logger.warning(f"Cannot ensure invalid property name: '{prop_name}'")

        except Exception as e:
            logger.error(f"Failed to check or update schema properties for collection '{self._collection.name}': {e}")

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

            self._multi_tenancy_enabled = actual_multi_tenancy_enabled

        except Exception as e:
            logger.warning(f"Failed to retrieve multi-tenancy configuration: {str(e)}")

    def _setup_collection(self, base_collection, collection_name, tenant_name):
        """
        Set up the collection with or without tenant context.

        Args:
            base_collection: The base Weaviate collection
            collection_name: Name of the collection
            tenant_name: Optional tenant name to use
        """
        if not tenant_name:
            self._collection = base_collection
            self._tenant_name = None
            return

        if not self._multi_tenancy_enabled:
            raise ValueError(
                f"Collection '{collection_name}' has multi-tenancy disabled, "
                f"but tenant_name '{tenant_name}' was provided. "
                f"To use a tenant, create the collection with a tenant_name parameter."
            )

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
            tenant = None
            try:
                tenant = collection.tenants.get_by_name(tenant_name)
            except AttributeError:
                tenants = collection.tenants.get()

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

        for key, val in metadata.items():
            if not self.is_valid_property_name(key):
                raise ValueError(
                    f"Invalid property name: '{key}'. Property names must match the pattern: [_A-Za-z][_0-9A-Za-z]*"
                )
            data[key] = val

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
            try:
                return self._collection.tenants.get_by_name(tenant_name)
            except AttributeError:
                tenants = self.list_tenants()

                for tenant in tenants:
                    if isinstance(tenant, dict) and tenant.get("name") == tenant_name:
                        return tenant
                    elif hasattr(tenant, "name") and tenant.name == tenant_name:
                        return tenant

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

                batch_params = {
                    "properties": self._to_data_object(doc, content_key=content_key),
                    "collection": self._collection.name,
                    "uuid": generate_uuid5(doc.id),
                    "vector": doc.embedding,
                }

                if self._multi_tenancy_enabled and self._tenant_name:
                    batch_params["tenant"] = self._tenant_name

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
                insert_params = {
                    "uuid": generate_uuid5(doc.id),
                    "properties": self._to_data_object(doc, content_key=content_key),
                    "vector": doc.embedding,
                }

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
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        content_key: str | None = None,
        dry_run_config: DryRunConfig | None = None,
    ) -> int:
        """
        Write documents to Weaviate using the specified policy.

        Args:
            documents (list[Document]): The list of documents to write.
            policy (DuplicatePolicy): The policy to use for handling duplicates.
            content_key (Optional[str]): The field used to store content in the storage.
            dry_run_config (Optional[DryRunConfig]): Override dry run configuration.

        Returns:
            int: The number of documents written.
        """
        active_dry_run = dry_run_config or self.dry_run_config

        if active_dry_run:

            if active_dry_run.mode == DryRunMode.WORKFLOW_ONLY:
                logger.info(f"WORKFLOW_ONLY mode: Simulating write of {len(documents)} documents")
                return len(documents)

        if active_dry_run:
            documents = self._prepare_dry_run_documents(documents, active_dry_run)

        if policy in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            return self._batch_write(documents, content_key=content_key)

        return self._write(documents, policy, content_key=content_key)

    def _prepare_dry_run_documents(self, documents: list[Document], config: DryRunConfig) -> list[Document]:
        """Prepare documents for dry run by adding ID prefixes and metadata.

        Args:
            documents: Documents to prepare
            config: Dry run configuration

        Returns:
            list[Document]: Prepared documents with dry run modifications
        """
        prepared_docs = []

        for doc in documents:
            prepared_doc = Document(
                id=f"{config.document_id_prefix}{doc.id}",
                content=doc.content,
                metadata=doc.metadata.copy() if doc.metadata else {},
                embedding=doc.embedding,
                score=doc.score,
            )

            if prepared_doc.metadata is None:
                prepared_doc.metadata = {}

            prepared_doc.metadata.update(
                {
                    "_dry_run": True,
                    "_dry_run_mode": config.mode,
                    "_original_id": doc.id,
                    "_dry_run_timestamp": datetime.datetime.now().isoformat(),
                }
            )

            prepared_docs.append(prepared_doc)

        logger.debug(f"Prepared {len(prepared_docs)} documents with dry run metadata")
        return prepared_docs

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

        query_alpha = self.alpha if alpha is None else alpha
        result = self._collection.query.hybrid(
            query=query,
            vector=query_embedding,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=not exclude_document_embeddings,
            query_properties=[content_key or self.content_key],
            return_properties=properties,
            return_metadata=["score"],
            alpha=query_alpha,
            fusion_type=fusion_type,
        )

        return [self._to_document(doc) for doc in result.objects]

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            collection_name = self._collection.name if hasattr(self, "_collection") else self.original_index_name
            return self.client.collections.exists(collection_name)
        except Exception as e:
            logger.warning(f"Failed to check collection existence: {str(e)}")
            return False

    def create_collection(self) -> bool:
        """Create the collection if it doesn't exist using existing infrastructure."""
        try:
            collection_name = self._collection.name if hasattr(self, "_collection") else self.original_index_name

            if self.collection_exists():
                logger.debug(f"Collection {collection_name} already exists")
                return True

            tenant_name = self._tenant_name if hasattr(self, "_tenant_name") else None
            self._create_collection(collection_name, tenant_name)
            return True

        except Exception as e:
            collection_name = self._collection.name if hasattr(self, "_collection") else self.original_index_name
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            return False

    def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            collection_name = self._collection.name if hasattr(self, "_collection") else self.original_index_name

            if not self.collection_exists():
                logger.debug(f"Collection {collection_name} doesn't exist, nothing to delete")
                return True

            self.client.collections.delete(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            collection_name = self._collection.name if hasattr(self, "_collection") else self.original_index_name
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            return False

    def health_check(self) -> dict[str, Any]:
        """Perform a health check on the vector store."""
        health_status = {"healthy": False, "collection_exists": False, "client_ready": False, "error": None}

        try:
            health_status["client_ready"] = True  # Assume ready if client exists
            health_status["collection_exists"] = self.collection_exists()
            health_status["healthy"] = health_status["client_ready"] and health_status["collection_exists"]

        except Exception as e:
            health_status["error"] = str(e)
            logger.warning(f"Health check failed: {str(e)}")

        return health_status
