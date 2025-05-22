import time
import uuid
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.nodes.embedders.base import DocumentEmbedder, DocumentEmbedderInputSchema
from dynamiq.prompts import Message, MessageRole
from dynamiq.storages.vector.weaviate import WeaviateVectorStore, WeaviateWriterVectorStoreParams
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class WeaviateMemoryError(Exception):
    """Base exception class for Weaviate Memory Backend errors."""

    pass


class Weaviate(MemoryBackend):
    """
    Weaviate implementation of the memory storage backend.

    Uses WeaviateVectorStore to manage documents (messages) in a Weaviate collection.
    Leverages vector embeddings for semantic search and metadata for filtering
    and chronological ordering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Weaviate"
    connection: WeaviateConnection = Field(default_factory=WeaviateConnection)
    embedder: DocumentEmbedder
    collection_name: str = Field(default="conversations")
    tenant_name: str | None = Field(default=None)
    create_if_not_exist: bool = Field(default=True)
    content_property_name: str = Field(default="message_content")
    alpha: float = Field(default=0.5, description="Alpha for hybrid search (0=keyword, 1=vector)")

    _vector_store: WeaviateVectorStore | None = PrivateAttr(default=None)

    _ROLE_KEY: ClassVar[str] = "message_role"
    _TIMESTAMP_KEY: ClassVar[str] = "message_timestamp"
    _MESSAGE_ID_KEY: ClassVar[str] = "message_id"

    _CORE_MEMORY_PROPERTIES: ClassVar[list[str]] = [
        "message_role",
        "message_timestamp",
        "message_id",
        "user_id",
        "session_id",
        "_original_id",
    ]

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {
            "embedder": True,
            "_vector_store": True,
        }

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)

        data["embedder"] = self.embedder.to_dict(include_secure_params=include_secure_params, **kwargs)

        if "type" not in data:
            data["type"] = self.type

        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Weaviate vector store and ensure schema properties."""
        try:
            writer_params = WeaviateWriterVectorStoreParams(
                collection_name=self.collection_name,
                create_if_not_exist=self.create_if_not_exist,
                content_property_name=self.content_property_name,
                tenant_name=self.tenant_name,
            )

            properties_to_define = list(self._CORE_MEMORY_PROPERTIES)
            properties_to_define.append(self.content_property_name)

            self._vector_store = WeaviateVectorStore(
                connection=self.connection,
                **writer_params.model_dump(),
                alpha=self.alpha,
            )

            logger.debug(
                f"Weaviate backend '{self.name}' (ID: {self.id}) initialized "
                f"for collection '{self._vector_store._collection.name}'"
                f"{f' with tenant {self.tenant_name}' if self.tenant_name else ''}."
            )

            if self._vector_store and self.create_if_not_exist:
                properties_to_ensure = list(self._CORE_MEMORY_PROPERTIES)
                properties_to_ensure.append(self.content_property_name)
                self._vector_store.ensure_properties_exist(properties_to_ensure)

        except Exception as e:
            logger.error(f"Weaviate backend '{self.name}' failed to initialize vector store: {e}")
            raise WeaviateMemoryError(f"Failed to initialize Weaviate vector store: {e}") from e

    def _message_to_document(self, message: Message) -> Document:
        """Converts a Message object to a Document object for Weaviate."""
        if not self._vector_store:
            raise WeaviateMemoryError("Vector store not initialized.")

        message_id = message.metadata.get(self._MESSAGE_ID_KEY, str(uuid.uuid4()))
        timestamp = message.metadata.get("timestamp", time.time())

        doc_metadata = {
            self._ROLE_KEY: message.role.value,
            self._TIMESTAMP_KEY: timestamp,
            self._MESSAGE_ID_KEY: message_id,
            **(message.metadata or {}),
        }

        sanitized_metadata = {}
        for k, v in doc_metadata.items():
            if self._vector_store.is_valid_property_name(k):
                sanitized_metadata[k] = v
            else:
                logger.warning(f"Skipping invalid metadata key for Weaviate: '{k}'")

        doc_id = message_id

        return Document(
            id=doc_id,
            content=message.content,
            metadata=sanitized_metadata,
            embedding=None,
        )

    def _document_to_message(self, document: Document) -> Message:
        """Converts a Document object from Weaviate back to a Message object."""
        if not document.metadata:
            logger.warning(f"Document {document.id} from Weaviate has no metadata. Cannot reconstruct message fully.")
            return Message(role=MessageRole.SYSTEM, content=document.content, metadata={"retrieval_issue": True})

        metadata = dict(document.metadata)

        role_str = metadata.pop(self._ROLE_KEY, MessageRole.USER.value)
        try:
            role = MessageRole(role_str)
        except ValueError:
            logger.warning(f"Invalid role '{role_str}' found in document {document.id}. Defaulting to USER.")
            role = MessageRole.USER

        timestamp = metadata.get(self._TIMESTAMP_KEY)
        message_id = metadata.get(self._MESSAGE_ID_KEY)

        if document.score is not None:
            metadata["score"] = document.score

        metadata.pop(self._TIMESTAMP_KEY, None)
        metadata.pop(self._MESSAGE_ID_KEY, None)

        final_metadata = metadata
        if timestamp is not None:
            final_metadata["timestamp"] = timestamp
        if message_id is not None:
            final_metadata["message_id"] = message_id

        return Message(role=role, content=document.content or "", metadata=final_metadata)

    def add(self, message: Message) -> None:
        """Adds a message to the Weaviate memory."""
        if self._vector_store is None:
            raise WeaviateMemoryError("Weaviate vector store not initialized.")
        if self.embedder is None:
            raise WeaviateMemoryError("Embedder is required for Weaviate memory backend.")

        try:
            document = self._message_to_document(message)

            embedding_input = DocumentEmbedderInputSchema(
                documents=[Document(id=document.id, content=document.content)]
            )
            embedding_result = self.embedder.execute(input_data=embedding_input)

            if not embedding_result or not embedding_result.get("documents"):
                raise WeaviateMemoryError("Failed to generate embedding for the message.")

            document.embedding = embedding_result["documents"][0].embedding
            if not document.embedding:
                raise WeaviateMemoryError("Generated embedding is empty.")

            self._vector_store.write_documents([document], content_property_name=self.content_property_name)
            logger.debug(f"Weaviate Memory ({self.collection_name}): Added message {document.id}")

        except Exception as e:
            logger.error(f"Error adding message to Weaviate: {e}")
            raise WeaviateMemoryError(f"Error adding message to Weaviate: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """
        Retrieves messages from Weaviate, sorted chronologically (oldest first).

        Note: This fetches all documents and sorts client-side by timestamp.
              May be inefficient for very large collections.
              Returns the `limit` most recent messages if limit is specified.
        """
        if self._vector_store is None:
            raise WeaviateMemoryError("Weaviate vector store not initialized.")

        try:
            documents = self._vector_store.list_documents(
                include_embeddings=False, content_property_name=self.content_property_name
            )

            messages = [self._document_to_message(doc) for doc in documents]

            messages.sort(key=lambda msg: msg.metadata.get("timestamp", 0))

            if limit is not None and limit > 0:
                retrieved_messages = messages[-limit:]
            else:
                retrieved_messages = messages

            logger.debug(
                f"Weaviate Memory ({self.collection_name}): Retrieved {len(retrieved_messages)} messages"
                f"{f' (limited to {limit})' if limit else ''}."
            )
            return retrieved_messages

        except Exception as e:
            logger.error(f"Error retrieving messages from Weaviate: {e}")
            raise WeaviateMemoryError(f"Error retrieving messages from Weaviate: {e}") from e

    def _prepare_filters(self, filters: dict | None = None) -> dict | None:
        """
        Convert simple key-value filters to the Weaviate filter format if necessary.
        If the input `filters` already seem to be in Weaviate format (contain 'operator'
        or 'field'), they are passed through directly. Otherwise, assumes a simple
        dictionary where keys are fields and values are the values to match with '=='
        operator, combined with 'AND'.

        Args:
            filters: Raw filters dictionary.

        Returns:
            Prepared filters in Weaviate-compatible format, or None.
        """
        if not filters:
            return None

        if "operator" in filters and "conditions" in filters:
            logger.debug("Filters appear to be in Weaviate logical format, passing through.")
            return filters
        if "field" in filters and "operator" in filters and "value" in filters:
            logger.debug("Filters appear to be in Weaviate comparison format, passing through.")
            return filters

        logger.debug("Filters appear to be simple key-value, converting to Weaviate AND format.")
        conditions = []
        for key, value in filters.items():
            if self._vector_store and self._vector_store.is_valid_property_name(key):
                conditions.append({"field": key, "operator": "==", "value": value})
            else:
                logger.warning(f"Skipping filter key '{key}' as it's not a valid Weaviate property name.")

        if not conditions:
            return None
        return {"operator": "AND", "conditions": conditions}

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """
        Searches for messages in Weaviate using vector similarity and/or filters.

        Args:
            query: Optional search string for semantic search.
            filters: Optional dictionary for filtering messages by metadata.
                     This should be in the Weaviate filter format.
            limit: Maximum number of messages to return.

        Returns:
            List of matching messages sorted by relevance (if query provided)
            or potentially unsorted/timestamp-sorted (if only filters provided).
            Note: Sorting for filter-only results happens in the Memory class if needed.
        """
        if self._vector_store is None:
            raise WeaviateMemoryError("Weaviate vector store not initialized.")
        if query and self.embedder is None:
            raise WeaviateMemoryError("Embedder is required for search with query.")

        prepared_filters = self._prepare_filters(filters)

        try:
            effective_limit = limit if limit is not None else 10

            if query:
                embedding_input = DocumentEmbedderInputSchema(documents=[Document(id="query", content=query)])
                embedding_result = self.embedder.execute(input_data=embedding_input)
                query_embedding = embedding_result["documents"][0].embedding

                if not query_embedding:
                    raise WeaviateMemoryError("Failed to generate embedding for the search query.")

                documents = self._vector_store._hybrid_retrieval(
                    query_embedding=query_embedding,
                    query=query,
                    filters=prepared_filters,
                    top_k=effective_limit,
                    exclude_document_embeddings=True,
                    alpha=self.alpha,
                    content_property_name=self.content_property_name,
                )
                retrieved_messages = [self._document_to_message(doc) for doc in documents]

            elif prepared_filters:
                documents = self._vector_store.filter_documents(
                    filters=prepared_filters, content_property_name=self.content_property_name
                )
                retrieved_messages = [self._document_to_message(doc) for doc in documents]
                if effective_limit > 0:
                    retrieved_messages = retrieved_messages[:effective_limit]

            else:
                logger.debug(
                    f"Weaviate Memory ({self.collection_name}): Search called with no "
                    f"query or filters. Returning empty."
                )
                retrieved_messages = []

            logger.debug(
                f"Weaviate Memory ({self.collection_name}):"
                f" Found {len(retrieved_messages)} search results "
                f"(Query: {'Yes' if query else 'No'}, "
                f"Filters: {'Yes' if prepared_filters else 'No'}, Limit: {effective_limit})"
            )
            return retrieved_messages

        except Exception as e:
            if isinstance(e, WeaviateMemoryError) and "key missing" in str(e):
                logger.error(f"Filter format error during Weaviate search. Filters received: {prepared_filters}")
            logger.error(f"Error searching Weaviate memory: {e}")
            raise WeaviateMemoryError(f"Error searching Weaviate memory: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the Weaviate collection associated with this memory is empty."""
        if self._vector_store is None:
            raise WeaviateMemoryError("Weaviate vector store not initialized.")
        try:
            count = self._vector_store.count_documents()
            return count == 0
        except Exception as e:
            logger.error(f"Error checking if Weaviate memory is empty: {e}")
            raise WeaviateMemoryError(f"Error checking if Weaviate memory is empty: {e}") from e

    def clear(self) -> None:
        """Clears the Weaviate memory by deleting all documents in the collection/tenant."""
        if self._vector_store is None:
            raise WeaviateMemoryError("Weaviate vector store not initialized.")
        try:
            count = self._vector_store.count_documents()
            if count > 0:
                self._vector_store.delete_documents(delete_all=True)
                logger.info(
                    f"Weaviate Memory ({self.collection_name}): Cleared {count} documents "
                    f"{f'from tenant {self.tenant_name}' if self.tenant_name else 'from collection'}."
                )
            else:
                logger.info(f"Weaviate Memory ({self.collection_name}): Clear called, but memory was already empty.")
        except Exception as e:
            logger.error(f"Error clearing Weaviate memory: {e}")
            raise WeaviateMemoryError(f"Error clearing Weaviate memory: {e}") from e
