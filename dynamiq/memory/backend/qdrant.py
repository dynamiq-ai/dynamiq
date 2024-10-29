import uuid

from qdrant_client.http.exceptions import UnexpectedResponse

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.base import MemoryBackend
from dynamiq.prompts import Message
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.qdrant import QdrantVectorStore
from dynamiq.types import Document


class QdrantError(Exception):
    """Base exception for Qdrant-related errors."""

    pass


class Qdrant(MemoryBackend):
    """Qdrant implementation of the memory storage backend."""

    name = "Qdrant"

    def __init__(
        self,
        connection: QdrantConnection,
        embedder: BaseEmbedder,
        index_name: str = "conversations",
        metric: str = "cosine",
        on_disk: bool = False,
        create_if_not_exist: bool = True,
    ):
        """Initializes the Qdrant memory storage.

        Args:
            connection: QdrantConnection instance for connecting to Qdrant
            embedder: Embedder instance for creating embeddings
            index_name: Name of the collection to store messages
        """
        self.connection = connection
        self.index_name = index_name
        self.embedder = embedder

        try:
            self.vector_store = QdrantVectorStore(
                connection=connection,
                index_name=index_name,
                dimension=embedder.dimensions,
                create_if_not_exist=create_if_not_exist,
                metric=metric,
                on_disk=on_disk,
                recreate_index=False,
            )
            self.client = self.vector_store._client
            if not self.client:
                raise QdrantError("Failed to initialize Qdrant client")
        except Exception as e:
            raise QdrantError(f"Failed to connect to Qdrant: {e}") from e

    def _message_to_document(self, message: Message) -> Document:
        """Converts a Message object to a Document object."""
        return Document(
            id=str(uuid.uuid4()),
            content=message.content,
            metadata={"role": message.role.value, **(message.metadata or {})},
            embedding=None,  # Will be populated during write
        )

    def _document_to_message(self, document: Document) -> Message:
        """Converts a Document object to a Message object."""
        metadata = dict(document.metadata)
        role = metadata.pop("role")
        return Message(content=document.content, role=role, metadata=metadata, score=document.score)

    def add(self, message: Message):
        """Stores a message in Qdrant.

        Args:
            message: Message to store

        Raises:
            QdrantError: If failed to add message
        """
        try:
            document = self._message_to_document(message)
            embedding_result = self.embedder.embed_text(document.content)
            document.embedding = embedding_result["embedding"]

            self.vector_store.write_documents(
                documents=[document], policy=DuplicatePolicy.SKIP  # Changed from OVERWRITE to prevent recreation
            )
        except Exception as e:
            raise QdrantError(f"Failed to add message to Qdrant: {e}") from e

    def get_all(self, limit: int = None) -> list[Message]:
        """Retrieves all messages from Qdrant.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages sorted by timestamp
        """
        try:
            documents = self.vector_store.list_documents(include_embeddings=False)
            messages = [self._document_to_message(doc) for doc in documents]
            return sorted(messages, key=lambda msg: msg.metadata.get("timestamp", 0))
        except Exception as e:
            raise QdrantError(f"Failed to retrieve messages from Qdrant: {e}") from e

    def search(self, query: str = None, limit: int = 10, filters: dict = None) -> list[Message]:
        """Searches for messages in Qdrant.

        Args:
            query: Text query to search for
            limit: Maximum number of results to return
            filters: Metadata filters to apply

        Returns:
            List of matching messages
        """
        try:
            qdrant_filters = self._prepare_filters(filters)
            if query:
                embedding_result = self.embedder.embed_text(query)
                documents = self.vector_store._query_by_embedding(
                    query_embedding=embedding_result["embedding"],
                    filters=qdrant_filters,
                    top_k=limit,
                    return_embedding=False,
                )
            elif filters:
                documents = self.vector_store.filter_documents(filters=qdrant_filters)
                if limit:
                    documents = documents[:limit]
            else:
                return []

            return [self._document_to_message(doc) for doc in documents]
        except Exception as e:
            raise QdrantError(f"Error searching in Qdrant: {e}") from e

    def _prepare_filters(self, filters: dict | None = None) -> dict | None:
        """Prepares simple filters for Qdrant vector store format."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                condition = {"operator": "==", "field": key, "value": value}
            elif isinstance(value, list):
                condition = {"operator": "in", "field": key, "value": value}
            elif isinstance(value, dict) and any(k in value for k in ["gte", "lte", "gt", "lt"]):
                condition = {"operator": "range", "field": key, **value}
            else:
                raise QdrantError(f"Unsupported filter value type for key '{key}': {type(value)}")

            conditions.append(condition)
        return {"operator": "AND", "conditions": conditions} if conditions else None

    def is_empty(self) -> bool:
        """Checks if the Qdrant collection is empty."""
        try:
            return self.vector_store.count_documents() == 0
        except UnexpectedResponse as e:
            if e.status_code == 404:  # Collection doesn't exist
                return True
            raise QdrantError(f"Failed to check if Qdrant collection is empty: {e}") from e

    def clear(self):
        """Clears the Qdrant collection."""
        try:
            self.vector_store.delete_documents(delete_all=True)
        except Exception as e:
            raise QdrantError(f"Failed to clear Qdrant collection: {e}") from e
