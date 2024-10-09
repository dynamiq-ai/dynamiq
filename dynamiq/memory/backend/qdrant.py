import uuid

from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.base import MemoryBackend
from dynamiq.prompts import Message


class QdrantError(Exception):
    """Base exception for Qdrant-related errors."""

    pass


class Qdrant(MemoryBackend):
    """Qdrant implementation of the memory storage backend."""

    name = "Qdrant"

    def __init__(self, connection: QdrantConnection, embedder: BaseEmbedder, index_name: str = "conversations"):
        """Initializes the Qdrant memory storage."""
        self.connection = connection
        self.index_name = index_name
        self.embedder = embedder

        try:
            self.client = self.connection.connect()
        except Exception as e:
            raise QdrantError(f"Failed to connect to Qdrant: {e}") from e

        # Check if the collection exists, create it if not
        if not self.client.collection_exists(collection_name=self.index_name):
            self._create_collection()

    def _create_collection(self):
        """Creates the Qdrant collection."""
        try:
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedder.dimensions, distance=qdrant_models.Distance.COSINE
                ),
            )
        except ApiException as e:
            raise QdrantError(f"Failed to create collection '{self.index_name}': {e}") from e

    def add(self, message: Message):
        """Stores a message in Qdrant."""
        try:
            embedding_result = self.embedder.embed_text(message.content)
            embedding = embedding_result["embedding"]
            message_id = str(uuid.uuid4())
            self.client.upsert(
                collection_name=self.index_name,
                points=[
                    qdrant_models.PointStruct(
                        id=message_id,
                        vector=embedding,
                        payload=message.model_dump(),
                    )
                ],
            )
        except Exception as e:
            raise QdrantError(f"Failed to add message to Qdrant: {e}") from e

    def get_all(self, limit: int = None) -> list[Message]:
        """Retrieves all messages from Qdrant."""
        try:
            search_result = self.client.scroll(
                collection_name=self.index_name,
                scroll_filter=None,
                with_payload=True,
                limit=limit,
            )
            messages = [Message(**point.payload) for point in search_result[0]]
            return sorted(messages, key=lambda msg: msg.metadata.get("timestamp", 0))
        except Exception as e:
            raise QdrantError(f"Failed to retrieve messages from Qdrant: {e}") from e

    def search(self, query: str = None, limit: int = None, filters: dict = None) -> list[Message]:
        """Handles all search scenarios correctly."""

        limit = limit or self.config.search_limit
        try:
            if query:
                embedding_result = self.embedder.embed_text(query)
                embedding = embedding_result["embedding"]
                search_result = self.client.search(
                    collection_name=self.index_name,
                    query_vector=embedding,
                    query_filter=self._create_filter(filters) if filters else None,
                    limit=limit,
                    with_payload=True,
                )
                return [Message(**hit.payload) for hit in search_result]
            elif filters:
                qdrant_filter = self._create_filter(filters)
                scroll_result = self.client.scroll(
                    collection_name=self.index_name,
                    scroll_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )[0]
                return [Message(**hit.payload) for hit in scroll_result]
            else:
                return []
        except Exception as e:
            raise QdrantError(f"Error searching in Qdrant: {e}") from e

    def _create_filter(self, filters: dict) -> Filter:
        """Creates filter with 'metadata.' prefix for keys."""
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                condition = FieldCondition(key=f"metadata.{key}", range=Range(**value))
            elif isinstance(value, list):
                condition = FieldCondition(key=f"metadata.{key}", match=MatchAny(any=value))
            else:
                condition = FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            conditions.append(condition)
        return Filter(must=conditions)

    def is_empty(self) -> bool:
        """Checks if the Qdrant collection is empty or doesn't exist."""
        try:
            collection_info = self.client.get_collection(collection_name=self.index_name)
            return collection_info.points_count == 0
        except UnexpectedResponse as e:
            if e.status_code == 404:  # Collection doesn't exist
                return True
            raise QdrantError(f"Failed to check if Qdrant collection is empty: {e}") from e  # Re-raise

    def clear(self):
        """Clears the Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=self.index_name)
        except Exception as e:
            raise QdrantError(f"Failed to clear Qdrant collection: {e}") from e
