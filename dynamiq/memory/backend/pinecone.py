import uuid
from typing import Any

from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backend.base import MemoryBackend
from dynamiq.prompts import Message


class PineconeError(Exception):
    """Base exception class for Pinecone-related errors."""

    pass


class Pinecone(MemoryBackend):
    """Pinecone implementation of the memory storage backend."""

    name = "Pinecone"

    def __init__(
        self,
        connection: PineconeConnection,
        embedder: BaseEmbedder,
        index_name: str = "conversations",
    ):
        """Initializes the Pinecone memory storage."""
        self.connection = connection
        self.index_name = index_name
        self.embedder = embedder

        self.api_key = self.connection.api_key
        if not self.api_key:
            raise PineconeError(
                "Pinecone API key not found. Set PINECONE_API_KEY environment variable or "
                "pass it in the PineconeConnection constructor."
            )

        try:
            self.pc = PineconeClient(api_key=self.api_key)
        except Exception as e:
            raise PineconeError(f"Failed to connect to Pinecone: {e}") from e

        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedder.embedding_size,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.connection.cloud, region=self.connection.region),
                )
            self._index = self.pc.Index(self.index_name)
        except Exception as e:
            raise PineconeError(f"Error initializing Pinecone index: {e}") from e

    def add(self, message: Message):
        """Stores a message in Pinecone."""
        try:
            embedding_result = self.embedder.embed_text(message.content)
            embedding = embedding_result["embedding"]
            metadata = {
                "content": message.content,
                "role": message.role.value,
            }
            if message.metadata:
                metadata.update(message.metadata)
            message_id = str(uuid.uuid4())
            self._index.upsert(vectors=[{"id": message_id, "values": embedding, "metadata": metadata}])
        except Exception as e:
            raise PineconeError(f"Error adding message to Pinecone: {e}") from e

    def get_all(self, limit: int = 10000) -> list[Message]:
        """Retrieves all messages from Pinecone."""
        try:
            query_response = self._index.query(
                vector=[0] * self.embedder.embedding_size,
                top_k=limit,
                include_metadata=True,
            )
            messages = [
                Message(
                    content=match.metadata["content"],
                    role=match.metadata["role"],
                    metadata={k: v for k, v in match.metadata.items() if k not in ("content", "role")},
                )
                for match in query_response["matches"]
            ]
            return sorted(messages, key=lambda msg: msg.metadata.get("timestamp", 0))
        except Exception as e:
            raise PineconeError(f"Error retrieving messages from Pinecone: {e}") from e

    def search(self, query: str = None, filters: dict = None, search_limit: int = None) -> list[Message]:
        """Searches for messages in Pinecone based on the query and/or filters."""
        search_limit = search_limit or self.config.search_limit
        normalized_filters = self._normalize_filters(filters) if filters else None
        try:
            if query:
                embedding_result = self.embedder.embed_text(query)
                embedding = embedding_result["embedding"]
                response = self._index.query(
                    vector=embedding,
                    top_k=search_limit,
                    include_metadata=True,
                    filter=normalized_filters,
                )
            elif normalized_filters:
                dummy_vector = [0.0] * self.embedder.embedding_size
                response = self._index.query(
                    vector=dummy_vector,
                    top_k=search_limit,
                    include_metadata=True,
                    filter=normalized_filters,
                )
            else:
                return []
            messages = [Message(**match.metadata) for match in response.matches]
            return messages
        except Exception as e:
            raise PineconeError(f"Error searching in Pinecone: {e}") from e

    def _normalize_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Normalizes filters to ensure Pinecone compatibility."""
        normalized_filters = {}
        for key, value in filters.items():
            if isinstance(value, dict):
                normalized_filters[key] = value
            elif isinstance(value, list):
                normalized_filters[key] = {"$in": value}
            else:
                normalized_filters[key] = value

        return normalized_filters

    def is_empty(self) -> bool:
        """Checks if the Pinecone index is empty."""
        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0) == 0
        except Exception as e:
            raise PineconeError(f"Error checking if Pinecone index is empty: {e}") from e

    def clear(self):
        """Clears the Pinecone index."""
        try:
            self._index.delete(delete_all=True)
        except Exception as e:
            raise PineconeError(f"Error clearing Pinecone index: {e}") from e
