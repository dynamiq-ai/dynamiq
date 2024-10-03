from typing import Any

from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
from pinecone.core.client.exceptions import PineconeException

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class PineconeError(Exception):
    """Base exception class for Pinecone-related errors."""

    pass


class Pinecone(Backend):
    """Pinecone implementation of the memory storage backend."""

    name = "Pinecone"

    def __init__(self, connection: PineconeConnection, embedder: BaseEmbedder, index_name: str = "conversations"):
        """Initializes the Pinecone memory storage."""
        self.connection = connection
        self.index_name = index_name
        self.embedder = embedder

        # Check for API key and raise an error if missing
        self.api_key = self.connection.api_key
        if not self.api_key:
            raise PineconeError(
                "Pinecone API key not found. Set PINECONE_API_KEY environment variable or "
                "pass it in the PineconeConnection constructor."
            )

        try:
            self.pc = PineconeClient(api_key=self.api_key)
        except PineconeException as e:
            raise PineconeError(f"Failed to connect to Pinecone: {e}") from e

        try:
            if self.index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.index_name,
                    dimension=embedder.embedding_size,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.connection.cloud, region=self.connection.region),
                )
            else:
                self.index = self.pc.Index(self.index_name)
        except PineconeException as e:
            raise PineconeError(f"Error initializing Pinecone index: {e}") from e

    def add(self, message: Message):
        """Stores a message in Pinecone."""
        try:
            embedding_result = self.embedder.embed_text(message.content)
            embedding = embedding_result["embedding"]
            cleaned_metadata = self._clean_metadata(message.model_dump())
            self.index.upsert(vectors=[(message.id, embedding, cleaned_metadata)])
        except PineconeException as e:
            raise PineconeError(f"Error adding message to Pinecone: {e}") from e

    def get_all(self, limit: int = 10000) -> list[Message]:
        """Retrieves all messages from Pinecone."""
        try:
            query_response = self.index.query(
                vector=[0] * self.embedder.embedding_size,
                top_k=limit,
                include_metadata=True,
            )
            messages = [Message(**match["metadata"]) for match in query_response["matches"]]
            return messages
        except PineconeException as e:
            raise PineconeError(f"Error retrieving messages from Pinecone: {e}") from e

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in Pinecone based on the query."""
        try:
            embedding_result = self.embedder.embed_text(query)
            embeddings = embedding_result["embedding"]
            results = self.index.query(vector=embeddings, top_k=search_limit, include_metadata=True)
            messages = [Message(**match["metadata"]) for match in results["matches"]]
            return messages
        except PineconeException as e:
            raise PineconeError(f"Error searching in Pinecone: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the Pinecone index is empty."""
        try:
            stats = self.index.describe_index_stats()
            return stats["total_vector_count"] == 0
        except PineconeException as e:
            raise PineconeError(f"Error checking if Pinecone index is empty: {e}") from e

    def clear(self):
        """Clears the Pinecone index."""
        try:
            self.index.delete(delete_all=True)
        except PineconeException as e:
            raise PineconeError(f"Error clearing Pinecone index: {e}") from e

    def _clean_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata to ensure it only contains valid types for Pinecone."""
        cleaned_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                cleaned_metadata[k] = v
            elif isinstance(v, list):
                cleaned_metadata[k] = [self._clean_metadata(item) if isinstance(item, dict) else item for item in v]
            elif isinstance(v, dict):
                cleaned_metadata[k] = self._clean_metadata(v)
            else:
                cleaned_metadata[k] = str(v)
        return cleaned_metadata
