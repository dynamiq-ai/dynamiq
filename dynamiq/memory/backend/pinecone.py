import uuid

from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

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
                    dimension=embedder.embedding_size,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=self.connection.cloud, region=self.connection.region),
                )
            self.index = self.pc.Index(self.index_name)
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
            self.index.upsert(vectors=[(message_id, embedding, metadata)])
        except Exception as e:
            raise PineconeError(f"Error adding message to Pinecone: {e}") from e

    def get_all(self, limit: int = 10000) -> list[Message]:
        """Retrieves all messages from Pinecone."""
        try:
            query_response = self.index.query(
                vector=[0] * self.embedder.embedding_size,
                top_k=limit,
                include_metadata=True,
            )
            messages = [
                Message(
                    content=match.metadata["content"],
                    role=match.metadata["role"],
                    metadata=match.metadata.get("metadata", {}),
                )
                for match in query_response["matches"]
            ]
            return messages
        except Exception as e:
            raise PineconeError(f"Error retrieving messages from Pinecone: {e}") from e

    def search(self, query: str = None, search_limit: int = None, filters: dict = None) -> list[Message]:
        """Searches for messages in Pinecone based on the query and/or filters."""
        search_limit = search_limit or self.config.search_limit  # Use default if not provided
        try:
            if query:
                embedding_result = self.embedder.embed_text(query)
                embeddings = embedding_result["embedding"]
                results = self.index.query(vector=embeddings, top_k=search_limit, include_metadata=True, filter=filters)
            elif filters:
                dummy_vector = [0.0] * self.embedder.embedding_size
                results = self.index.query(
                    vector=dummy_vector, top_k=search_limit, include_metadata=True, filter=filters
                )
            else:
                return []

            messages = [
                Message(
                    content=match["metadata"]["content"],
                    role=match["metadata"]["role"],
                    metadata=match["metadata"].get("metadata"),
                    timestamp=match["metadata"].get("timestamp"),
                )
                for match in results["matches"]
            ]
            return messages
        except Exception as e:
            raise PineconeError(f"Error searching in Pinecone: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the Pinecone index is empty."""
        try:
            stats = self.index.describe_index_stats()
            is_empty = stats.get("total_vector_count", 0) == 0
            return is_empty
        except Exception as e:
            raise PineconeError(f"Error checking if Pinecone index is empty: {e}") from e

    def clear(self):
        """Clears the Pinecone index."""
        try:
            self.index.delete(delete_all=True)
        except Exception as e:
            raise PineconeError(f"Error clearing Pinecone index: {e}") from e
