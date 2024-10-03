from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import ApiException

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class QdrantError(Exception):
    """Base exception for Qdrant-related errors."""

    pass


class Qdrant(Backend):
    """Qdrant implementation of the memory storage backend."""

    name = "Qdrant"

    def __init__(self, connection: QdrantConnection, embedder: BaseEmbedder, collection_name: str = "conversations"):
        """Initializes the Qdrant memory storage."""
        self.connection = connection
        self.collection_name = collection_name
        self.embedder = embedder

        try:
            self.client = self.connection.connect()
        except Exception as e:
            raise QdrantError(f"Failed to connect to Qdrant: {e}") from e

        # Check if the collection exists, create it if not
        if not self.client.collection_exists(collection_name=self.collection_name):
            print(f"Qdrant collection '{self.collection_name}' not found, creating it...")
            self._create_collection()
        else:
            print(f"Connected to existing Qdrant collection '{self.collection_name}'")

    def _create_collection(self):
        """Creates the Qdrant collection."""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedder.embedding_size, distance=qdrant_models.Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        except ApiException as e:
            raise QdrantError(f"Failed to create collection '{self.collection_name}': {e}") from e

    def add(self, message: Message):
        """Stores a message in Qdrant."""
        try:
            embedding_result = self.embedder.embed_text(message.content)
            embedding = embedding_result["embedding"]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=message.id,
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
                collection_name=self.collection_name,
                scroll_filter=None,
                with_payload=True,
                limit=limit,
            )
            return [Message(**point.payload) for point in search_result[0]]
        except Exception as e:
            raise QdrantError(f"Failed to retrieve messages from Qdrant: {e}") from e

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in Qdrant based on the query."""
        try:
            embedding_result = self.embedder.embed_text(query)
            embedding = embedding_result["embedding"]
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=search_limit,
                with_payload=True,
            )
            return [Message(**hit.payload) for hit in search_result]
        except Exception as e:
            raise QdrantError(f"Failed to search in Qdrant: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the Qdrant collection is empty."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info.points_count == 0
        except Exception as e:
            raise QdrantError(f"Failed to check if Qdrant collection is empty: {e}") from e

    def clear(self):
        """Clears the Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            raise QdrantError(f"Failed to clear Qdrant collection: {e}") from e
