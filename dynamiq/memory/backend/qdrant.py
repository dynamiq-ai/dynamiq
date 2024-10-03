from qdrant_client.http import models as qdrant_models

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class Qdrant(Backend):
    """Qdrant implementation of the memory storage backend."""

    def __init__(self, connection: QdrantConnection, embedder: BaseEmbedder, collection_name: str = "conversations"):
        """Initializes the Qdrant memory storage."""
        self.connection = connection
        self.collection_name = collection_name
        self.embedder = embedder

        try:
            self.client = self.connection.connect()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        # Check if the collection exists, create it if not
        try:
            self.client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            if "Collection not found" in str(e) or "Not found: Collection" in str(e):  # Handle different error messages
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedder.embedding_size, distance=qdrant_models.Distance.COSINE
                    ),
                )
            else:
                raise

    def add(self, message: Message):
        """Stores a message in Qdrant."""
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

    def get_all(self, limit: int = None) -> list[Message]:
        """Retrieves all messages from Qdrant."""
        search_result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=None,
            with_payload=True,
            limit=limit,
        )
        return [Message(**point.payload) for point in search_result[0]]

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in Qdrant based on the query."""
        embedding_result = self.embedder.embed_text(query)
        embedding = embedding_result["embedding"]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=search_limit,
            with_payload=True,
        )
        return [Message(**hit.payload) for hit in search_result]

    def is_empty(self) -> bool:
        """Checks if the Qdrant collection is empty."""
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return collection_info.points_count == 0

    def clear(self):
        """Clears the Qdrant collection."""
        self.client.delete_collection(collection_name=self.collection_name)
