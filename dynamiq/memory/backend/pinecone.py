from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


def clean_metadata(metadata):
    """Clean metadata to ensure it only contains valid types for Pinecone."""
    if isinstance(metadata, dict):
        return {k: clean_metadata(v) for k, v in metadata.items() if v is not None}
    elif isinstance(metadata, list):
        return [clean_metadata(item) for item in metadata if item is not None]
    elif isinstance(metadata, (str, int, float, bool)):
        return metadata
    else:
        return str(metadata)


class Pinecone(Backend):
    """Pinecone implementation of the memory storage backend."""

    connection: PineconeConnection

    def __init__(self, connection: PineconeConnection, embedder: BaseEmbedder, index_name: str = "conversations"):
        """Initializes the Pinecone memory storage."""
        self.connection = connection
        self.index_name = index_name
        self.api_key = self.connection.api_key
        if not self.api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY or pass it in the constructor.")

        self.pc = PineconeClient(api_key=self.api_key)

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=embedder.embedding_size,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.connection.cloud, region=self.connection.region),
            )

        self.index = self.pc.Index(self.index_name)
        self.embedder = embedder

    def add(self, message: Message):
        """Stores a message in Pinecone."""
        embedding_result = self.embedder.embed_text(message.content)
        embedding = embedding_result["embedding"]
        cleaned_metadata = clean_metadata(message.model_dump())
        self.index.upsert(vectors=[(message.id, embedding, cleaned_metadata)])

    def get_all(self) -> list[Message]:
        """Retrieves all messages from Pinecone."""
        query_response = self.index.query(vector=[0] * 1536, top_k=10000, include_metadata=True)
        return [Message(**match["metadata"]) for match in query_response["matches"]]

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in Pinecone based on the query."""
        embedding_result = self.embedder.embed_text(query)
        embeddings = embedding_result["embedding"]
        results = self.index.query(vector=embeddings, top_k=search_limit, include_metadata=True)
        return [Message(**match["metadata"]) for match in results["matches"]]

    def is_empty(self) -> bool:
        """Checks if the Pinecone index is empty."""
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"] == 0

    def clear(self):
        """Clears the Pinecone index."""
        self.index.delete(delete_all=True)
