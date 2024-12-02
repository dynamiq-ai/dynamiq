import uuid

from pydantic import ConfigDict

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message
from dynamiq.storages.vector.pinecone import PineconeVectorStore
from dynamiq.types import Document


class PineconeError(Exception):
    """Base exception class for Pinecone-related errors."""

    pass


class Pinecone(MemoryBackend):
    """Pinecone memory backend implementation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Pinecone"
    connection: PineconeConnection
    embedder: BaseEmbedder
    vector_store: PineconeVectorStore

    def model_post_init(self, __context) -> None:
        """Verify connection after model initialization."""
        try:
            if not self.vector_store._index:
                raise PineconeError("Failed to initialize Pinecone index")
        except Exception as e:
            raise PineconeError(f"Failed to initialize Pinecone vector store: {e}")

    def _message_to_document(self, message: Message) -> Document:
        """Converts a Message object to a Document object."""
        return Document(
            id=str(uuid.uuid4()),
            content=message.content,
            metadata={"role": message.role.value, **(message.metadata or {})},
            embedding=None,
        )

    def _document_to_message(self, document: Document) -> Message:
        """Converts a Document object to a Message object."""
        metadata = dict(document.metadata)
        role = metadata.pop("role")
        return Message(content=document.content, role=role, metadata=metadata, score=document.score)

    def add(self, message: Message) -> None:
        """Stores a message in Pinecone."""
        try:
            document = self._message_to_document(message)
            embedding_result = self.embedder.embed_text(document.content)
            document.embedding = embedding_result["embedding"]

            self.vector_store.write_documents([document])

        except Exception as e:
            raise PineconeError(f"Error adding message to Pinecone: {e}") from e

    def get_all(self, limit: int = 10000) -> list[Message]:
        """Retrieves all messages from Pinecone."""
        try:
            documents = self.vector_store.list_documents(include_embeddings=False)
            messages = [self._document_to_message(doc) for doc in documents]
            return sorted(messages, key=lambda msg: msg.metadata.get("timestamp", 0))
        except Exception as e:
            raise PineconeError(f"Error retrieving messages from Pinecone: {e}") from e

    def _prepare_filters(self, filters: dict | None = None) -> dict | None:
        """Convert simple filters to Pinecone filter format."""
        if not filters:
            return None

        if all(isinstance(v, (str, int, float, bool)) for v in filters.values()):
            conditions = []
            for key, value in filters.items():
                conditions.append({"field": key, "operator": "==", "value": value})
            return {"operator": "AND", "conditions": conditions}
        return filters

    def search(self, query: str | None = None, filters: dict | None = None, limit: int = 10) -> list[Message]:
        """Searches for messages in Pinecone based on the query and/or filters."""
        try:
            normalized_filters = self._prepare_filters(filters)

            if query:
                embedding_result = self.embedder.embed_text(query)
                documents = self.vector_store._embedding_retrieval(
                    query_embedding=embedding_result["embedding"],
                    namespace=self.namespace,
                    filters=normalized_filters,
                    top_k=limit,
                    exclude_document_embeddings=True,
                )
            elif normalized_filters:
                dummy_vector = [0.0] * self.embedder.dimensions
                documents = self.vector_store._embedding_retrieval(
                    query_embedding=dummy_vector,
                    namespace=self.namespace,
                    filters=normalized_filters,
                    top_k=limit,
                    exclude_document_embeddings=True,
                )
            else:
                return []

            return [self._document_to_message(doc) for doc in documents]
        except Exception as e:
            raise PineconeError(f"Error searching in Pinecone: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the Pinecone index is empty."""
        try:
            return self.vector_store.count_documents() == 0
        except Exception as e:
            raise PineconeError(f"Error checking if Pinecone index is empty: {e}") from e

    def clear(self) -> None:
        """Clears the Pinecone index."""
        try:
            self.vector_store.delete_documents(delete_all=True)
        except Exception as e:
            raise PineconeError(f"Error clearing Pinecone index: {e}") from e
