import uuid

from dynamiq.components.embedders.base import BaseEmbedder
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backend.base import MemoryBackend
from dynamiq.prompts import Message
from dynamiq.storages.vector.pinecone import PineconeVectorStore
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from dynamiq.types import Document


class PineconeError(Exception):
    """Base exception class for Pinecone-related errors."""
    pass


class Pinecone(MemoryBackend):
    name = "Pinecone"

    def __init__(
        self,
        connection: PineconeConnection,
        embedder: BaseEmbedder,
        index_type: PineconeIndexType,
        index_name: str = "conversations",
        namespace: str = "default",
        cloud: str | None = None,
        region: str | None = None,
        environment: str | None = None,
        pod_type: str | None = None,
        pods: int = 1,
    ):
        """Initializes the Pinecone memory storage."""
        self.connection = connection
        self.index_name = index_name
        self.embedder = embedder
        self.namespace = namespace

        try:
            self.vector_store = PineconeVectorStore(
                connection=connection,
                index_name=index_name,
                namespace=namespace,
                dimension=embedder.dimensions,
                create_if_not_exist=True,
                index_type=index_type,
                cloud=cloud,
                region=region,
                environment=environment,
                pod_type=pod_type,
                pods=pods,
            )
            # Verify connection
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
            embedding=None,  # Will be populated during write
        )

    def _document_to_message(self, document: Document) -> Message:
        """Converts a Document object to a Message object."""
        metadata = dict(document.metadata)
        role = metadata.pop("role")
        return Message(content=document.content, role=role, metadata=metadata, score=document.score)

    def add(self, message: Message):
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

    def _prepare_filters(self, filters: dict = None) -> dict:
        """Convert simple filters to Pinecone filter format."""
        if not filters:
            return None

        if all(isinstance(v, (str, int, float, bool)) for v in filters.values()):
            conditions = []
            for key, value in filters.items():
                conditions.append({"field": key, "operator": "==", "value": value})
            return {"operator": "AND", "conditions": conditions}
        return filters

    def search(self, query: str = None, filters: dict = None, limit: int = 10) -> list[Message]:
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

    def clear(self):
        """Clears the Pinecone index."""
        try:
            self.vector_store.delete_documents(delete_all=True)
        except Exception as e:
            raise PineconeError(f"Error clearing Pinecone index: {e}") from e
