import uuid

from pydantic import ConfigDict, Field

from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.nodes.embedders.base import DocumentEmbedder, DocumentEmbedderInputSchema
from dynamiq.prompts import Message
from dynamiq.storages.vector.pinecone import PineconeVectorStore
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType
from dynamiq.types import Document


class PineconeError(Exception):
    """Base exception class for Pinecone-related errors."""

    pass


class Pinecone(MemoryBackend):
    """Pinecone memory backend implementation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Pinecone"
    connection: PineconeConnection
    embedder: DocumentEmbedder
    index_type: PineconeIndexType
    index_name: str = Field(default="conversations")
    create_if_not_exist: bool = Field(default=True)
    namespace: str = Field(default="default")
    cloud: str | None = Field(default=None)
    region: str | None = Field(default=None)
    environment: str | None = Field(default=None)
    pod_type: str | None = Field(default=None)
    pods: int = Field(default=1)
    vector_store: PineconeVectorStore | None = None

    @property
    def to_dict_exclude_params(self):
        """Define parameters to exclude when converting the class instance to a dictionary."""
        return super().to_dict_exclude_params | {"embedder": True, "vector_store": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        kwargs.pop("include_secure_params", None)
        data = super().to_dict(**kwargs)
        data["embedder"] = self.embedder.to_dict(include_secure_params=include_secure_params, **kwargs)
        return data

    def model_post_init(self, __context) -> None:
        """Initialize the vector store after model initialization."""
        if not self.vector_store:
            self.vector_store = PineconeVectorStore(
                connection=self.connection,
                index_name=self.index_name,
                namespace=self.namespace,
                create_if_not_exist=self.create_if_not_exist,
                index_type=self.index_type,
                cloud=self.cloud,
                region=self.region,
                environment=self.environment,
                pod_type=self.pod_type,
                pods=self.pods,
            )

        if not self.vector_store._index:
            raise PineconeError("Failed to initialize Pinecone index")

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
        return Message(content=document.content, role=role, metadata=metadata)

    def add(self, message: Message) -> None:
        """Stores a message in Pinecone."""
        try:
            document = self._message_to_document(message)
            embedding_result = self.embedder.execute(input_data=DocumentEmbedderInputSchema(documents=[document]))
            document_embedding = embedding_result.get("documents")[0].embedding
            document.embedding = document_embedding
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
                embedding_result = (
                    self.embedder.execute(
                        input_data=DocumentEmbedderInputSchema(
                            documents=[Document(id=str(uuid.uuid4()), content=query)]
                        )
                    )
                    .get("documents")[0]
                    .embedding
                )
                documents = self.vector_store._embedding_retrieval(
                    query_embedding=embedding_result,
                    namespace=self.namespace,
                    filters=normalized_filters,
                    top_k=limit,
                    exclude_document_embeddings=True,
                )
            elif normalized_filters:
                dummy_vector = [0.0] * self.vector_store.dimension
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
