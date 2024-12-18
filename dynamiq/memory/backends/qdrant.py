import uuid

from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.nodes.embedders.base import DocumentEmbedder, DocumentEmbedderInputSchema
from dynamiq.prompts import Message
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.qdrant import QdrantVectorStore
from dynamiq.types import Document


class QdrantError(Exception):
    """Base exception for Qdrant-related errors."""
    pass


class Qdrant(MemoryBackend):
    """Qdrant implementation of the memory storage backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Qdrant"
    connection: QdrantConnection
    embedder: DocumentEmbedder
    index_name: str = Field(default="conversations")
    metric: str = Field(default="cosine")
    on_disk: bool = Field(default=False)
    create_if_not_exist: bool = Field(default=True)
    recreate_index: bool = Field(default=False)
    vector_store: QdrantVectorStore | dict | None = None
    _client: QdrantClient | None = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"embedder": True, "vector_store": True}

    def model_post_init(self, __context) -> None:
        """Initialize the vector store after model initialization."""
        if isinstance(self.vector_store, dict):
            self.vector_store = QdrantVectorStore(
                connection=self.connection,
                index_name=self.vector_store["index_name"],
                dimension=self.vector_store["dimension"],
                on_disk=self.vector_store["on_disk"],
                use_sparse_embeddings=self.vector_store["use_sparse_embeddings"],
                sparse_idf=self.vector_store.get("sparse_idf", False),
                metric=self.vector_store["metric"],
                return_embedding=self.vector_store["return_embedding"],
                write_batch_size=self.vector_store["write_batch_size"],
                scroll_size=self.vector_store["scroll_size"],
                content_key=self.vector_store["content_key"],
                create_if_not_exist=self.vector_store.get("create_if_not_exist", True),
                recreate_index=self.vector_store.get("recreate_index", False),
                payload_fields_to_index=self.vector_store.get("payload_fields_to_index"),
            )
        elif not self.vector_store:
            self.vector_store = QdrantVectorStore(
                connection=self.connection,
                index_name=self.index_name,
                metric=self.metric,
                on_disk=self.on_disk,
                create_if_not_exist=self.create_if_not_exist,
                recreate_index=self.recreate_index,
            )

        self._client = self.vector_store._client
        if not self._client:
            raise QdrantError("Failed to initialize Qdrant client")

    @model_validator(mode="after")
    def validate_vector_store(self):
        """Validate and initialize vector store if needed."""
        if isinstance(self.vector_store, dict):
            try:
                self.vector_store = QdrantVectorStore(connection=self.connection, **self.vector_store)
            except Exception as e:
                raise ValueError(f"Failed to initialize vector store from dict: {e}")
        return self

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        kwargs.pop("include_secure_params", None)
        data = super().to_dict(**kwargs)

        # Add embedder data
        data["embedder"] = self.embedder.to_dict(include_secure_params=include_secure_params, **kwargs)

        # Add vector store data if exists
        if self.vector_store:
            data["vector_store"] = {
                "index_name": self.vector_store.index_name,
                "dimension": self.vector_store.dimension,
                "on_disk": self.vector_store.on_disk,
                "use_sparse_embeddings": self.vector_store.use_sparse_embeddings,
                "sparse_idf": getattr(self.vector_store, "sparse_idf", False),
                "metric": self.vector_store.metric,
                "return_embedding": self.vector_store.return_embedding,
                "write_batch_size": self.vector_store.write_batch_size,
                "scroll_size": self.vector_store.scroll_size,
                "content_key": self.vector_store.content_key,
                "create_if_not_exist": self.vector_store.create_if_not_exist,
                "recreate_index": self.vector_store.recreate_index,
                "payload_fields_to_index": self.vector_store.payload_fields_to_index,
            }
        else:
            data["vector_store"] = None

        return data

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
        """Stores a message in Qdrant."""
        try:
            document = self._message_to_document(message)
            embedding_result = (
                self.embedder.execute(input_data=DocumentEmbedderInputSchema(documents=[document]))
                .get("documents")[0]
                .embedding
            )
            document.embedding = embedding_result

            self.vector_store.write_documents(documents=[document], policy=DuplicatePolicy.SKIP)
        except Exception as e:
            raise QdrantError(f"Failed to add message to Qdrant: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Retrieves all messages from Qdrant."""
        try:
            documents = self.vector_store.list_documents(include_embeddings=False)
            messages = [self._document_to_message(doc) for doc in documents]
            return sorted(messages, key=lambda msg: msg.metadata.get("timestamp", 0))
        except Exception as e:
            raise QdrantError(f"Failed to retrieve messages from Qdrant: {e}") from e

    def search(self, query: str | None = None, limit: int = 10, filters: dict | None = None) -> list[Message]:
        """Searches for messages in Qdrant."""
        try:
            qdrant_filters = self._prepare_filters(filters)
            if query:
                embedding_result = (
                    self.embedder.execute(
                        input_data=DocumentEmbedderInputSchema(documents=[Document(id="query", content=query)])
                    )
                    .get("documents")[0]
                    .embedding
                )
                documents = self.vector_store._query_by_embedding(
                    query_embedding=embedding_result,
                    filters=qdrant_filters,
                    top_k=limit,
                    return_embedding=False,
                )
            elif filters:
                documents = self.vector_store.filter_documents(filters=qdrant_filters)
                if limit:
                    documents = documents[:limit]
            else:
                return []

            return [self._document_to_message(doc) for doc in documents]
        except Exception as e:
            raise QdrantError(f"Error searching in Qdrant: {e}") from e

    def _prepare_filters(self, filters: dict | None = None) -> dict | None:
        """Prepares simple filters for Qdrant vector store format."""
        if not filters:
            return None

        conditions = []
        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                condition = {"operator": "==", "field": key, "value": value}
            elif isinstance(value, list):
                condition = {"operator": "in", "field": key, "value": value}
            elif isinstance(value, dict) and any(k in value for k in ["gte", "lte", "gt", "lt"]):
                condition = {"operator": "range", "field": key, **value}
            else:
                raise QdrantError(f"Unsupported filter value type for key '{key}': {type(value)}")

            conditions.append(condition)
        return {"operator": "AND", "conditions": conditions} if conditions else None

    def is_empty(self) -> bool:
        """Checks if the Qdrant collection is empty."""
        try:
            return self.vector_store.count_documents() == 0
        except UnexpectedResponse as e:
            if e.status_code == 404:  # Collection doesn't exist
                return True
            raise QdrantError(f"Failed to check if Qdrant collection is empty: {e}") from e

    def clear(self) -> None:
        """Clears the Qdrant collection."""
        try:
            self.vector_store.delete_documents(delete_all=True)
        except Exception as e:
            raise QdrantError(f"Failed to clear Qdrant collection: {e}") from e
