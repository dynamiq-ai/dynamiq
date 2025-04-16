import time
import uuid
from datetime import datetime, timezone
from typing import Any

import pymongo
from pydantic import ConfigDict, Field, PrivateAttr
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from dynamiq.connections import MongoDB as MongoDBConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class MongoDBMemoryError(Exception):
    """Base exception class for MongoDB Memory Backend errors."""

    pass


class MongoDB(MemoryBackend):
    """
    MongoDB implementation of the memory storage backend.

    Stores messages as documents in a specified MongoDB collection.
    Uses MongoDB's native querying for filtering and text search (if index enabled).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "MongoDB"
    connection: MongoDBConnection = Field(default_factory=MongoDBConnection)
    database_name: str | None = Field(
        default=None, description="MongoDB database name (overrides connection default if set)"
    )
    index_name: str = Field(default="conversations")
    create_indices_if_not_exists: bool = Field(default=True)

    role_field: str = Field(default="role")
    content_field: str = Field(default="content")
    metadata_field: str = Field(default="metadata")
    timestamp_field: str = Field(default="timestamp_dt")

    _client: pymongo.MongoClient | None = PrivateAttr(default=None)
    _db: Database | None = PrivateAttr(default=None)
    _collection: Collection | None = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {
            "_client": True,
            "_db": True,
            "_collection": True,
        }

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)
        data["connection"] = self.connection.to_dict(include_secure_params=include_secure_params)
        if "type" not in data:
            data["type"] = self.type
        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize the MongoDB connection and ensure indices exist."""
        try:
            self._client = self.connection.connect()
            db_name = self.database_name or self.connection.database
            if not db_name:
                raise ValueError("MongoDB database name must be configured either in backend or connection.")
            self._db = self._client[db_name]
            self._collection = self._db[self.index_name]
            logger.debug(f"MongoDB backend connected to db='{db_name}', collection='{self.index_name}'.")

            if self.create_indices_if_not_exists:
                self._create_indices()

        except PyMongoError as e:
            logger.error(f"Failed to initialize MongoDB connection or collection '{self.index_name}': {e}")
            raise MongoDBMemoryError(f"Failed to initialize MongoDB connection or collection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing MongoDB backend: {e}")
            raise MongoDBMemoryError(f"Unexpected error initializing MongoDB backend: {e}") from e

    def _create_indices(self) -> None:
        """Creates standard indices for efficient querying."""
        if self._collection is None:
            raise MongoDBMemoryError("MongoDB collection not initialized.")
        try:
            self._collection.create_index(
                [(self.timestamp_field, pymongo.ASCENDING)], name="timestamp_asc_idx", background=True
            )
            self._collection.create_index(
                [(f"{self.metadata_field}.user_id", pymongo.ASCENDING)],
                name="metadata_user_id_idx",
                background=True,
                sparse=True,
            )
            self._collection.create_index(
                [(f"{self.metadata_field}.session_id", pymongo.ASCENDING)],
                name="metadata_session_id_idx",
                background=True,
                sparse=True,
            )

            logger.debug(f"Ensured indices exist for MongoDB collection '{self.index_name}'.")
        except PyMongoError as e:
            logger.warning(
                f"Could not ensure all indices for MongoDB "
                f"collection '{self.index_name}': {e}. Existing indices might be used."
            )
        except Exception as e:
            logger.error(f"Unexpected error creating " f"indices for MongoDB: {e}")

    def _message_to_doc(self, message: Message) -> dict:
        """Converts a Message object to a MongoDB document dictionary."""
        message_id = message.metadata.get("message_id", str(uuid.uuid4()))
        timestamp_float = message.metadata.get("timestamp", time.time())

        try:
            timestamp_dt = datetime.fromtimestamp(timestamp_float, tz=timezone.utc)
        except (TypeError, ValueError):
            logger.warning(f"Invalid timestamp {timestamp_float} for message {message_id}. Using current time.")
            timestamp_dt = datetime.now(timezone.utc)

        metadata_to_store = message.metadata or {}
        metadata_to_store["message_id"] = message_id
        metadata_to_store["timestamp_float"] = timestamp_float

        doc = {
            self.role_field: message.role.value,
            self.content_field: message.content,
            self.metadata_field: metadata_to_store,
            self.timestamp_field: timestamp_dt,
        }
        return doc

    def _doc_to_message(self, doc: dict) -> Message:
        """Converts a MongoDB document dictionary back to a Message object."""
        metadata = doc.get(self.metadata_field, {})

        if "timestamp_float" in metadata:
            metadata["timestamp"] = metadata.pop("timestamp_float")
        elif self.timestamp_field in doc and isinstance(doc[self.timestamp_field], datetime):
            metadata["timestamp"] = doc[self.timestamp_field].timestamp()

        if "_id" in doc:
            metadata["doc_id"] = str(doc["_id"])

        return Message(
            role=MessageRole(doc.get(self.role_field, MessageRole.USER.value)),
            content=doc.get(self.content_field, ""),
            metadata=metadata,
        )

    def add(self, message: Message) -> None:
        """Adds a message to the MongoDB collection."""
        try:
            doc = self._message_to_doc(message)
            result = self._collection.insert_one(doc)
            logger.debug(f"MongoDB Memory ({self.index_name}): Added message with doc_id {result.inserted_id}")
        except PyMongoError as e:
            logger.error(f"Error adding message to MongoDB: {e}")
            raise MongoDBMemoryError(f"Error adding message to MongoDB: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error preparing message for MongoDB: {e}")
            raise MongoDBMemoryError(f"Error preparing message data: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Retrieves messages from MongoDB, sorted chronologically (oldest first)."""
        try:
            cursor = self._collection.find().sort(self.timestamp_field, pymongo.ASCENDING)
            if limit is not None and limit > 0:
                cursor = cursor.limit(limit)

            docs = list(cursor)
            messages = [self._doc_to_message(doc) for doc in docs]
            logger.debug(f"MongoDB Memory ({self.index_name}): Retrieved {len(messages)} messages.")
            return messages
        except PyMongoError as e:
            logger.error(f"Error retrieving messages from MongoDB: {e}")
            raise MongoDBMemoryError(f"Error retrieving messages from MongoDB: {e}") from e

    def _build_mongo_filter(self, query: str | None, filters: dict | None) -> dict:
        """Builds the MongoDB filter dictionary."""
        mongo_filter = {}

        if filters:
            for key, value in filters.items():

                filter_key = f"{self.metadata_field}.{key}"
                mongo_filter[filter_key] = value

        if query:
            mongo_filter["$text"] = {"$search": query}

        return mongo_filter

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Searches messages using MongoDB's find() with filters and optional text search."""

        mongo_filter = self._build_mongo_filter(query, filters)
        projection = None
        sort_order = []

        if query:
            projection = {"score": {"$meta": "textScore"}}
            sort_order.append(("score", {"$meta": "textScore"}))
        else:
            sort_order.append((self.timestamp_field, pymongo.DESCENDING))

        try:
            cursor = self._collection.find(mongo_filter, projection=projection).sort(sort_order)

            if limit is not None and limit > 0:
                cursor = cursor.limit(limit)

            docs = list(cursor)
            messages = [self._doc_to_message(doc) for doc in docs]

            logger.debug(
                f"MongoDB Memory ({self.index_name}): Found {len(messages)} search results "
                f"(Query: {'Yes' if query else 'No'}, Filters: {'Yes' if filters else 'No'}, Limit: {limit})"
            )

            return messages

        except PyMongoError as e:
            if query and "text index required" in str(e).lower():
                logger.error(
                    f"Text search failed: Text index on "
                    f"field '{self.content_field}' is required in"
                    f" collection '{self.index_name}'. "
                    f"Create it via `create_indices_if_not_exists=True` or manually."
                )
                raise MongoDBMemoryError(f"Text search requires a text index on '{self.content_field}'.") from e
            logger.error(f"Error searching MongoDB: {e}")
            raise MongoDBMemoryError(f"Error searching MongoDB: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the MongoDB collection is empty."""
        try:
            return self._collection.count_documents({}, limit=1) == 0
        except PyMongoError as e:
            logger.error(f"Error checking if MongoDB collection is empty: {e}")
            raise MongoDBMemoryError(f"Error checking if MongoDB collection is empty: {e}") from e

    def clear(self) -> None:
        """Clears the MongoDB collection by deleting all documents."""
        try:
            logger.warning(f"Clearing all documents from MongoDB collection '{self.index_name}'.")
            result = self._collection.delete_many({})
            logger.info(f"MongoDB Memory ({self.index_name}): Cleared {result.deleted_count} documents.")
        except PyMongoError as e:
            logger.error(f"Error clearing MongoDB collection: {e}")
            raise MongoDBMemoryError(f"Error clearing MongoDB collection: {e}") from e
