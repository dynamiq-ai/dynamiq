import json
import time
import uuid
from typing import Any, ClassVar

import redis
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import Redis as RedisConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class RedisMemoryError(Exception):
    """Base exception class for Redis-related errors in the memory backend."""

    pass


class Redis(MemoryBackend):
    """
    Redis implementation of the memory storage backend using Hashes and Sorted Sets.

    Note:
        - Metadata is stored as a JSON string within the Hash.
        - The `search` method performs filtering and basic keyword matching
          client-side after retrieving potentially *all* relevant messages based
          on timestamp order.
        - Assumes `decode_responses=True` is set on the Redis client provided
          by the connection.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Redis"
    connection: RedisConnection = Field(default_factory=RedisConnection)
    index_name: str = Field(default="default", description="Prefix for all Redis keys used by this memory instance.")

    _redis_client: redis.Redis | None = PrivateAttr(default=None)

    _MESSAGE_HASH_PREFIX: ClassVar[str] = "message"
    _TIMESTAMP_ZSET_KEY: ClassVar[str] = "messages_by_time"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {"_redis_client": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary, including connection details."""
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)

        data["connection"] = self.connection.to_dict(include_secure_params=include_secure_params, **kwargs)
        data["index_name"] = self.index_name

        if "type" not in data:
            data["type"] = self.type

        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Redis client using the connection configuration."""
        try:
            self._redis_client = self.connection.connect()
            logger.debug(
                f"Redis backend '{self.name}' (ID: {self.id}) successfully connected via {self.connection.type}"
            )
            if not self._redis_client:
                raise RedisMemoryError("Failed to establish Redis connection.")
        except ConnectionError as e:
            logger.error(f"Redis backend '{self.name}' failed to initialize connection: {e}")
            raise RedisMemoryError(f"Failed to initialize Redis connection: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis backend '{self.name}': {e}")
            raise RedisMemoryError(f"Unexpected error initializing Redis backend: {e}") from e

    def _get_message_hash_key(self, message_id: str) -> str:
        """Generates the Redis key for a message hash."""
        safe_message_id = str(message_id).replace(":", "_")
        return f"{self.index_name}:{self._MESSAGE_HASH_PREFIX}:{safe_message_id}"

    def _get_timestamp_zset_key(self) -> str:
        """Generates the Redis key for the timestamp sorted set."""
        return f"{self.index_name}:{self._TIMESTAMP_ZSET_KEY}"

    def add(self, message: Message) -> None:
        """
        Adds a message to Redis. Stores message data in a Hash
        and its ID (the Hash key) in a Sorted Set scored by timestamp.

        Args:
            message: Message to add to storage

        Raises:
            RedisMemoryError: If the message cannot be added due to Redis errors or data issues.
        """
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized. Connection may have failed.")

        try:
            message_id = message.metadata.get("message_id", str(uuid.uuid4()))
            message.metadata["message_id"] = message_id

            hash_key = self._get_message_hash_key(message_id)
            zset_key = self._get_timestamp_zset_key()

            timestamp = message.metadata.get("timestamp")
            if timestamp is None:
                timestamp = time.time()
                message.metadata["timestamp"] = timestamp
            elif not isinstance(timestamp, (int, float)):
                try:
                    timestamp = float(timestamp)
                    message.metadata["timestamp"] = timestamp
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid timestamp format in metadata for message {message_id}. Using current time."
                    )
                    timestamp = time.time()
                    message.metadata["timestamp"] = timestamp

            try:
                metadata_json = json.dumps(message.metadata or {})
            except TypeError as e:
                logger.error(
                    f"Metadata for message {message_id} is not JSON serializable: {message.metadata}. Error: {e}"
                )
                raise RedisMemoryError(f"Metadata not JSON serializable: {e}") from e

            message_data = {
                "role": message.role.value,
                "content": message.content,
                "metadata": metadata_json,
            }

            with self._redis_client.pipeline() as pipe:
                pipe.hset(hash_key, mapping=message_data)
                pipe.zadd(zset_key, {hash_key: timestamp})
                results = pipe.execute()

                if not all(isinstance(res, int) for res in results):
                    logger.warning(f"Potential issue adding message {message_id} to Redis. Pipeline results: {results}")

            logger.debug(
                f"Redis Memory ({self.index_name}): Added message {message_id} "
                f"(key: {hash_key}) with timestamp {timestamp}"
            )

        except redis.RedisError as e:
            logger.error(f"Redis error adding message: {e}")
            raise RedisMemoryError(f"Redis error adding message: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error adding message to Redis: {e}")
            raise RedisMemoryError(f"Unexpected error adding message: {e}") from e

    def _fetch_messages_by_keys(self, message_keys: list[str]) -> list[Message]:
        """Fetches full message details from Hashes given their keys."""
        if not message_keys:
            return []
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized.")

        messages = []
        try:
            with self._redis_client.pipeline(transaction=False) as pipe:
                for key in message_keys:
                    pipe.hgetall(key)
                hash_data_list = pipe.execute()

            for key, hash_data in zip(message_keys, hash_data_list):
                if not hash_data:
                    logger.warning(
                        f"Redis Memory ({self.index_name}): Message key {key} found in ZSET but Hash data missing."
                    )
                    continue

                try:
                    metadata_str = hash_data.get("metadata", "{}")
                    metadata = json.loads(metadata_str)

                    role_str = hash_data.get("role")
                    try:
                        role = MessageRole(role_str) if role_str else MessageRole.USER
                    except ValueError:
                        logger.warning(
                            f"Redis Memory ({self.index_name}): Invalid role '{role_str}'"
                            f" found for key {key}. Defaulting to USER."
                        )
                        role = MessageRole.USER

                    message = Message(
                        role=role,
                        content=hash_data.get("content", ""),
                        metadata=metadata,
                    )
                    messages.append(message)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Redis Memory ({self.index_name}): Error parsing metadata JSON for key {key}: {e}. "
                        f"Data: {hash_data.get('metadata')}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Redis Memory ({self.index_name}): Unexpected error reconstructing message for key {key}: {e}."
                        f" Data: {hash_data}"
                    )
                    continue

            return messages

        except redis.RedisError as e:
            logger.error(f"Redis error fetching messages by keys: {e}")
            raise RedisMemoryError(f"Redis error fetching messages: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching messages from Redis: {e}")
            raise RedisMemoryError(f"Unexpected error fetching messages: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """
        Retrieves messages from Redis, sorted chronologically (oldest first).

        Args:
            limit: Maximum number of *most recent* messages to return.
                   If None or 0, returns all messages.

        Returns:
            List of messages sorted by timestamp (oldest first).

        Raises:
            RedisMemoryError: If messages cannot be retrieved.
        """
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized.")

        try:
            zset_key = self._get_timestamp_zset_key()
            if limit is not None and limit > 0:
                message_keys = self._redis_client.zrevrange(zset_key, 0, limit - 1)
                message_keys.reverse()
            else:
                message_keys = self._redis_client.zrange(zset_key, 0, -1)

            logger.debug(
                f"Redis Memory ({self.index_name}): Retrieving {len(message_keys)} message keys (limit={limit})."
            )
            return self._fetch_messages_by_keys(message_keys)

        except redis.RedisError as e:
            logger.error(f"Redis error retrieving messages: {e}")
            raise RedisMemoryError(f"Error retrieving messages from Redis: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error retrieving all messages from Redis: {e}")
            raise RedisMemoryError(f"Unexpected error retrieving messages: {e}") from e

    def _apply_filters_client_side(self, messages: list[Message], filters: dict[str, Any]) -> list[Message]:
        """Applies metadata filters to a list of messages (client-side)."""
        if not filters:
            return messages

        filtered_messages = []
        for msg in messages:
            match = True
            for key, value in filters.items():
                metadata_value = msg.metadata.get(key)

                if isinstance(value, list):
                    if metadata_value not in value:
                        match = False
                        break
                elif value is None:
                    if key not in msg.metadata or metadata_value is not None:
                        match = False
                        break

                else:
                    if metadata_value != value:
                        match = False
                        break
            if match:
                filtered_messages.append(msg)
        return filtered_messages

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """
        Searches messages in Redis, returning results sorted chronologically.

        Process:
        1. Retrieves all message keys from the sorted set (chronological order).
        2. Fetches the full data for all messages.
        3. Applies metadata `filters` (client-side).
        4. If `query` is provided, performs a case-insensitive substring match
           on the content of the filtered messages (client-side).
        5. Applies the `limit` to return the most recent matching messages.

        Args:
            query: Optional search string for basic keyword matching (case-insensitive substring).
            filters: Optional dictionary for filtering messages by metadata (exact match or list membership).
            limit: Maximum number of *most recent* matching messages to return. If None, returns all matches.

        Returns:
            List of matching messages sorted chronologically (oldest first).

        Raises:
            RedisMemoryError: If the search operation fails due to Redis errors.

        Note:
            This implementation fetches all messages before filtering/searching,
            which can be inefficient for large datasets. It sorts results
            chronologically, not by relevance score.
        """
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized.")

        try:
            zset_key = self._get_timestamp_zset_key()
            all_message_keys = self._redis_client.zrange(zset_key, 0, -1)
            if not all_message_keys:
                logger.debug(f"Redis Memory ({self.index_name}): Search found no messages in ZSET.")
                return []

            all_messages = self._fetch_messages_by_keys(all_message_keys)
            logger.debug(
                f"Redis Memory ({self.index_name}): Search "
                f"fetched {len(all_messages)} total messages for filtering/querying."
            )

            if filters:
                messages_to_search = self._apply_filters_client_side(all_messages, filters)
                logger.debug(
                    f"Redis Memory ({self.index_name}): Search - {len(messages_to_search)} messages "
                    f"after applying filters: {filters}"
                )
            else:
                messages_to_search = all_messages

            if query:
                query_lower = query.lower()
                results = [msg for msg in messages_to_search if query_lower in msg.content.lower()]
                logger.debug(
                    f"Redis Memory ({self.index_name}): Search - {len(results)} messages "
                    f"after applying query: '{query}'"
                )
            else:
                results = messages_to_search

            if limit is not None and limit > 0:
                final_results = results[-limit:]
            else:
                final_results = results

            logger.debug(
                f"Redis Memory ({self.index_name}): Search final results count = {len(final_results)} (limit={limit})"
            )
            return final_results

        except redis.RedisError as e:
            logger.error(f"Redis error during search: {e}")
            raise RedisMemoryError(f"Error searching Redis memory: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error searching Redis memory: {e}")
            raise RedisMemoryError(f"Unexpected error searching Redis memory: {e}") from e

    def is_empty(self) -> bool:
        """
        Checks if the Redis memory (sorted set) is empty.

        Returns:
            True if the memory is empty, False otherwise.

        Raises:
            RedisMemoryError: If the check fails.
        """
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized.")
        try:
            zset_key = self._get_timestamp_zset_key()
            count = self._redis_client.zcard(zset_key)
            return count == 0
        except redis.RedisError as e:
            logger.error(f"Redis error checking if memory is empty: {e}")
            raise RedisMemoryError(f"Error checking if Redis memory is empty: {e}") from e

    def clear(self) -> None:
        """
        Clears the Redis memory by deleting all associated message Hashes
        and the timestamp Sorted Set.

        Raises:
            RedisMemoryError: If the memory cannot be cleared.
        """
        if self._redis_client is None:
            raise RedisMemoryError("Redis client not initialized.")
        try:
            zset_key = self._get_timestamp_zset_key()
            message_keys = self._redis_client.zrange(zset_key, 0, -1)
            keys_to_delete = [zset_key]
            if message_keys:
                keys_to_delete.extend(message_keys)

            if not keys_to_delete:
                logger.info(f"Redis Memory ({self.index_name}): Clear called, but memory is already empty.")
                return

            with self._redis_client.pipeline(transaction=False) as pipe:
                chunk_size = 500
                for i in range(0, len(keys_to_delete), chunk_size):
                    chunk = keys_to_delete[i : i + chunk_size]
                    pipe.unlink(*chunk)
                pipe.execute()

            logger.info(
                f"Redis Memory ({self.index_name}): Cleared memory (attempted to delete {len(message_keys)}"
                f" messages and the index set)."
            )

        except redis.RedisError as e:
            logger.error(f"Redis error clearing memory: {e}")
            raise RedisMemoryError(f"Error clearing Redis memory: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error clearing Redis memory: {e}")
            raise RedisMemoryError(f"Unexpected error clearing Redis memory: {e}") from e
