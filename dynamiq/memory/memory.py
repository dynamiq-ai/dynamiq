from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.memory.backends import InMemory, MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class FormatType(str, Enum):
    """Enum for message format types."""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    XML = "xml"


class MemoryRetrievalStrategy(str, Enum):
    """Enum for memory retrieval strategies."""
    ALL = "all"
    RELEVANT = "relevant"
    BOTH = "both"


class MemoryError(Exception):
    """Base exception for Memory errors."""

    pass


class Memory(BaseModel):
    """Manages the storage and retrieval of messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_LIMIT: ClassVar[int] = 1000

    message_limit: int = Field(default=DEFAULT_LIMIT, gt=0, description="Default limit for message retrieval")
    backend: MemoryBackend = Field(default_factory=InMemory, description="Backend storage implementation")
    filters: dict[str, Any] = Field(default_factory=dict, description="Default filters to apply to searches")

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"backend": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        kwargs.pop("include_secure_params", None)
        data = self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)
        data["backend"] = self.backend.to_dict(include_secure_params=include_secure_params, **kwargs)
        return data

    def add(self, role: MessageRole, content: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Adds a message to the memory.

        Args:
            role: The role of the message sender
            content: The message content
            metadata: Additional metadata for the message

        Raises:
            MemoryError: If the message cannot be added
        """
        try:
            metadata = metadata or {}
            if "timestamp" not in metadata:
                metadata["timestamp"] = datetime.utcnow().timestamp()

            sanitized_metadata = {}
            for key, value in metadata.items():
                sanitized_metadata[key] = "" if value is None else value

            message = Message(role=role, content=content, metadata=sanitized_metadata)
            self.backend.add(message)

            logger.debug(
                f"Memory {self.backend.name}: "
                f"Added message: {message.role}: {message.content[:min(20, len(message.content))]}..."
            )
        except Exception as e:
            logger.error(f"Unexpected error adding message: {e}")
            raise MemoryError(f"Unexpected error adding message: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """
        Retrieves all messages from the memory, optionally limited to most recent.

        Args:
            limit: Maximum number of messages to return. If provided, returns the most recent messages.
                  If None, uses the configured message_limit.

        Returns:
            List of messages sorted by timestamp (oldest first)

        Raises:
            MemoryError: If messages cannot be retrieved
        """
        try:
            effective_limit = limit if limit is not None else self.message_limit

            messages = self.backend.get_all(limit=effective_limit)
            logger.debug(f"Memory {self.backend.name}: Retrieved {len(messages)} messages")
            return messages
        except Exception as e:
            logger.error(f"Unexpected error retrieving messages: {e}")
            raise MemoryError(f"Unexpected error retrieving messages: {e}") from e

    def get_all_messages_as_string(self, format_type: FormatType = FormatType.PLAIN) -> str:
        """
        Retrieves all messages as a formatted string.

        Args:
            format_type: Format to use for the string output

        Returns:
            Formatted string representation of all messages

        Raises:
            MemoryError: If messages cannot be retrieved or formatted
        """
        messages = self.get_all()
        return self._format_messages_as_string(messages=messages, format_type=format_type)

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """
        Searches for messages relevant to the query or filters.

        Args:
            query: Search query string (optional)
            filters: Optional metadata filters to apply (overrides default filters)
            limit: Maximum number of messages to return (defaults to message_limit)

        Returns:
            List of matching messages sorted by relevance

        Raises:
            MemoryError: If search operation fails
        """
        try:
            effective_filters = self.filters.copy()
            if filters:
                effective_filters.update(filters)

            effective_limit = limit if limit is not None else self.message_limit

            results = self.backend.search(query=query, filters=effective_filters, limit=effective_limit)

            logger.debug(
                f"Memory {self.backend.name}: Found {len(results)} search results for query: {query}, "
                f"filters: {effective_filters}"
            )
            return results
        except Exception as e:
            logger.error(f"Unexpected error searching memory: {e}")
            raise MemoryError(f"Unexpected error searching memory: {e}") from e

    def get_search_results_as_string(
        self, query: str, filters: dict[str, Any] | None = None, format_type: FormatType = FormatType.PLAIN
    ) -> str:
        """
        Searches for messages relevant to the query and returns them as a string.

        Args:
            query: Search query string
            filters: Optional metadata filters to apply
            format_type: Format to use for the string output

        Returns:
            Formatted string representation of search results

        Raises:
            MemoryError: If search operation fails or results cannot be formatted
        """
        messages = self.search(query, filters)
        return self._format_messages_as_string(messages=messages, format_type=format_type)

    def _format_messages_as_string(self, messages: list[Message], format_type: FormatType = FormatType.PLAIN) -> str:
        """
        Converts a list of messages to a formatted string.

        Args:
            messages: List of messages to format
            format_type: Format to use for the string output

        Returns:
            Formatted string representation of messages

        Raises:
            ValueError: If an unsupported format type is provided
        """
        if format_type == FormatType.PLAIN:
            return "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])

        elif format_type == FormatType.MARKDOWN:
            return "\n".join([f"**{msg.role.value}:** {msg.content}" for msg in messages])

        elif format_type == FormatType.XML:
            return "\n".join(
                [
                    "<messages>",
                    *[
                        "\n".join(
                            [
                                "  <message>",
                                f"    <role>{msg.role.value}</role>",
                                f"    <content>{msg.content}</content>",
                                "  </message>",
                            ]
                        )
                        for msg in messages
                    ],
                    "</messages>",
                ]
            )

        raise ValueError(f"Unsupported format: {format_type}")

    def is_empty(self) -> bool:
        """
        Checks if the memory is empty.

        Returns:
            True if the memory is empty, False otherwise

        Raises:
            MemoryError: If the check fails
        """
        try:
            return self.backend.is_empty()
        except Exception as e:
            logger.error(f"Unexpected error checking if memory is empty: {e}")
            raise MemoryError(f"Unexpected error checking if memory is empty: {e}") from e

    def clear(self) -> None:
        """
        Clears the memory.

        Raises:
            MemoryError: If the memory cannot be cleared
        """
        try:
            self.backend.clear()
            logger.debug(f"Memory {self.backend.name}: Cleared memory")
        except Exception as e:
            logger.error(f"Unexpected error clearing memory: {e}")
            raise MemoryError(f"Unexpected error clearing memory: {e}") from e

    def get_agent_conversation(
        self,
        query: str | None = None,
        limit: int | None = None,
        filters: dict[str, Any] | None = None,
        strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.ALL,
    ) -> list[Message]:
        """
        Retrieves messages from an agent's conversation history based on various filtering criteria.

        This method supports three retrieval strategies:
        - ALL: Returns the most recent messages without semantic search
        - RELEVANT: Returns only messages relevant to the query using semantic search
        - BOTH: Returns a combination of recent messages and those relevant to the query

        Parameters:
        -----------
        query : str | None, optional
            The search query to filter messages by relevance. Required for RELEVANT and BOTH strategies.
            Ignored when strategy is ALL. Default is None.

        limit : int | None, optional
            Maximum number of messages to return. If None, falls back to self.message_limit.

        filters : dict[str, Any] | None, optional
            Additional metadata filters to apply to the search results.

        strategy : MemoryRetrievalStrategy, optional
            The strategy to use for retrieving messages. Choices are:
            - MemoryRetrievalStrategy.ALL: Return most recent messages
            - MemoryRetrievalStrategy.RELEVANT: Return messages relevant to query
            - MemoryRetrievalStrategy.BOTH: Return both recent and relevant messages
            Default is MemoryRetrievalStrategy.ALL.

        Returns:
        --------
        list[Message]
            A list of conversation messages matching the search criteria, ordered chronologically.

        Raises:
        -------
        MemoryError
            If there is an error retrieving the conversation history.
        """
        try:
            effective_limit = limit if limit is not None else self.message_limit

            if strategy == MemoryRetrievalStrategy.RELEVANT and query:
                messages = self.search(query=query, filters=filters, limit=effective_limit)
            elif strategy == MemoryRetrievalStrategy.BOTH and query:
                recent_messages = self.search(
                    query=None, filters=filters, limit=max(effective_limit, self.DEFAULT_LIMIT)
                )

                relevant_messages = self.search(query=query, filters=filters, limit=effective_limit)

                message_dict = {msg.metadata.get("timestamp", 0): msg for msg in recent_messages}
                for msg in relevant_messages:
                    message_dict[msg.metadata.get("timestamp", 0)] = msg

                messages = [msg for _, msg in sorted(message_dict.items())]
            else:
                messages = self.search(query=None, filters=filters, limit=effective_limit)

            return self._extract_valid_conversation(messages, effective_limit)
        except Exception as e:
            logger.error(f"Error retrieving agent conversation: {e}")
            raise MemoryError(f"Failed to retrieve agent conversation: {e}") from e

    def _extract_valid_conversation(self, messages: list[Message], limit: int) -> list[Message]:
        """
        Extracts a valid conversation from a list of messages.

        Ensures:
        1. Messages are sorted by timestamp
        2. For messages with identical timestamps, user messages come before assistant messages
        3. The first message is from a user
        4. We respect the limit but prioritize keeping full conversation flows

        Args:
            messages: List of messages to process
            limit: Maximum number of messages to include in the result

        Returns:
            List of messages forming a valid conversation
        """
        if not messages:
            return []

        def message_sort_key(msg):
            timestamp = msg.metadata.get("timestamp", 0)
            role_priority = 0 if msg.role == MessageRole.USER else 0.1
            return (timestamp, role_priority)

        sorted_messages = sorted(messages, key=message_sort_key)

        if limit and len(sorted_messages) > limit:
            sorted_messages = sorted_messages[-limit:]

        if sorted_messages and sorted_messages[0].role != MessageRole.USER:
            first_user_idx = next((i for i, msg in enumerate(sorted_messages) if msg.role == MessageRole.USER), None)

            if first_user_idx is not None:
                sorted_messages = sorted_messages[first_user_idx:]

        return sorted_messages
