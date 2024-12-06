from datetime import datetime
from enum import Enum
from typing import ClassVar

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


class Memory(BaseModel):
    """Manages the storage and retrieval of messages."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Class constants
    DEFAULT_SEARCH_LIMIT: ClassVar[int] = 5

    # Instance fields
    search_limit: int = Field(default=DEFAULT_SEARCH_LIMIT, gt=0)
    search_filters: dict = Field(default_factory=dict)
    backend: MemoryBackend = Field(default_factory=InMemory)

    def add(self, role: MessageRole, content: str, metadata: dict | None = None) -> None:
        """Adds a message to the memory."""
        try:
            metadata = metadata or {}
            metadata["timestamp"] = datetime.utcnow().timestamp()
            message = Message(role=role, content=content, metadata=metadata)
            self.backend.add(message)
            logger.debug(
                f"Memory {self.backend.name}: "
                f"Added message: {message.role}: {message.content[:min(20, len(message.content))]}..."
            )
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the memory."""
        messages = self.backend.get_all()
        logger.debug(f"Memory {self.backend.name}: Retrieved {len(messages)} messages")
        return messages

    def get_all_messages_as_string(self, format_type: FormatType = FormatType.PLAIN) -> str:
        """Retrieves all messages as a formatted string."""
        messages = self.get_all()
        return self._format_messages_as_string(messages=messages, format_type=format_type)

    def search(self, query: str | None = None, filters: dict | None = None) -> list[Message]:
        """Searches for messages relevant to the query or filters."""
        search_results = self.backend.search(
            query=query, limit=self.search_limit, filters=filters or self.search_filters
        )
        logger.debug(
            f"Memory {self.backend.name}: Found {len(search_results)} search results for query: {query}, "
            f"filters: {filters}"
        )
        return search_results

    def get_search_results_as_string(
        self, query: str, filters: dict | None = None, format_type: FormatType = FormatType.PLAIN
    ) -> str:
        """Searches for messages relevant to the query and returns them as a string."""
        messages = self.search(query, filters)
        return self._format_messages_as_string(messages=messages, format_type=format_type)

    def _format_messages_as_string(self, messages: list[Message], format_type: FormatType = FormatType.PLAIN) -> str:
        """Converts a list of messages to a formatted string."""
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
        """Checks if the memory is empty."""
        return self.backend.is_empty()

    def clear(self) -> None:
        """Clears the memory."""
        try:
            self.backend.clear()
            logger.debug(f"Memory {self.backend.name}: Cleared memory")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise
