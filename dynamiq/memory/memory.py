from datetime import datetime
from enum import Enum

from dynamiq.memory.backends import InMemory, MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class FormatType(Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    XML = "xml"


class MemoryRetrievalStrategy(Enum):
    ALL = "all"
    RELEVANT = "relevant"
    BOTH = "both"


class Memory:
    """Manages the storage and retrieval of messages."""

    def __init__(self, search_limit: int = 5, search_filters: dict = None, backend: MemoryBackend | None = None):
        """Initializes the Memory with the given search parameters and backend.

        If no backend is provided, an InMemory backend is used by default.
        """
        backend = backend or InMemory()
        if not isinstance(backend, MemoryBackend):
            raise TypeError("backend must be an instance of Backend")

        self.search_limit = search_limit
        self.search_filters = search_filters or {}
        self.backend = backend

    def add(self, role: MessageRole, content: str, metadata: dict | None = None):
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

    def search(self, query: str = None, filters: dict = None) -> list[Message]:
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
        self, query: str, filters: dict = None, format_type: FormatType = FormatType.PLAIN
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
        elif format == FormatType.XML:
            xml_string = "<messages>\n"
            for msg in messages:
                xml_string += "  <message>\n"
                xml_string += f"    <role>{msg.role.value}</role>\n"
                xml_string += f"    <content>{msg.content}</content>\n"
                xml_string += "  </message>\n"
            xml_string += "</messages>"
            return xml_string
        else:
            raise ValueError(f"Unsupported format: {format}")

    def is_empty(self) -> bool:
        """Checks if the memory is empty."""
        return self.backend.is_empty()

    def clear(self):
        """Clears the memory."""
        try:
            self.backend.clear()
            logger.debug(f"Memory {self.backend.name}: Cleared memory")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
            raise e
