from datetime import datetime

from dynamiq.memory.backend import InMemory, MemoryBackend
from dynamiq.memory.config import Config
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class Memory:
    """Manages the storage and retrieval of messages."""

    def __init__(self, config: Config = Config(), backend: MemoryBackend = InMemory()):
        """Initializes the Memory with the given configuration and backend.

        If no backend is provided, an InMemory backend is used by default.
        """
        if not isinstance(backend, MemoryBackend):
            raise TypeError("backend must be an instance of Backend")
        self.config = config
        self.backend = backend

    def add_message(self, role: MessageRole, content: str, metadata: dict = None):
        """Adds a message to the memory."""
        try:
            metadata = metadata or {}
            metadata["timestamp"] = datetime.utcnow().timestamp()
            message = Message(role=role, content=content, metadata=metadata)
            self.backend.add(message)
            logger.debug(f"Memory {self.backend.name}: Added message: {message.role}: {message.content[:20]}...")
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    def get_all_messages(self) -> list[Message]:
        """Retrieves all messages from the memory."""
        messages = self.backend.get_all()
        logger.debug(f"Memory {self.backend.name}: Retrieved {len(messages)} messages")
        return messages

    def get_all_messages_as_string(self, format: str = "plain") -> str:
        """Retrieves all messages as a formatted string."""
        messages = self.get_all_messages()
        return self._format_messages_as_string(messages, format)

    def search_messages(self, query: str = None, filters: dict = None) -> list[Message]:
        """Searches for messages relevant to the query or filters."""
        search_results = self.backend.search(
            query=query, search_limit=self.config.search_limit, filters=filters or self.config.search_filters
        )
        logger.debug(
            f"Memory {self.backend.name}: Found {len(search_results)} search results for query: {query}, filters: {filters}"  # noqa: E501
        )
        return search_results

    def get_search_results_as_string(self, query: str, filters: dict = None, format: str = "plain") -> str:
        """Searches for messages relevant to the query and returns them as a string."""
        messages = self.search_messages(query, filters)
        return self._format_messages_as_string(messages, format)

    def _format_messages_as_string(self, messages: list[Message], format: str = "plain") -> str:
        """Converts a list of messages to a formatted string."""
        if format == "plain":
            return "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])
        elif format == "markdown":
            return "\n".join([f"**{msg.role.value}:** {msg.content}" for msg in messages])
        elif format == "xml":
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

    def is_memory_empty(self) -> bool:
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
