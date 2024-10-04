from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class InMemoryError(Exception):
    """Base exception class for InMemory backend errors."""

    pass


class InMemory(Backend):
    """In-memory implementation of the memory storage backend."""

    name = "InMemory"

    def __init__(self):
        """Initializes the in-memory storage."""
        self.messages: list[Message] = []

    def add(self, message: Message):
        """Adds a message to the in-memory list."""
        try:
            self.messages.append(message)
        except Exception as e:
            raise InMemoryError(f"Error adding message to InMemory backend: {e}") from e

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the in-memory list."""
        return self.messages

    def search(self, query: str = None, search_limit: int = None, filters: dict = None) -> list[Message]:
        """Searches for messages, applying optional query and/or filters."""
        search_limit = search_limit or self.config.search_limit
        matching_messages = self.messages

        if query:
            matching_messages = [msg for msg in matching_messages if query.lower() in msg.content.lower()]

        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    matching_messages = [
                        msg for msg in matching_messages if any(v in str(msg.metadata.get(key, "")) for v in value)
                    ]
                else:
                    matching_messages = [msg for msg in matching_messages if value in str(msg.metadata.get(key, ""))]

        return matching_messages[:search_limit]

    def is_empty(self) -> bool:
        """Checks if the in-memory list is empty."""
        return len(self.messages) == 0

    def clear(self):
        """Clears the in-memory list."""
        self.messages = []
