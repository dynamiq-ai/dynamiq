from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class InMemory(Backend):
    """In-memory implementation of the memory storage backend."""

    def __init__(self):
        """Initializes the in-memory storage."""
        self.messages: list[Message] = []

    def add(self, message: Message):
        """Adds a message to the in-memory list."""
        self.messages.append(message)

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the in-memory list."""
        return self.messages

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in the in-memory list based on substring matching."""
        return [msg for msg in self.messages if query.lower() in msg.content.lower()][:search_limit]

    def is_empty(self) -> bool:
        """Checks if the in-memory list is empty."""
        return len(self.messages) == 0

    def clear(self):
        """Clears the in-memory list."""
        self.messages = []
