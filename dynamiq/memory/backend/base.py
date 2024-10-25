from abc import ABC, abstractmethod

from dynamiq.prompts import Message


# TODO: Vector stores and backend consolidation
class MemoryBackend(ABC):
    """Abstract base class for memory storage backends."""

    name = "MemoryBackend"

    @abstractmethod
    def add(self, message: Message):
        """Adds a message to the memory storage."""
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> list[Message]:
        """Retrieves all messages from the memory storage."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int) -> list[Message]:
        """Searches for messages relevant to the query."""
        raise NotImplementedError

    @abstractmethod
    def is_empty(self) -> bool:
        """Checks if the memory storage is empty."""
        raise NotImplementedError

    @abstractmethod
    def clear(self):
        """Clears the memory storage."""
        raise NotImplementedError
