from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.prompts import Message
from dynamiq.utils import generate_uuid


class MemoryBackend(ABC, BaseModel):
    """Abstract base class for memory storage backends."""

    name: str = "MemoryBackend"
    id: str = Field(default_factory=generate_uuid)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        kwargs.pop("include_secure_params", None)
        return self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)

    @computed_field
    @cached_property
    def type(self) -> str:
        """Returns the backend type as a string."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @abstractmethod
    def add(self, message: Message) -> None:
        """
        Adds a message to the memory storage.

        Args:
            message: Message to add to storage

        Raises:
            MemoryBackendError: If the message cannot be added
        """
        raise NotImplementedError

    @abstractmethod
    def get_all(self, limit: int | None = None) -> list[Message]:
        """
        Retrieves all messages from the memory storage, optionally limited.

        Args:
            limit: Maximum number of messages to return. If provided, returns the most recent messages.
                  If None, uses the backend's default limit (if applicable).

        Returns:
            List of messages sorted by timestamp (oldest first)

        Raises:
            MemoryBackendError: If messages cannot be retrieved
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """
        Searches for messages relevant to the query.

        Args:
            query: Search query string (optional)
            filters: Optional metadata filters to apply
            limit: Maximum number of messages to return. If None, uses the backend's default limit.

        Returns:
            List of messages sorted by relevance (most relevant first)

        Raises:
            MemoryBackendError: If search operation fails
        """
        raise NotImplementedError

    @abstractmethod
    def is_empty(self) -> bool:
        """
        Checks if the memory storage is empty.

        Returns:
            True if the memory is empty, False otherwise

        Raises:
            MemoryBackendError: If the check fails
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Clears the memory storage.

        Raises:
            MemoryBackendError: If the memory cannot be cleared
        """
        raise NotImplementedError

    def _prepare_filters(self, filters: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """
        Default implementation for preparing filters. Override in backend-specific implementations.

        Args:
            filters: Raw filters to prepare

        Returns:
            Prepared filters in backend-specific format
        """
        return filters
