from abc import ABC, abstractmethod
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.prompts import Message
from dynamiq.utils import generate_uuid


# TODO: Vector stores and backend consolidation
class MemoryBackend(ABC, BaseModel):
    """Abstract base class for memory storage backends."""

    name: str = "MemoryBackend"
    id: str = Field(default_factory=generate_uuid)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self):
        """Define parameters to exclude during serialization."""
        return {}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        kwargs.pop("include_secure_params", None)
        return self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

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
