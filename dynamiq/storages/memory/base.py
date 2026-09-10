"""Base memory-store interface and common data structures.

A memory store holds what an agent has learned and wants to carry into later conversations: short
text notes addressed by path. It is deliberately *not* a ``FileStore``. That interface carries
`list_files_bytes`, extracted-text caching, content types and ``BytesIO`` — all of which exist for
agent file plumbing memory does not take part in, and each of which caused a bug when memory was
built on top of it.
"""

import abc
from datetime import datetime
from typing import Any, ClassVar, Hashable

from pydantic import BaseModel, ConfigDict, Field, computed_field


class MemoryStoreError(Exception):
    """Base exception for memory-store operations."""

    def __init__(self, message: str, operation: str | None = None, path: str | None = None):
        self.message = message
        self.operation = operation
        self.path = path
        super().__init__(self.message)


class MemoryNotFoundError(MemoryStoreError):
    """Raised when a memory does not exist."""


class MemoryPermissionError(MemoryStoreError):
    """Raised when a memory operation is not permitted."""


class MemoryEntry(BaseModel):
    """One memory, without its content."""

    path: str
    size: int = 0
    updated_at: datetime | None = None


class MemoryStore(abc.ABC, BaseModel):
    """Abstract base class for memory backends.

    Four operations over text addressed by path. There is no ``exists``: ``read`` raises when a path
    is absent and ``delete`` reports whether it removed anything, which covers every caller and
    keeps one endpoint out of the API.

    ``_clone_shared`` tells ``Node.clone()`` to share this instance by reference rather than
    deep-copying it, so parallel cloned tools reach the same store.
    """

    description: str | None = Field(
        default=None,
        description="What this memory holds. Shown to the model so it can tell memories apart.",
    )

    _clone_shared: ClassVar[bool] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def type(self) -> str:
        """Dotted path used to rebuild this store from a serialized workflow."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Parameters to drop when serializing."""
        return {}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the store to a dictionary."""
        kwargs.pop("for_tracing", None)
        kwargs.pop("include_secure_params", None)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        return data

    def describe_namespaces(self) -> dict[str, str]:
        """Path prefix -> what lives there.

        The empty key means everything this store holds. A composite overrides this to compose its
        routes' descriptions, so the caller never assembles a parallel structure by hand.
        """
        return {"": self.description} if self.description else {}

    def identity(self) -> Hashable:
        """What underlying store this addresses; two stores with the same identity hold the same data.

        A composite strips its route prefix before calling a store, so two routes pointing at one
        store would map different agent paths onto the same key and silently overwrite each other.
        This is how the composite tells them apart and refuses. Subclasses that address something
        remote should key on what identifies it there, not on the Python object.
        """
        return id(self)

    @abc.abstractmethod
    def list(self, prefix: str = "") -> list[MemoryEntry]:
        """List memories under ``prefix``. An empty prefix lists everything.

        Returns metadata only — read a path to get its content.
        """

    @abc.abstractmethod
    def read(self, path: str) -> str:
        """Return the content of one memory.

        Raises:
            MemoryNotFoundError: If the path does not exist.
        """

    @abc.abstractmethod
    def write(self, path: str, content: str) -> MemoryEntry:
        """Create or replace one memory, returning its entry."""

    @abc.abstractmethod
    def delete(self, path: str) -> bool:
        """Delete one memory. Returns False when it was not there."""


class MemoryStoreConfig(BaseModel):
    """Configuration for an agent's memory store.

    Attributes:
        enabled: Whether the agent gets a memory tool at all.
        backend: The store holding memories. A ``CompositeMemoryStore`` when there are several.
        write_enabled: Whether the agent may create, edit and delete memories.
    """

    enabled: bool = False
    backend: MemoryStore = Field(..., description="Store holding the agent's memories.")
    write_enabled: bool = Field(
        default=True,
        description="Whether the agent is permitted to write memories.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """The backend serializes itself, so exclude it from the plain dump."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the config to a dictionary, delegating the backend to its own ``to_dict``."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["backend"] = self.backend.to_dict(for_tracing=for_tracing, include_secure_params=include_secure_params)
        return data


def render_namespaces(namespaces: dict[str, str]) -> str:
    """One line per memory, for the tool description and the prompt.

    Empty when nothing is described, in which case the surrounding wording already covers a single
    unnamed memory.
    """
    if not namespaces:
        return ""
    return "\n".join(f"- {prefix} - {text}" if prefix else f"- {text}" for prefix, text in namespaces.items() if text)
