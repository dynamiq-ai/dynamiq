"""Base memory-store interface and common data structures.

A memory store holds what an agent has learned and wants to carry into later conversations: short
text notes addressed by path. It is deliberately *not* a ``FileStore``. That interface carries
`list_files_bytes`, extracted-text caching, content types and ``BytesIO`` — all of which exist for
agent file plumbing memory does not take part in, and each of which caused a bug when memory was
built on top of it.
"""

import abc
import builtins
import re
from datetime import datetime
from typing import Any, ClassVar, Hashable

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

_WORD = re.compile(r"\w+")


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
    """One memory, without its content.

    ``description`` is what the store reports the memory holds: its ``description`` property, else its
    first line of prose. A store that reports none leaves it empty.
    """

    path: str
    size: int = 0
    updated_at: datetime | None = None
    description: str = ""

    @field_validator("description", mode="before")
    @classmethod
    def _absent_is_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class MemoryHit(MemoryEntry):
    """A memory that matches a search, with the passage that matched."""

    snippet: str = ""
    score: float = 0.0


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

    def scope_of(self, path: str) -> str:
        """The prefix of the memory that holds ``path``; links between memories resolve inside it."""
        return ""

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

    def content_limit(self) -> int | None:
        """The most characters one memory may hold, or None when the store sets no limit."""
        return None

    @abc.abstractmethod
    def list(self, prefix: str = "", user_id: str | None = None) -> builtins.list[MemoryEntry]:
        """List memories under ``prefix``. An empty prefix lists everything.

        Returns metadata only — read a path to get its content. ``user_id`` is the end user this
        call is for; it overrides whatever default the store was configured with.
        """

    @abc.abstractmethod
    def read(self, path: str, user_id: str | None = None) -> str:
        """Return the content of one memory.

        Raises:
            MemoryNotFoundError: If the path does not exist.
        """

    @abc.abstractmethod
    def write(self, path: str, content: str, user_id: str | None = None) -> MemoryEntry:
        """Create or replace one memory, returning its entry."""

    @abc.abstractmethod
    def delete(self, path: str, user_id: str | None = None) -> bool:
        """Delete one memory. Returns False when it was not there."""

    def search(
        self, query: str, prefix: str = "", limit: int = 10, user_id: str | None = None
    ) -> builtins.list[MemoryHit]:
        """Memories under ``prefix`` that hold the words of ``query``, best first.

        The default reads every memory and counts the query's words in each, which is fine for the few
        hundred short notes a memory holds; a store with an index of its own overrides it.
        """
        words = {word for word in _WORD.findall(query.lower()) if len(word) > 2}
        if not words:
            return []
        hits: builtins.list[MemoryHit] = []
        for entry in self.list(prefix, user_id):
            try:
                text = f"{entry.path} {self.read(entry.path, user_id)}".lower()
            except MemoryNotFoundError:
                continue
            score = sum(1 for word in words if word in text) / len(words)
            if score:
                hits.append(MemoryHit(**entry.model_dump(), score=score))
        hits.sort(key=lambda hit: (-hit.score, hit.path))
        return hits[:limit]


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
