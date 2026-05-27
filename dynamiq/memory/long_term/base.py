from abc import ABC, abstractmethod
from datetime import datetime
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.memory.long_term.schemas import Fact
from dynamiq.utils import generate_uuid


class LongTermMemoryBackend(ABC, BaseModel):
    """Fact-shaped, scope-filtered storage backend for `LongTermMemory`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "long-term-memory-backend"
    id: str = Field(default_factory=generate_uuid)

    @computed_field
    @cached_property
    def type(self) -> str:
        """Fully-qualified class id used by the YAML loader for polymorphic reconstruction."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Field names to exclude from serialization (overridden by subclasses)."""
        return {}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Serialize the backend to a dict for workflow YAML round-trip."""
        kwargs.pop("include_secure_params", None)
        kwargs.pop("for_tracing", None)
        return self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)

    @abstractmethod
    def insert(self, fact: Fact, embedding: list[float]) -> None:
        """Insert a new fact and its embedding. Caller has already deduped via `get_by_hash`."""

    @abstractmethod
    def get(self, fact_id: str) -> Fact | None:
        """Fetch a fact by id, or `None` if it does not exist."""

    @abstractmethod
    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        """Fetch the fact matching `(user_id, content_hash)`, or `None`."""

    @abstractmethod
    def delete(self, fact_id: str) -> None:
        """Hard-delete a single fact. No-op if not present."""

    @abstractmethod
    def search(
        self, *, query_embedding: list[float],
        scope: dict[str, str], limit: int,
    ) -> list[tuple[Fact, float]]:
        """Return up to `limit` `(fact, score)` tuples matching `scope`, most relevant first."""

    @abstractmethod
    def list_by_scope(
        self, scope: dict[str, str], limit: int = 100,
    ) -> list[Fact]:
        """Return up to `limit` facts matching `scope`, non-semantically."""

    @abstractmethod
    def delete_scope(self, scope: dict[str, str]) -> int:
        """Hard-delete every fact matching `scope` and return the count deleted."""

    @abstractmethod
    def update(
        self,
        fact_id: str,
        *,
        content: str,
        content_hash: str,
        embedding: list[float],
        updated_at: datetime,
    ) -> None:
        """Replace content/hash/embedding/updated_at for an existing fact in place.

        Preserves `id`, `user_id`, `metadata`, and `created_at`. Used by the
        semantic-upsert path in `LongTermMemory.remember`.
        """
