from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from dynamiq.memory.long_term.schemas import Fact


class LongTermMemoryBackend(ABC, BaseModel):
    """Fact-shaped, scope-filtered storage backend for `LongTermMemory`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def insert(self, fact: Fact, embedding: list[float]) -> None:
        """Insert a new fact. Caller has already deduped via `get_by_hash`."""

    @abstractmethod
    def get(self, fact_id: str) -> Fact | None: ...

    @abstractmethod
    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        """Exact-content dedup gate. Returns the existing Fact or None."""

    @abstractmethod
    def delete(self, fact_id: str) -> None:
        """Hard-delete a single fact. No-op if not present."""

    @abstractmethod
    def search(
        self, *, query_embedding: list[float],
        scope: dict[str, str], limit: int,
    ) -> list[tuple[Fact, float]]:
        """Vector similarity search filtered by scope. Returns (fact, score) tuples,
        most relevant first. No threshold filtering — caller decides."""

    @abstractmethod
    def list_by_scope(
        self, scope: dict[str, str], limit: int = 100,
    ) -> list[Fact]:
        """Non-semantic listing for admin / introspection."""

    @abstractmethod
    def delete_scope(self, scope: dict[str, str]) -> int:
        """Hard-delete every fact matching `scope`. Returns count deleted."""

    def update(self, fact_id: str, content: str, embedding: list[float]) -> None:
        """Replace an existing fact in-place (Phase 2). Use delete + insert in v1."""
        raise NotImplementedError(
            "update() lands in Phase 2 with the auto-extractor. "
            "In v1, correct a fact via delete() + insert()."
        )
