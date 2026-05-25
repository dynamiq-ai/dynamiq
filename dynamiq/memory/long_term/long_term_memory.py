from datetime import UTC, datetime
from hashlib import md5
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact


def _content_hash(user_id: str, content: str) -> str:
    """Per-user stable hash used only as a dedup key, never as a security primitive."""
    normalised = content.strip().lower()
    return md5(f"{user_id}:{normalised}".encode(), usedforsecurity=False).hexdigest()


def _embed(embedder: Any, text: str) -> list[float]:
    result = embedder.execute({"query": text})
    return list(result["embedding"])


class LongTermMemory(BaseModel):
    """Tool-driven, user-scoped, fact-shaped memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: LongTermMemoryBackend
    embedder: Any

    def remember(
        self, *, content: str, user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add a fact. Idempotent on (user_id, normalised content)."""
        if not content or not content.strip():
            raise ValueError("Fact content cannot be empty")

        normalised = content.strip()
        content_hash = _content_hash(user_id, normalised)

        existing = self.backend.get_by_hash(user_id=user_id, content_hash=content_hash)
        if existing is not None:
            return existing

        now = datetime.now(UTC)
        embedding = _embed(self.embedder, normalised)
        fact = Fact(
            id=str(uuid4()),
            content=normalised,
            hash=content_hash,
            user_id=user_id,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.backend.insert(fact, embedding)
        return fact

    def recall(
        self, *, query: str, user_id: str, limit: int = 5,
    ) -> list[tuple[Fact, float]]:
        """Semantic search for facts relevant to `query`, scoped to `user_id`."""
        stripped = query.strip() if query else ""
        if not stripped:
            raise ValueError("recall query cannot be empty")
        embedding = _embed(self.embedder, stripped)
        return self.backend.search(
            query_embedding=embedding,
            scope={"user_id": user_id},
            limit=limit,
        )

    def forget(self, *, fact_id: str, user_id: str) -> str:
        """Delete a fact by id, returning 'deleted' | 'not_found' | 'forbidden'.

        Never raises on user mismatch — defence in depth above the
        construction-time `user_id` binding on the tool.
        """
        fact = self.backend.get(fact_id)
        if fact is None:
            return "not_found"
        if fact.user_id != user_id:
            return "forbidden"
        self.backend.delete(fact_id)
        return "deleted"

    def list_all(self, *, user_id: str, limit: int = 100) -> list[Fact]:
        """Return up to `limit` facts for `user_id`, most recent first (admin/introspection)."""
        return self.backend.list_by_scope({"user_id": user_id}, limit=limit)

    def get(self, fact_id: str) -> Fact | None:
        """Fetch a fact by id, or `None` if it does not exist."""
        return self.backend.get(fact_id)

    def clear_user(self, *, user_id: str) -> int:
        """Hard-delete every fact owned by `user_id` and return the count deleted."""
        return self.backend.delete_scope({"user_id": user_id})


class LongTermMemoryConfig(BaseModel):
    """Per-agent configuration for long-term memory tool exposure."""

    tools: tuple[Literal["remember", "recall", "forget"], ...] = (
        "remember",
        "recall",
        "forget",
    )
