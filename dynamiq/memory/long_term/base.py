from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cached_property
from hashlib import md5
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import ForgetStatus, RememberOutcome
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger


class LongTermMemoryError(Exception):
    """Base exception for long-term memory operations."""

    pass


def _content_hash(user_id: str, content: str) -> str:
    """Per-user stable hash used only as a dedup key, never as a security primitive."""
    normalised = content.strip().lower()
    return md5(f"{user_id}:{normalised}".encode(), usedforsecurity=False).hexdigest()


class LongTermMemoryBackend(ABC, BaseModel):
    """Fact-shaped, user-scoped storage + embedding engine for long-term memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "long-term-memory-backend"
    id: str = Field(default_factory=generate_uuid)
    embedder: TextEmbedder = Field(
        ...,
        description="Text embedder used to vectorize facts on write and queries on read.",
    )
    upsert_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity threshold above which `remember()` upserts the nearest fact in place.",
    )

    _storage_ensured: bool = PrivateAttr(default=False)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {"embedder": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        data = self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)
        data["embedder"] = self.embedder.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        return data

    def _embed(self, text: str) -> list[float]:
        result = self.embedder.execute(input_data=TextEmbedderInputSchema(query=text))
        return list(result["embedding"])

    def _ensure_storage(self) -> None:
        """Provision tables / collections / indexes if absent. No-op by default."""

    def _guarded_ensure(self) -> None:
        """Call `_ensure_storage` at most once per instance; retry on failure."""
        if self._storage_ensured:
            return
        self._ensure_storage()
        self._storage_ensured = True

    def remember(
        self, *, content: str, user_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Fact, RememberOutcome]:
        """Add or upsert a fact for `user_id`. Returns the fact and a `RememberOutcome`."""
        if not content or not content.strip():
            raise LongTermMemoryError("Fact content cannot be empty")
        try:
            self._guarded_ensure()
            normalised = content.strip()
            content_hash = _content_hash(user_id, normalised)

            existing = self.get_by_hash(user_id=user_id, content_hash=content_hash)
            if existing is not None:
                logger.debug(f"LongTermMemory: exact-dedup hit for user={user_id}, fact {existing.id}")
                return existing, RememberOutcome.UNCHANGED

            embedding = self._embed(normalised)

            nearest = self.search(query_embedding=embedding, scope={"user_id": user_id}, limit=1)
            if nearest and nearest[0][1] >= self.upsert_threshold:
                old_fact, score = nearest[0]
                now = datetime.now(UTC)
                new_metadata = metadata if metadata is not None else old_fact.metadata
                self.update(
                    old_fact.id,
                    content=normalised,
                    content_hash=content_hash,
                    embedding=embedding,
                    metadata=new_metadata,
                    updated_at=now,
                )
                logger.debug(
                    f"LongTermMemory: upsert hit (score={score:.3f}) — updated fact {old_fact.id} for user={user_id}"
                )
                updated = old_fact.model_copy(
                    update={
                        "content": normalised,
                        "hash": content_hash,
                        "metadata": new_metadata,
                        "updated_at": now,
                    }
                )
                return updated, RememberOutcome.UPDATED

            now = datetime.now(UTC)
            fact = Fact(
                id=str(uuid4()),
                content=normalised,
                hash=content_hash,
                user_id=user_id,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            self.insert(fact, embedding)
            logger.debug(f"LongTermMemory: stored fact {fact.id} for user={user_id}")
            return fact, RememberOutcome.CREATED
        except Exception as e:
            logger.error(f"LongTermMemory.remember failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to remember fact: {e}") from e

    def recall(self, *, query: str, user_id: str, limit: int = 5) -> list[tuple[Fact, float]]:
        """Semantic search for facts relevant to `query`, scoped to `user_id`."""
        stripped = query.strip() if query else ""
        if not stripped:
            raise LongTermMemoryError("Recall query cannot be empty")
        if limit <= 0:
            return []
        try:
            self._guarded_ensure()
            embedding = self._embed(stripped)
            results = self.search(query_embedding=embedding, scope={"user_id": user_id}, limit=limit)
            logger.debug(f"LongTermMemory: recall for user={user_id} returned {len(results)} facts")
            return results
        except Exception as e:
            logger.error(f"LongTermMemory.recall failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to recall facts: {e}") from e

    def forget(self, *, fact_id: str, user_id: str) -> ForgetStatus:
        """Delete a fact by id, returning a `ForgetStatus`. Never raises on user mismatch."""
        try:
            self._guarded_ensure()
            fact = self.get(fact_id)
            if fact is None:
                return ForgetStatus.NOT_FOUND
            if fact.user_id != user_id:
                logger.warning(
                    f"LongTermMemory.forget: cross-user delete blocked "
                    f"(owner={fact.user_id}, caller={user_id}, fact={fact_id})"
                )
                return ForgetStatus.FORBIDDEN
            self.delete(fact_id)
            logger.debug(f"LongTermMemory: deleted fact {fact_id} for user={user_id}")
            return ForgetStatus.DELETED
        except Exception as e:
            logger.error(f"LongTermMemory.forget failed for fact={fact_id}, user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to forget fact: {e}") from e

    def list_all(self, *, user_id: str, limit: int = 100) -> list[Fact]:
        """Return up to `limit` facts for `user_id` (admin/introspection)."""
        if limit <= 0:
            return []
        try:
            self._guarded_ensure()
            return self.list_by_scope({"user_id": user_id}, limit=limit)
        except Exception as e:
            logger.error(f"LongTermMemory.list_all failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to list facts: {e}") from e

    def clear_user(self, *, user_id: str) -> int:
        """Hard-delete every fact owned by `user_id` and return the count deleted."""
        if not user_id:
            raise LongTermMemoryError("clear_user requires a non-empty user_id")
        try:
            self._guarded_ensure()
            deleted = self.delete_scope({"user_id": user_id})
            logger.debug(f"LongTermMemory: cleared {deleted} facts for user={user_id}")
            return deleted
        except Exception as e:
            logger.error(f"LongTermMemory.clear_user failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to clear user facts: {e}") from e

    @abstractmethod
    def insert(self, fact: Fact, embedding: list[float]) -> None:
        """Insert a new fact and its embedding."""

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
        metadata: dict[str, Any],
        updated_at: datetime,
    ) -> None:
        """Replace mutable fields of a fact in place; preserves id, user_id, created_at."""
