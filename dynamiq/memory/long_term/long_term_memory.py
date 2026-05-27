from datetime import UTC, datetime
from functools import cached_property
from hashlib import md5
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.memory.long_term.backends import InMemoryLongTermMemoryBackend
from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import ForgetStatus, MemoryToolKind
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema
from dynamiq.utils.logger import logger


class LongTermMemoryError(Exception):
    """Base exception for `LongTermMemory` operations."""

    pass


def _content_hash(user_id: str, content: str) -> str:
    """Per-user stable hash used only as a dedup key, never as a security primitive."""
    normalised = content.strip().lower()
    return md5(f"{user_id}:{normalised}".encode(), usedforsecurity=False).hexdigest()


def _embed(embedder: TextEmbedder, text: str) -> list[float]:
    result = embedder.execute(input_data=TextEmbedderInputSchema(query=text))
    return list(result["embedding"])


class LongTermMemory(BaseModel):
    """Tool-driven, user-scoped, fact-shaped memory that persists across sessions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: LongTermMemoryBackend = Field(
        default_factory=InMemoryLongTermMemoryBackend,
        description="Backend storage implementation for facts and their embeddings.",
    )
    embedder: TextEmbedder = Field(
        ...,
        description="Text embedder used to vectorize facts on write and queries on read.",
    )

    @computed_field
    @cached_property
    def type(self) -> str:
        """Fully-qualified class id used by the YAML loader for reconstruction."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Fields excluded from default model_dump; re-added by `to_dict`."""
        return {"backend": True, "embedder": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Serialize so backend and embedder round-trip via their own `to_dict`."""
        for_tracing = kwargs.pop("for_tracing", False)
        data = self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)
        data["backend"] = self.backend.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        data["embedder"] = self.embedder.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        return data

    def remember(
        self, *, content: str, user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Fact:
        """Add a fact for `user_id`. Idempotent on the (user_id, normalised content) pair.

        Args:
            content: The fact text.
            user_id: Owner of the fact. Required.
            metadata: Optional free-form metadata stored alongside the fact.

        Raises:
            LongTermMemoryError: If content is empty or storage fails.
        """
        if not content or not content.strip():
            raise LongTermMemoryError("Fact content cannot be empty")
        try:
            normalised = content.strip()
            content_hash = _content_hash(user_id, normalised)

            existing = self.backend.get_by_hash(user_id=user_id, content_hash=content_hash)
            if existing is not None:
                logger.debug(f"LongTermMemory: dedup hit for user={user_id}, returning existing fact {existing.id}")
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
            logger.debug(f"LongTermMemory: stored fact {fact.id} for user={user_id}")
            return fact
        except Exception as e:
            logger.error(f"LongTermMemory.remember failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to remember fact: {e}") from e

    def recall(
        self, *, query: str, user_id: str, limit: int = 5,
    ) -> list[tuple[Fact, float]]:
        """Semantic search for facts relevant to `query`, scoped to `user_id`.

        Args:
            query: Natural-language query string.
            user_id: Owner whose facts to search.
            limit: Maximum number of (fact, score) tuples to return.

        Raises:
            LongTermMemoryError: If the query is empty or search fails.
        """
        stripped = query.strip() if query else ""
        if not stripped:
            raise LongTermMemoryError("Recall query cannot be empty")
        try:
            embedding = _embed(self.embedder, stripped)
            results = self.backend.search(
                query_embedding=embedding,
                scope={"user_id": user_id},
                limit=limit,
            )
            logger.debug(f"LongTermMemory: recall for user={user_id} returned {len(results)} facts")
            return results
        except Exception as e:
            logger.error(f"LongTermMemory.recall failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to recall facts: {e}") from e

    def forget(self, *, fact_id: str, user_id: str) -> ForgetStatus:
        """Delete a fact by id, returning a `ForgetStatus`.

        Never raises on user mismatch — defence in depth above the
        construction-time `user_id` binding on the tool.

        Raises:
            LongTermMemoryError: If the storage delete fails for any other reason.
        """
        try:
            fact = self.backend.get(fact_id)
            if fact is None:
                return ForgetStatus.NOT_FOUND
            if fact.user_id != user_id:
                logger.warning(
                    f"LongTermMemory.forget: cross-user delete blocked "
                    f"(owner={fact.user_id}, caller={user_id}, fact={fact_id})"
                )
                return ForgetStatus.FORBIDDEN
            self.backend.delete(fact_id)
            logger.debug(f"LongTermMemory: deleted fact {fact_id} for user={user_id}")
            return ForgetStatus.DELETED
        except Exception as e:
            logger.error(f"LongTermMemory.forget failed for fact={fact_id}, user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to forget fact: {e}") from e

    def list_all(self, *, user_id: str, limit: int = 100) -> list[Fact]:
        """Return up to `limit` facts for `user_id`, most recent first (admin/introspection)."""
        try:
            return self.backend.list_by_scope({"user_id": user_id}, limit=limit)
        except Exception as e:
            logger.error(f"LongTermMemory.list_all failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to list facts: {e}") from e

    def get(self, fact_id: str) -> Fact | None:
        """Fetch a fact by id, or `None` if it does not exist."""
        try:
            return self.backend.get(fact_id)
        except Exception as e:
            logger.error(f"LongTermMemory.get failed for fact={fact_id}: {e}")
            raise LongTermMemoryError(f"Failed to fetch fact: {e}") from e

    def clear_user(self, *, user_id: str) -> int:
        """Hard-delete every fact owned by `user_id` and return the count deleted."""
        try:
            deleted = self.backend.delete_scope({"user_id": user_id})
            logger.debug(f"LongTermMemory: cleared {deleted} facts for user={user_id}")
            return deleted
        except Exception as e:
            logger.error(f"LongTermMemory.clear_user failed for user={user_id}: {e}")
            raise LongTermMemoryError(f"Failed to clear user facts: {e}") from e


class LongTermMemoryConfig(BaseModel):
    """Per-agent configuration for long-term memory tool exposure."""

    tools: tuple[MemoryToolKind, ...] = (
        MemoryToolKind.REMEMBER,
        MemoryToolKind.RECALL,
        MemoryToolKind.FORGET,
    )
