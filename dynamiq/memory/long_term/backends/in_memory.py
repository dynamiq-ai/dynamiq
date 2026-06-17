from datetime import datetime

import numpy as np
from pydantic import PrivateAttr

from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact


class InMemoryLongTermMemoryBackend(LongTermMemoryBackend):
    """Dict + numpy-cosine backend. Loses data on restart."""

    name: str = "in-memory-long-term-memory-backend"

    _facts: dict[str, Fact] = PrivateAttr(default_factory=dict)
    _vectors: dict[str, list[float]] = PrivateAttr(default_factory=dict)

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        self._facts[fact.id] = fact
        self._vectors[fact.id] = list(embedding)

    def get(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        for fact in self._facts.values():
            if fact.user_id == user_id and fact.hash == content_hash:
                return fact
        return None

    def delete(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)
        self._vectors.pop(fact_id, None)

    def update(
        self,
        fact_id: str,
        *,
        content: str,
        content_hash: str,
        embedding: list[float],
        metadata: dict,
        updated_at: datetime,
    ) -> None:
        existing = self._facts.get(fact_id)
        if existing is None:
            return
        self._facts[fact_id] = existing.model_copy(
            update={"content": content, "hash": content_hash, "metadata": metadata, "updated_at": updated_at}
        )
        self._vectors[fact_id] = list(embedding)

    def search(
        self, *, query_embedding: list[float],
        scope: dict[str, str], limit: int,
    ) -> list[tuple[Fact, float]]:
        if not self._facts:
            return []

        matched_facts = [f for f in self._facts.values() if _matches_scope(f, scope)]
        if not matched_facts:
            return []

        matrix = np.asarray([self._vectors[f.id] for f in matched_facts], dtype=np.float64)
        query = np.asarray(query_embedding, dtype=np.float64)

        # Cosine = (M @ q) / (||rows|| * ||q||); zero-norm rows fall back to 1
        # to avoid div-by-zero (the dot product is 0 anyway, so the score is 0).
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms[row_norms == 0] = 1.0
        query_norm = np.linalg.norm(query) or 1.0
        scores = (matrix @ query) / (row_norms * query_norm)

        k = min(limit, len(matched_facts))
        if k <= 0:
            return []
        # argpartition gives the top-k unsorted; sort just that slice.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(matched_facts[i], float(scores[i])) for i in top_idx]

    def list_by_scope(
        self, scope: dict[str, str], limit: int = 100,
    ) -> list[Fact]:
        matched = [f for f in self._facts.values() if _matches_scope(f, scope)]
        return matched[:limit]

    def delete_scope(self, scope: dict[str, str]) -> int:
        if not scope:
            raise ValueError("delete_scope requires a non-empty scope")
        to_delete = [fid for fid, f in self._facts.items() if _matches_scope(f, scope)]
        for fid in to_delete:
            self.delete(fid)
        return len(to_delete)


def _matches_scope(fact: Fact, scope: dict[str, str]) -> bool:
    for key, value in scope.items():
        if getattr(fact, key, None) != value:
            return False
    return True
