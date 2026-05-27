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

    def search(
        self, *, query_embedding: list[float],
        scope: dict[str, str], limit: int,
    ) -> list[tuple[Fact, float]]:
        if not self._facts:
            return []

        query = np.asarray(query_embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query) or 1.0

        scored: list[tuple[Fact, float]] = []
        for fact_id, fact in self._facts.items():
            if not _matches_scope(fact, scope):
                continue
            vec = np.asarray(self._vectors[fact_id], dtype=np.float64)
            vec_norm = np.linalg.norm(vec) or 1.0
            cosine = float(np.dot(query, vec) / (query_norm * vec_norm))
            scored.append((fact, cosine))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    def list_by_scope(
        self, scope: dict[str, str], limit: int = 100,
    ) -> list[Fact]:
        matched = [f for f in self._facts.values() if _matches_scope(f, scope)]
        return matched[:limit]

    def delete_scope(self, scope: dict[str, str]) -> int:
        to_delete = [fid for fid, f in self._facts.items() if _matches_scope(f, scope)]
        for fid in to_delete:
            self.delete(fid)
        return len(to_delete)


def _matches_scope(fact: Fact, scope: dict[str, str]) -> bool:
    for key, value in scope.items():
        if getattr(fact, key, None) != value:
            return False
    return True
