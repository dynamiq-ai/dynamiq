"""Tests for QdrantLongTermMemoryBackend against qdrant-client's in-memory mode. No live Qdrant calls are made."""

from datetime import datetime, timezone

import pytest
from qdrant_client import QdrantClient

from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.memory.long_term.backends.qdrant import QdrantLongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact


@pytest.fixture
def backend(fake_embedder, monkeypatch):
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(QdrantConnection, "connect", lambda self: client)
    backend = QdrantLongTermMemoryBackend(
        connection=QdrantConnection(url="http://localhost:6333", api_key="test-key"),
        embedder=fake_embedder,
        dimension=fake_embedder.DIM,
    )
    backend.ensure_collection()
    return backend


def _fact(fact_id: str, user_id: str, content: str) -> Fact:
    now = datetime.now(timezone.utc)
    return Fact(id=fact_id, content=content, hash=f"h-{fact_id}", user_id=user_id, created_at=now, updated_at=now)


def test_qdrant_search_ranks_facts_in_scope(backend, fake_embedder):
    for fact_id, user_id, content in [
        ("f1", "u1", "likes tea"),
        ("f2", "u1", "plays chess"),
        ("f3", "u2", "likes tea"),
    ]:
        backend.insert(_fact(fact_id, user_id, content), fake_embedder.embed(content))

    results = backend.search(query_embedding=fake_embedder.embed("likes tea"), scope={"user_id": "u1"}, limit=5)

    assert [fact.id for fact, _ in results] == ["f1", "f2"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[0][1] > results[1][1]


def test_qdrant_search_honours_limit(backend, fake_embedder):
    for fact_id in ("f1", "f2", "f3"):
        backend.insert(_fact(fact_id, "u1", fact_id), fake_embedder.embed(fact_id))

    assert len(backend.search(query_embedding=fake_embedder.embed("f1"), scope={"user_id": "u1"}, limit=2)) == 2
    assert backend.search(query_embedding=fake_embedder.embed("f1"), scope={"user_id": "u1"}, limit=0) == []


def test_qdrant_insert_get_and_delete_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "likes tea"), fake_embedder.embed("likes tea"))
    backend.insert(_fact("f2", "u2", "plays chess"), fake_embedder.embed("plays chess"))

    assert backend.get("f1").content == "likes tea"
    assert backend.get_by_hash(user_id="u1", content_hash="h-f1").id == "f1"
    assert backend.delete_scope({"user_id": "u1"}) == 1
    assert backend.get("f1") is None
    assert backend.get("f2") is not None
