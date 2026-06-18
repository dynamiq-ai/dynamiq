import os
from datetime import UTC, datetime
from urllib.parse import urlparse

import pytest

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.long_term.backends.pgvector import PostgresLongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

DSN = os.getenv("POSTGRES_DSN")
pytestmark = pytest.mark.skipif(DSN is None, reason="POSTGRES_DSN not set")


def _connection_from_dsn(dsn: str) -> PostgreSQLConnection:
    parsed = urlparse(dsn)
    return PostgreSQLConnection(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/postgres").lstrip("/"),
        user=parsed.username or "postgres",
        password=parsed.password or "",
    )


@pytest.fixture
def backend(fake_embedder):
    b = PostgresLongTermMemoryBackend(
        connection=_connection_from_dsn(DSN),
        embedder=fake_embedder,
        table_name="test_user_facts",
        dimension=16,
    )
    b.recreate_table()
    yield b
    b.drop_table()


def _fact(fact_id, user_id, content, content_hash=None):
    now = datetime.now(UTC)
    return Fact(
        id=fact_id,
        content=content,
        hash=content_hash or f"h-{fact_id}",
        user_id=user_id,
        metadata={},
        created_at=now,
        updated_at=now,
    )


# --- insert / get / get_by_hash ---


def test_pgvector_insert_then_get(backend, fake_embedder):
    fact = _fact("f1", "u1", "hello")
    backend.insert(fact, fake_embedder.embed("hello"))
    fetched = backend.get("f1")
    assert fetched.id == "f1"
    assert fetched.content == "hello"
    assert fetched.user_id == "u1"


def test_pgvector_get_unknown_returns_none(backend):
    assert backend.get("does-not-exist") is None


def test_pgvector_get_by_hash(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    found = backend.get_by_hash(user_id="u1", content_hash="h-shared")
    assert found is not None and found.id == "f1"


def test_pgvector_get_by_hash_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    assert backend.get_by_hash(user_id="u2", content_hash="h-shared") is None


def test_pgvector_metadata_round_trip(backend, fake_embedder):
    fact = _fact("f1", "u1", "x")
    fact = fact.model_copy(update={"metadata": {"category": "preference", "score": 0.8}})
    backend.insert(fact, fake_embedder.embed("x"))
    fetched = backend.get("f1")
    assert fetched.metadata == {"category": "preference", "score": 0.8}


# --- delete / list_by_scope / delete_scope ---


def test_pgvector_delete(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x"), fake_embedder.embed("x"))
    backend.delete("f1")
    assert backend.get("f1") is None


def test_pgvector_delete_unknown_is_noop(backend):
    backend.delete("does-not-exist")


def test_pgvector_list_by_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    listed = backend.list_by_scope({"user_id": "u1"})
    assert {f.id for f in listed} == {"f1", "f2"}


def test_pgvector_delete_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    deleted = backend.delete_scope({"user_id": "u1"})
    assert deleted == 2
    assert backend.list_by_scope({"user_id": "u1"}) == []
    assert len(backend.list_by_scope({"user_id": "u2"})) == 1


# --- search ---


def test_pgvector_search_relevance_ordered(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u1", "alpha-2"), fake_embedder.embed("alpha-2"))
    backend.insert(_fact("f3", "u1", "zulu"), fake_embedder.embed("zulu"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=3,
    )
    assert hits[0][0].id == "f1"
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_pgvector_search_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u2", "alpha"), fake_embedder.embed("alpha"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert [f.id for f, _ in hits] == ["f1"]


def test_pgvector_search_empty_returns_empty(backend, fake_embedder):
    hits = backend.search(
        query_embedding=fake_embedder.embed("x"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert hits == []
