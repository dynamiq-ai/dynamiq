"""Tests for InMemoryLongTermMemoryBackend storage primitives."""
from datetime import UTC, datetime, timedelta

import pytest

from dynamiq.memory.long_term.schemas import Fact


def _fact(fact_id: str, user_id: str, content: str,
          content_hash: str | None = None) -> Fact:
    now = datetime.now(UTC)
    return Fact(
        id=fact_id, content=content,
        hash=content_hash or f"h-{fact_id}",
        user_id=user_id, metadata={},
        created_at=now, updated_at=now,
    )


# --- insert / get / get_by_hash ---

def test_insert_then_get(backend, fake_embedder):
    fact = _fact("f1", "u1", "hello")
    backend.insert(fact, fake_embedder.embed("hello"))
    assert backend.get("f1") == fact


def test_get_unknown_returns_none(backend):
    assert backend.get("does-not-exist") is None


def test_get_by_hash_returns_match(backend, fake_embedder):
    fact = _fact("f1", "u1", "hello", content_hash="h-shared")
    backend.insert(fact, fake_embedder.embed("hello"))
    assert backend.get_by_hash(user_id="u1", content_hash="h-shared") == fact


def test_get_by_hash_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "hello", "h-shared"), fake_embedder.embed("hello"))
    assert backend.get_by_hash(user_id="u2", content_hash="h-shared") is None


def test_get_by_hash_unknown_returns_none(backend):
    assert backend.get_by_hash(user_id="u1", content_hash="nope") is None


# --- search ---

def test_search_returns_relevance_ordered(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u1", "alpha-2"), fake_embedder.embed("alpha-2"))
    backend.insert(_fact("f3", "u1", "zulu"), fake_embedder.embed("zulu"))

    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=3,
    )
    assert hits[0][0].id == "f1"
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_filters_by_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u2", "alpha"), fake_embedder.embed("alpha"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=10,
    )
    assert [f.id for f, _ in hits] == ["f1"]


def test_search_respects_limit(backend, fake_embedder):
    for i in range(5):
        backend.insert(_fact(f"f{i}", "u1", f"text{i}"), fake_embedder.embed(f"text{i}"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("text0"),
        scope={"user_id": "u1"}, limit=2,
    )
    assert len(hits) == 2


def test_search_empty_store_returns_empty(backend, fake_embedder):
    hits = backend.search(
        query_embedding=fake_embedder.embed("anything"),
        scope={"user_id": "u1"}, limit=5,
    )
    assert hits == []


def test_search_with_zero_limit_returns_empty(backend, fake_embedder):
    """`limit=0` must return [] without hitting argpartition (which would be
    called with `kth=-1` and only happen to work by accident)."""
    for i in range(3):
        backend.insert(_fact(f"f{i}", "u1", f"text{i}"), fake_embedder.embed(f"text{i}"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("text0"),
        scope={"user_id": "u1"}, limit=0,
    )
    assert hits == []


# --- delete / list_by_scope / delete_scope ---

def test_delete_removes_fact(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x"), fake_embedder.embed("x"))
    backend.delete("f1")
    assert backend.get("f1") is None


def test_delete_unknown_is_noop(backend):
    backend.delete("does-not-exist")     # must not raise


def test_list_by_scope_returns_in_scope_facts(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    listed = backend.list_by_scope({"user_id": "u1"})
    assert {f.id for f in listed} == {"f1", "f2"}


def test_list_by_scope_respects_limit(backend, fake_embedder):
    for i in range(5):
        backend.insert(_fact(f"f{i}", "u1", f"x{i}"), fake_embedder.embed(f"x{i}"))
    assert len(backend.list_by_scope({"user_id": "u1"}, limit=2)) == 2


# --- update ---


def test_update_replaces_content_hash_embedding_and_timestamp(backend, fake_embedder):
    original = _fact("f1", "u1", "hello", content_hash="h-old")
    backend.insert(original, fake_embedder.embed("hello"))

    new_time = original.updated_at + timedelta(seconds=5)
    backend.update(
        "f1",
        content="hello world",
        content_hash="h-new",
        embedding=fake_embedder.embed("hello world"),
        metadata={"category": "greeting"},
        updated_at=new_time,
    )

    updated = backend.get("f1")
    assert updated.content == "hello world"
    assert updated.hash == "h-new"
    assert updated.metadata == {"category": "greeting"}
    assert updated.updated_at == new_time
    assert updated.id == original.id
    assert updated.created_at == original.created_at

    hits = backend.search(
        query_embedding=fake_embedder.embed("hello world"),
        scope={"user_id": "u1"},
        limit=1,
    )
    assert hits[0][0].content == "hello world"


def test_update_unknown_is_noop(backend, fake_embedder):
    backend.update(
        "does-not-exist",
        content="x",
        content_hash="h",
        embedding=fake_embedder.embed("x"),
        metadata={},
        updated_at=datetime.now(UTC),
    )  # must not raise
    assert backend.get("does-not-exist") is None


def test_delete_scope_removes_all_in_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    deleted = backend.delete_scope({"user_id": "u1"})
    assert deleted == 2
    assert backend.list_by_scope({"user_id": "u1"}) == []
    assert len(backend.list_by_scope({"user_id": "u2"})) == 1


def test_delete_scope_rejects_empty_scope(backend, fake_embedder):
    """Empty scope is rejected to prevent accidental whole-store wipes — callers
    must opt in via an explicit scope like `{"user_id": ...}`."""
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    with pytest.raises(ValueError, match="non-empty scope"):
        backend.delete_scope({})
    assert len(backend.list_by_scope({"user_id": "u1"})) == 1
