"""Tests for PineconeLongTermMemoryBackend.

These exercise the backend's storage primitives against an in-process fake
Pinecone index that mimics the v3 client API used by the backend
(`upsert`, `fetch`, `query`, `delete`). No live Pinecone calls are made.
"""
import math
from datetime import UTC, datetime

import pytest

from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.memory.long_term.backends.pinecone import PineconeLongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

# --- Fake Pinecone client / index ------------------------------------------

# Pinecone metadata filter is MongoDB-style. We support the subset the backend
# emits: `{key: {"$eq": value}}` and `{"$and": [..., ...]}`.


def _matches_filter(metadata: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    if "$and" in flt:
        return all(_matches_filter(metadata, sub) for sub in flt["$and"])
    for key, predicate in flt.items():
        if isinstance(predicate, dict) and "$eq" in predicate:
            if metadata.get(key) != predicate["$eq"]:
                return False
        else:
            if metadata.get(key) != predicate:
                return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class _FakeIndex:
    def __init__(self) -> None:
        # namespace -> id -> {"id", "values", "metadata"}
        self.store: dict[str, dict[str, dict]] = {}

    def _ns(self, namespace: str) -> dict[str, dict]:
        return self.store.setdefault(namespace, {})

    def upsert(self, vectors, namespace="default"):
        ns = self._ns(namespace)
        for vec in vectors:
            ns[vec["id"]] = {"id": vec["id"], "values": list(vec["values"]), "metadata": dict(vec["metadata"])}
        return {"upserted_count": len(vectors)}

    def fetch(self, ids, namespace="default"):
        ns = self._ns(namespace)
        return {"vectors": {i: ns[i] for i in ids if i in ns}}

    def delete(self, ids=None, namespace="default", filter=None):
        ns = self._ns(namespace)
        if ids is not None:
            for i in ids:
                ns.pop(i, None)
        elif filter is not None:
            for i, item in list(ns.items()):
                if _matches_filter(item["metadata"], filter):
                    ns.pop(i, None)
        return {}

    def query(self, vector, top_k, namespace="default", filter=None, include_metadata=True, **_):
        ns = self._ns(namespace)
        candidates = [item for item in ns.values() if _matches_filter(item["metadata"], filter)]
        scored = [(item, _cosine(vector, item["values"])) for item in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        matches = []
        for item, score in scored[:top_k]:
            entry = {"id": item["id"], "score": score}
            if include_metadata:
                entry["metadata"] = item["metadata"]
            matches.append(entry)
        return {"matches": matches}


class _FakeClient:
    def __init__(self) -> None:
        self.indexes: dict[str, _FakeIndex] = {}

    def Index(self, name):  # noqa: N802 — mirrors Pinecone client API
        return self.indexes.setdefault(name, _FakeIndex())


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_pinecone_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(PineconeConnection, "connect", lambda self: client)
    return client


@pytest.fixture
def backend(fake_embedder, fake_pinecone_client):
    return PineconeLongTermMemoryBackend(
        connection=PineconeConnection(api_key="test-key"),
        embedder=fake_embedder,
        index_name="user_facts",
        namespace="test",
        dimension=fake_embedder.DIM,
    )


def _fact(fact_id: str, user_id: str, content: str, content_hash: str | None = None) -> Fact:
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


# --- insert / get / get_by_hash --------------------------------------------


def test_pinecone_insert_then_get(backend, fake_embedder):
    fact = _fact("f1", "u1", "hello")
    backend.insert(fact, fake_embedder.embed("hello"))
    fetched = backend.get("f1")
    assert fetched is not None and fetched.id == "f1" and fetched.content == "hello"


def test_pinecone_get_unknown_returns_none(backend):
    assert backend.get("does-not-exist") is None


def test_pinecone_get_by_hash(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    found = backend.get_by_hash(user_id="u1", content_hash="h-shared")
    assert found is not None and found.id == "f1"


def test_pinecone_get_by_hash_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    assert backend.get_by_hash(user_id="u2", content_hash="h-shared") is None


def test_pinecone_metadata_round_trip(backend, fake_embedder):
    """Free-form metadata must survive Pinecone's flat-schema constraint via JSON encoding."""
    fact = _fact("f1", "u1", "x").model_copy(
        update={"metadata": {"category": "preference", "score": 0.8}}
    )
    backend.insert(fact, fake_embedder.embed("x"))
    fetched = backend.get("f1")
    assert fetched.metadata == {"category": "preference", "score": 0.8}


# --- delete / list_by_scope / delete_scope ---------------------------------


def test_pinecone_delete(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x"), fake_embedder.embed("x"))
    backend.delete("f1")
    assert backend.get("f1") is None


def test_pinecone_list_by_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    listed = backend.list_by_scope({"user_id": "u1"})
    assert {f.id for f in listed} == {"f1", "f2"}


def test_pinecone_list_by_scope_zero_limit_returns_empty(backend, fake_embedder):
    """Pinecone's `top_k` is required to be >= 1, so `limit=0` cannot be
    expressed as a query — short-circuit so callers see the same empty result
    they'd get from in-memory / other backends."""
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    assert backend.list_by_scope({"user_id": "u1"}, limit=0) == []
    assert backend.list_by_scope({"user_id": "u1"}, limit=-3) == []


def test_pinecone_delete_scope_returns_accurate_count(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    assert backend.delete_scope({"user_id": "u1"}) == 2
    assert backend.list_by_scope({"user_id": "u1"}) == []
    assert len(backend.list_by_scope({"user_id": "u2"})) == 1


def test_pinecone_delete_scope_empty_returns_zero(backend):
    assert backend.delete_scope({"user_id": "nobody"}) == 0


def test_pinecone_delete_scope_paginates_beyond_single_page(backend, fake_embedder, monkeypatch):
    """clear_user on users with more facts than fit in one query page must still
    delete everything and report the true count — not silently cap at one page."""
    monkeypatch.setattr(backend, "_LIST_PAGE_SIZE", 2)
    for i in range(5):
        backend.insert(_fact(f"f{i}", "u1", f"c{i}"), fake_embedder.embed(f"c{i}"))
    assert backend.delete_scope({"user_id": "u1"}) == 5
    assert backend.list_by_scope({"user_id": "u1"}) == []


# --- search ----------------------------------------------------------------


def test_pinecone_search_relevance_ordered(backend, fake_embedder):
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


def test_pinecone_search_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u2", "alpha"), fake_embedder.embed("alpha"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert [fact.id for fact, _ in hits] == ["f1"]


def test_pinecone_search_empty_returns_empty(backend, fake_embedder):
    hits = backend.search(
        query_embedding=fake_embedder.embed("x"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert hits == []


# --- high-level operations via Template Method ------------------------------


def test_pinecone_remember_and_recall_through_backend(backend, fake_embedder):
    """End-to-end remember/recall must work through the backend's high-level API,
    confirming the storage primitives are wired correctly."""
    backend.remember(content="User likes pizza", user_id="u1")
    backend.remember(content="User likes Python", user_id="u1")
    hits = backend.recall(query="pizza preferences", user_id="u1", limit=5)
    contents = {fact.content for fact, _ in hits}
    assert {"User likes pizza", "User likes Python"} <= contents


# --- serialization ---------------------------------------------------------


def test_pinecone_to_dict_excludes_live_clients_and_includes_connection(backend):
    """`to_dict` must drop the runtime client/index but emit connection + embedder
    so the YAML round-trip rebuilds an equivalent backend."""
    data = backend.to_dict()
    assert "_client" not in data and "_index" not in data
    assert isinstance(data["connection"], dict)
    assert isinstance(data["embedder"], dict)
    # Persistent backend identity must survive serialization.
    assert data["index_name"] == "user_facts"
    assert data["namespace"] == "test"


def test_pinecone_to_dict_accepts_include_secure_params(backend):
    """`include_secure_params=True` must propagate through backend → connection
    without raising."""
    data = backend.to_dict(include_secure_params=True)
    assert "connection" in data and "embedder" in data
