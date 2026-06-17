"""Tests for WeaviateLongTermMemoryBackend.

These exercise the backend's storage primitives against an in-process fake
Weaviate v4 collection that mimics the subset of the API the backend uses.
No live Weaviate calls are made.
"""
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.memory.long_term.backends.weaviate import (
    WeaviateLongTermMemoryBackend,
    _to_weaviate_uuid,
)
from dynamiq.memory.long_term.schemas import Fact

# --- Fake Weaviate client / collection -------------------------------------


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return 1.0 - dot / (na * nb)


class _FakeData:
    def __init__(self, store: dict) -> None:
        self.store = store

    def insert(self, *, uuid, properties, vector):
        self.store[uuid] = {"uuid": uuid, "properties": dict(properties), "vector": list(vector)}

    def replace(self, *, uuid, properties, vector):
        self.store[uuid] = {"uuid": uuid, "properties": dict(properties), "vector": list(vector)}

    def delete_by_id(self, *, uuid):
        self.store.pop(uuid, None)

    def delete_many(self, *, where):
        for uid, item in list(self.store.items()):
            if where.matches(item):
                self.store.pop(uid, None)


class _FakeQuery:
    def __init__(self, store: dict) -> None:
        self.store = store

    def fetch_object_by_id(self, *, uuid):
        item = self.store.get(uuid)
        if item is None:
            return None
        return SimpleNamespace(uuid=uuid, properties=item["properties"], metadata=None)

    def fetch_objects(self, *, filters=None, limit=10):
        candidates = [
            SimpleNamespace(uuid=uid, properties=item["properties"], metadata=None)
            for uid, item in self.store.items()
            if filters is None or filters.matches(item)
        ]
        return SimpleNamespace(objects=candidates[:limit])

    def near_vector(self, *, near_vector, limit=10, filters=None, return_metadata=None):
        candidates = [
            (uid, item, _cosine_distance(near_vector, item["vector"]))
            for uid, item in self.store.items()
            if filters is None or filters.matches(item)
        ]
        candidates.sort(key=lambda t: t[2])
        return SimpleNamespace(
            objects=[
                SimpleNamespace(
                    uuid=uid,
                    properties=item["properties"],
                    metadata=SimpleNamespace(distance=dist),
                )
                for uid, item, dist in candidates[:limit]
            ]
        )


class _FakeCollection:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}
        self.data = _FakeData(self.store)
        self.query = _FakeQuery(self.store)


class _FakeCollections:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def get(self, name):
        return self.collections.setdefault(name, _FakeCollection())

    def exists(self, name):
        return name in self.collections

    def create(self, *, name, **_):
        self.collections.setdefault(name, _FakeCollection())

    def delete(self, name):
        self.collections.pop(name, None)


class _FakeClient:
    def __init__(self) -> None:
        self.collections = _FakeCollections()


# We bypass the real weaviate `Filter` objects entirely — the backend's
# `_scope_to_filter` builds them via `Filter.by_property(...).equal(...)`,
# which calls into the weaviate library. For mock tests we monkeypatch the
# scope-to-filter helper to return a callable predicate the fakes can evaluate.


class _PredicateFilter:
    def __init__(self, predicate) -> None:
        self._predicate = predicate

    def matches(self, item) -> bool:
        return self._predicate(item)

    def __and__(self, other):
        return _PredicateFilter(lambda item: self._predicate(item) and other._predicate(item))


def _fake_scope_to_filter(scope: dict):
    if not scope:
        return None
    return _PredicateFilter(lambda item: all(item["properties"].get(k) == v for k, v in scope.items()))


def _fake_id_in_filter(uuids):
    uuid_set = set(uuids)
    return _PredicateFilter(lambda item: item["uuid"] in uuid_set)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def fake_weaviate_client(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(WeaviateConnection, "connect", lambda self: client)
    # Swap in a fake scope_to_filter so the backend uses our predicate fakes
    # instead of real weaviate Filter objects (which the fake store can't evaluate).
    import dynamiq.memory.long_term.backends.weaviate as weaviate_backend

    monkeypatch.setattr(weaviate_backend, "_scope_to_filter", _fake_scope_to_filter)
    monkeypatch.setattr(weaviate_backend, "_id_in_filter", _fake_id_in_filter)
    return client


@pytest.fixture
def backend(fake_embedder, fake_weaviate_client):
    backend = WeaviateLongTermMemoryBackend(
        connection=WeaviateConnection(api_key="test-key", url="http://localhost"),
        embedder=fake_embedder,
        collection_name="UserFacts",
        dimension=fake_embedder.DIM,
    )
    # The real backend's `model_post_init` already called `collections.get`,
    # which our fake auto-creates on access — so the collection is ready.
    return backend


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


def test_weaviate_insert_then_get(backend, fake_embedder):
    fact = _fact("f1", "u1", "hello")
    backend.insert(fact, fake_embedder.embed("hello"))
    fetched = backend.get("f1")
    assert fetched is not None and fetched.id == "f1" and fetched.content == "hello"


def test_weaviate_get_unknown_returns_none(backend):
    assert backend.get("does-not-exist") is None


def test_weaviate_get_by_hash(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    found = backend.get_by_hash(user_id="u1", content_hash="h-shared")
    assert found is not None and found.id == "f1"


def test_weaviate_get_by_hash_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x", "h-shared"), fake_embedder.embed("x"))
    assert backend.get_by_hash(user_id="u2", content_hash="h-shared") is None


def test_weaviate_metadata_round_trip(backend, fake_embedder):
    """Free-form metadata must round-trip through the JSON-encoded property."""
    fact = _fact("f1", "u1", "x").model_copy(update={"metadata": {"category": "preference", "score": 0.8}})
    backend.insert(fact, fake_embedder.embed("x"))
    assert backend.get("f1").metadata == {"category": "preference", "score": 0.8}


def test_weaviate_construction_does_not_touch_collection(fake_embedder, monkeypatch):
    """A fresh backend must construct cleanly without resolving the collection —
    that lookup is deferred to first use so `ensure_collection()` can run after."""

    class _StrictCollections:
        def __init__(self) -> None:
            self.get_called_with: list = []

        def get(self, name):
            self.get_called_with.append(name)
            return _FakeCollection()

    class _StrictClient:
        def __init__(self) -> None:
            self.collections = _StrictCollections()

    client = _StrictClient()
    monkeypatch.setattr(WeaviateConnection, "connect", lambda self: client)
    backend = WeaviateLongTermMemoryBackend(
        connection=WeaviateConnection(api_key="k", url="http://localhost"),
        embedder=fake_embedder,
        collection_name="UserFacts",
        dimension=fake_embedder.DIM,
    )
    assert client.collections.get_called_with == []  # not yet resolved
    _ = backend._collection  # first access resolves
    assert client.collections.get_called_with == ["UserFacts"]


def test_weaviate_first_op_auto_ensures_collection(backend, monkeypatch):
    """First high-level op must auto-provision the collection — consumers should
    not have to remember to call `ensure_collection()` manually."""
    calls: list[int] = []
    original = WeaviateLongTermMemoryBackend.ensure_collection

    def wrapped(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(WeaviateLongTermMemoryBackend, "ensure_collection", wrapped)
    backend.remember(content="hello", user_id="u1")
    backend.remember(content="world", user_id="u1")
    backend.recall(query="hello", user_id="u1")
    assert len(calls) == 1, f"expected exactly one ensure_collection call, got {len(calls)}"


def test_weaviate_fact_id_maps_to_deterministic_uuid():
    """Two backends must resolve the same fact_id to the same UUID — so a fact
    inserted by one process can be deleted by another via the original id."""
    assert _to_weaviate_uuid("fact-1") == _to_weaviate_uuid("fact-1")
    assert _to_weaviate_uuid("fact-1") != _to_weaviate_uuid("fact-2")


# --- delete / list_by_scope / delete_scope ---------------------------------


def test_weaviate_delete(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "x"), fake_embedder.embed("x"))
    backend.delete("f1")
    assert backend.get("f1") is None


def test_weaviate_update_replaces_content_and_vector(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "old"), fake_embedder.embed("old"))
    backend.update(
        "f1",
        content="new",
        content_hash="h-new",
        embedding=fake_embedder.embed("new"),
        metadata={"k": "v"},
        updated_at=datetime.now(UTC),
    )
    fetched = backend.get("f1")
    assert fetched.content == "new" and fetched.hash == "h-new" and fetched.metadata == {"k": "v"}


def test_weaviate_list_by_scope(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    listed = backend.list_by_scope({"user_id": "u1"})
    assert {f.id for f in listed} == {"f1", "f2"}


def test_weaviate_delete_scope_returns_accurate_count(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    backend.insert(_fact("f2", "u1", "b"), fake_embedder.embed("b"))
    backend.insert(_fact("f3", "u2", "c"), fake_embedder.embed("c"))
    assert backend.delete_scope({"user_id": "u1"}) == 2
    assert backend.list_by_scope({"user_id": "u1"}) == []
    assert len(backend.list_by_scope({"user_id": "u2"})) == 1


def test_weaviate_delete_scope_empty_returns_zero(backend):
    assert backend.delete_scope({"user_id": "nobody"}) == 0


def test_weaviate_delete_scope_paginates_beyond_single_page_with_scope(backend, fake_embedder, monkeypatch):
    """A scoped delete must remove every match and return the true count even
    when the matched set exceeds Weaviate's per-call fetch cap."""
    monkeypatch.setattr(type(backend), "_SCOPE_PAGE_SIZE", 2)
    for i in range(5):
        backend.insert(_fact(f"f{i}", "u1", f"c{i}"), fake_embedder.embed(f"c{i}"))
    assert backend.delete_scope({"user_id": "u1"}) == 5
    assert backend.list_by_scope({"user_id": "u1"}) == []


def test_weaviate_delete_scope_rejects_empty_scope(backend, fake_embedder):
    """Empty scope is rejected to prevent accidental whole-collection wipes."""
    backend.insert(_fact("f1", "u1", "a"), fake_embedder.embed("a"))
    with pytest.raises(ValueError, match="non-empty scope"):
        backend.delete_scope({})
    assert len(backend.list_by_scope({"user_id": "u1"})) == 1


# --- search ----------------------------------------------------------------


def test_weaviate_search_relevance_ordered(backend, fake_embedder):
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


def test_weaviate_search_isolates_users(backend, fake_embedder):
    backend.insert(_fact("f1", "u1", "alpha"), fake_embedder.embed("alpha"))
    backend.insert(_fact("f2", "u2", "alpha"), fake_embedder.embed("alpha"))
    hits = backend.search(
        query_embedding=fake_embedder.embed("alpha"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert [fact.id for fact, _ in hits] == ["f1"]


def test_weaviate_search_empty_returns_empty(backend, fake_embedder):
    hits = backend.search(
        query_embedding=fake_embedder.embed("x"),
        scope={"user_id": "u1"},
        limit=5,
    )
    assert hits == []


# --- high-level operations via Template Method ------------------------------


def test_weaviate_remember_and_recall_through_backend(backend):
    backend.remember(content="User likes pizza", user_id="u1")
    backend.remember(content="User likes Python", user_id="u1")
    hits = backend.recall(query="pizza preferences", user_id="u1", limit=5)
    contents = {fact.content for fact, _ in hits}
    assert {"User likes pizza", "User likes Python"} <= contents


# --- serialization ---------------------------------------------------------


def test_weaviate_to_dict_excludes_live_clients_and_includes_connection(backend):
    data = backend.to_dict()
    assert "_client" not in data and "_collection" not in data
    assert isinstance(data["connection"], dict)
    assert isinstance(data["embedder"], dict)
    assert data["collection_name"] == "UserFacts"


def test_weaviate_to_dict_accepts_include_secure_params(backend):
    data = backend.to_dict(include_secure_params=True)
    assert "connection" in data and "embedder" in data
