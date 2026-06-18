"""Tests for the long-term memory backend operations (remember/recall/forget/...)."""
import pytest

from dynamiq.memory.long_term import LongTermMemoryError, RememberOutcome
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

# --- remember ---


def test_remember_returns_a_fact_and_persists_it(backend, user_id):
    fact, outcome = backend.remember(content="User likes pizza", user_id=user_id)
    assert outcome == RememberOutcome.CREATED
    assert fact.id
    assert fact.content == "User likes pizza"
    assert fact.user_id == user_id
    assert backend.get(fact.id) == fact


def test_remember_exact_duplicate_returns_unchanged(backend, user_id):
    first, first_outcome = backend.remember(content="User likes pizza", user_id=user_id)
    second, second_outcome = backend.remember(content="User likes pizza", user_id=user_id)
    assert first_outcome == RememberOutcome.CREATED
    assert second_outcome == RememberOutcome.UNCHANGED
    assert first.id == second.id


def test_remember_duplicate_applies_new_metadata(backend, user_id):
    """Re-stating the same content with changed metadata must persist the new
    metadata and return UPDATED, not silently drop it as UNCHANGED."""
    first, _ = backend.remember(content="User likes pizza", user_id=user_id, metadata={"source": "chat"})
    second, outcome = backend.remember(
        content="User likes pizza", user_id=user_id, metadata={"source": "profile"}
    )
    assert outcome == RememberOutcome.UPDATED
    assert second.id == first.id
    assert second.metadata == {"source": "profile"}
    assert backend.get(first.id).metadata == {"source": "profile"}


def test_remember_duplicate_without_metadata_stays_unchanged(backend, user_id):
    """No metadata supplied → don't churn an UPDATED outcome on an exact duplicate."""
    first, _ = backend.remember(content="User likes pizza", user_id=user_id, metadata={"source": "chat"})
    second, outcome = backend.remember(content="User likes pizza", user_id=user_id)
    assert outcome == RememberOutcome.UNCHANGED
    assert second.metadata == {"source": "chat"}


def test_remember_recovers_when_backend_raises_duplicate_error(backend, user_id, monkeypatch):
    """Simulates a cross-process race: another writer wins between our recheck
    and our insert. Backend raises LongTermMemoryDuplicateError; base.remember
    must re-fetch the winner and return UNCHANGED instead of propagating."""
    from datetime import UTC, datetime

    from dynamiq.memory.long_term.base import LongTermMemoryDuplicateError
    from dynamiq.memory.long_term.schemas import Fact

    cls = type(backend)
    winner = Fact(
        id="winner-id", content="User likes pizza", hash="dummy", user_id=user_id,
        metadata={}, created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )
    call_count = {"get_by_hash": 0}

    def flaky_get_by_hash(self, *, user_id, content_hash):
        call_count["get_by_hash"] += 1
        # Initial check returns None (we don't see the race yet); subsequent
        # calls (recheck under lock + post-error refetch) return the winner.
        return None if call_count["get_by_hash"] == 1 else winner

    def empty_search(self, *, query_embedding, scope, limit):
        return []  # so we skip the upsert path and reach the CREATE-with-insert path

    def boom_insert(self, fact, embedding):
        raise LongTermMemoryDuplicateError("simulated unique violation")

    monkeypatch.setattr(cls, "get_by_hash", flaky_get_by_hash)
    monkeypatch.setattr(cls, "search", empty_search)
    monkeypatch.setattr(cls, "insert", boom_insert)

    fact, outcome = backend.remember(content="User likes pizza", user_id=user_id)
    assert outcome == RememberOutcome.UNCHANGED
    assert fact.id == winner.id


def test_concurrent_remember_same_content_yields_one_fact(backend, user_id):
    """Two threads racing on the same content must produce exactly one stored
    fact: one CREATED, the other UNCHANGED (no duplicate row, no exception)."""
    import threading

    outcomes: list[RememberOutcome] = []
    outcomes_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def worker():
        barrier.wait()
        _, outcome = backend.remember(content="User likes pizza", user_id=user_id)
        with outcomes_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(o.value for o in outcomes) == ["created", "unchanged"]
    assert len(backend.list_by_scope({"user_id": user_id})) == 1


def test_remember_does_not_dedup_across_users(backend, user_id, other_user_id):
    a, _ = backend.remember(content="User likes pizza", user_id=user_id)
    b, b_outcome = backend.remember(content="User likes pizza", user_id=other_user_id)
    assert b_outcome == RememberOutcome.CREATED
    assert a.id != b.id
    assert a.user_id != b.user_id


def test_remember_normalises_whitespace_for_dedup(backend, user_id):
    a, _ = backend.remember(content="  User likes pizza  ", user_id=user_id)
    b, b_outcome = backend.remember(content="USER LIKES PIZZA", user_id=user_id)
    assert b_outcome == RememberOutcome.UNCHANGED
    assert a.id == b.id


def test_remember_paraphrase_upserts_existing(fake_embedder, user_id):
    """With a low threshold, a near-similar fact replaces the earlier one in place."""
    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder, upsert_threshold=0.0)
    original, _ = backend.remember(content="User likes pizza", user_id=user_id)
    updated, outcome = backend.remember(content="User loves pizza", user_id=user_id)

    assert outcome == RememberOutcome.UPDATED
    assert updated.id == original.id
    assert updated.content == "User loves pizza"
    assert backend.get(original.id).content == "User loves pizza"
    assert len(backend.list_all(user_id=user_id)) == 1


def test_remember_distinct_content_inserts_new_when_threshold_high(backend, user_id):
    """Default high threshold (0.85) keeps unrelated facts separate."""
    a, _ = backend.remember(content="User likes pizza", user_id=user_id)
    b, outcome = backend.remember(content="User dislikes mushrooms", user_id=user_id)
    assert outcome == RememberOutcome.CREATED
    assert a.id != b.id
    assert len(backend.list_all(user_id=user_id)) == 2


def test_upsert_replaces_metadata_when_provided(fake_embedder, user_id):
    """A corrected fact's new metadata must overwrite the old fact's metadata."""
    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder, upsert_threshold=0.0)
    original, _ = backend.remember(content="User likes pizza", user_id=user_id, metadata={"category": "food"})
    updated, outcome = backend.remember(
        content="User loves pizza", user_id=user_id, metadata={"category": "preference"}
    )
    assert outcome == RememberOutcome.UPDATED
    assert updated.id == original.id
    assert updated.metadata == {"category": "preference"}
    assert backend.get(original.id).metadata == {"category": "preference"}


def test_upsert_preserves_metadata_when_omitted(fake_embedder, user_id):
    """When the corrected call passes no metadata, the old metadata is kept."""
    backend = InMemoryLongTermMemoryBackend(embedder=fake_embedder, upsert_threshold=0.0)
    original, _ = backend.remember(content="User likes pizza", user_id=user_id, metadata={"category": "food"})
    updated, _ = backend.remember(content="User loves pizza", user_id=user_id)
    assert updated.id == original.id
    assert updated.metadata == {"category": "food"}


def test_remember_rejects_empty_content(backend, user_id):
    with pytest.raises(LongTermMemoryError):
        backend.remember(content="   ", user_id=user_id)


def test_remember_stores_metadata(backend, user_id):
    fact, _ = backend.remember(content="x", user_id=user_id, metadata={"category": "preference"})
    assert backend.get(fact.id).metadata == {"category": "preference"}


# --- recall ---


def test_recall_returns_scored_facts(backend, user_id):
    backend.remember(content="User likes pizza", user_id=user_id)
    backend.remember(content="User dislikes mushrooms", user_id=user_id)
    hits = backend.recall(query="pizza preferences", user_id=user_id, limit=2)
    assert len(hits) == 2
    fact, score = hits[0]
    assert fact.content
    assert isinstance(score, float)


def test_recall_isolates_users(backend, user_id, other_user_id):
    backend.remember(content="A's fact", user_id=user_id)
    backend.remember(content="B's fact", user_id=other_user_id)
    hits = backend.recall(query="fact", user_id=user_id, limit=5)
    assert all(f.user_id == user_id for f, _ in hits)


def test_recall_respects_limit(backend, user_id):
    for i in range(5):
        backend.remember(content=f"fact-{i}", user_id=user_id)
    hits = backend.recall(query="fact", user_id=user_id, limit=2)
    assert len(hits) == 2


def test_recall_empty_store_returns_empty(backend, user_id):
    assert backend.recall(query="anything", user_id=user_id, limit=5) == []


def test_recall_rejects_empty_query(backend, user_id):
    with pytest.raises(LongTermMemoryError):
        backend.recall(query="   ", user_id=user_id, limit=5)


# --- forget (programmatic API; not exposed to agents) ---


def test_forget_deletes_known_fact(backend, user_id):
    fact, _ = backend.remember(content="x", user_id=user_id)
    assert backend.forget(fact_id=fact.id, user_id=user_id) == "deleted"
    assert backend.get(fact.id) is None


def test_forget_unknown_returns_not_found(backend, user_id):
    assert backend.forget(fact_id="does-not-exist", user_id=user_id) == "not_found"


def test_forget_cross_user_returns_forbidden(backend, user_id, other_user_id):
    fact, _ = backend.remember(content="x", user_id=user_id)
    result = backend.forget(fact_id=fact.id, user_id=other_user_id)
    assert result == "forbidden"
    assert backend.get(fact.id) is not None


# --- admin / introspection ---


def test_list_all_returns_user_facts(backend, user_id, other_user_id):
    backend.remember(content="a", user_id=user_id)
    backend.remember(content="b", user_id=user_id)
    backend.remember(content="c", user_id=other_user_id)
    facts = backend.list_all(user_id=user_id)
    assert {f.content for f in facts} == {"a", "b"}


def test_get_returns_fact_by_id(backend, user_id):
    fact, _ = backend.remember(content="x", user_id=user_id)
    assert backend.get(fact.id) == fact


def test_get_unknown_returns_none(backend):
    assert backend.get("nope") is None


def test_clear_user_deletes_all_user_facts(backend, user_id, other_user_id):
    backend.remember(content="a", user_id=user_id)
    backend.remember(content="b", user_id=user_id)
    backend.remember(content="c", user_id=other_user_id)
    deleted = backend.clear_user(user_id=user_id)
    assert deleted == 2
    assert backend.list_all(user_id=user_id) == []
    assert len(backend.list_all(user_id=other_user_id)) == 1
