"""Tests for the LongTermMemory facade."""
import pytest

from dynamiq.memory.long_term import LongTermMemory
from dynamiq.memory.long_term.backends.in_memory import InMemoryFactBackend
from dynamiq.memory.long_term.long_term_memory import LongTermMemoryError


@pytest.fixture
def ltm(fake_embedder):
    return LongTermMemory(
        backend=InMemoryFactBackend(),
        embedder=fake_embedder,
    )


# --- remember ---

def test_remember_returns_a_fact_and_persists_it(ltm, user_id):
    fact = ltm.remember(content="User likes pizza", user_id=user_id)
    assert fact.id
    assert fact.content == "User likes pizza"
    assert fact.user_id == user_id
    assert ltm.backend.get(fact.id) == fact


def test_remember_dedups_exact_duplicate_in_same_user(ltm, user_id):
    first = ltm.remember(content="User likes pizza", user_id=user_id)
    second = ltm.remember(content="User likes pizza", user_id=user_id)
    assert first.id == second.id


def test_remember_does_not_dedup_across_users(ltm, user_id, other_user_id):
    a = ltm.remember(content="User likes pizza", user_id=user_id)
    b = ltm.remember(content="User likes pizza", user_id=other_user_id)
    assert a.id != b.id
    assert a.user_id != b.user_id


def test_remember_normalises_whitespace_for_dedup(ltm, user_id):
    a = ltm.remember(content="  User likes pizza  ", user_id=user_id)
    b = ltm.remember(content="USER LIKES PIZZA", user_id=user_id)
    assert a.id == b.id


def test_remember_rejects_empty_content(ltm, user_id):
    with pytest.raises(LongTermMemoryError):
        ltm.remember(content="   ", user_id=user_id)


def test_remember_stores_metadata(ltm, user_id):
    fact = ltm.remember(content="x", user_id=user_id,
                        metadata={"category": "preference"})
    assert ltm.backend.get(fact.id).metadata == {"category": "preference"}


# --- recall ---

def test_recall_returns_scored_facts(ltm, user_id):
    ltm.remember(content="User likes pizza", user_id=user_id)
    ltm.remember(content="User dislikes mushrooms", user_id=user_id)
    hits = ltm.recall(query="pizza preferences", user_id=user_id, limit=2)
    assert len(hits) == 2
    fact, score = hits[0]
    assert fact.content
    assert isinstance(score, float)


def test_recall_isolates_users(ltm, user_id, other_user_id):
    ltm.remember(content="A's fact", user_id=user_id)
    ltm.remember(content="B's fact", user_id=other_user_id)
    hits = ltm.recall(query="fact", user_id=user_id, limit=5)
    assert all(f.user_id == user_id for f, _ in hits)


def test_recall_respects_limit(ltm, user_id):
    for i in range(5):
        ltm.remember(content=f"fact-{i}", user_id=user_id)
    hits = ltm.recall(query="fact", user_id=user_id, limit=2)
    assert len(hits) == 2


def test_recall_empty_store_returns_empty(ltm, user_id):
    assert ltm.recall(query="anything", user_id=user_id, limit=5) == []


def test_recall_rejects_empty_query(ltm, user_id):
    with pytest.raises(LongTermMemoryError):
        ltm.recall(query="   ", user_id=user_id, limit=5)


# --- forget ---

def test_forget_deletes_known_fact(ltm, user_id):
    fact = ltm.remember(content="x", user_id=user_id)
    assert ltm.forget(fact_id=fact.id, user_id=user_id) == "deleted"
    assert ltm.backend.get(fact.id) is None


def test_forget_unknown_returns_not_found(ltm, user_id):
    assert ltm.forget(fact_id="does-not-exist", user_id=user_id) == "not_found"


def test_forget_cross_user_returns_forbidden(ltm, user_id, other_user_id):
    fact = ltm.remember(content="x", user_id=user_id)
    result = ltm.forget(fact_id=fact.id, user_id=other_user_id)
    assert result == "forbidden"
    assert ltm.backend.get(fact.id) is not None


# --- admin / introspection ---

def test_list_all_returns_user_facts(ltm, user_id, other_user_id):
    ltm.remember(content="a", user_id=user_id)
    ltm.remember(content="b", user_id=user_id)
    ltm.remember(content="c", user_id=other_user_id)
    facts = ltm.list_all(user_id=user_id)
    assert {f.content for f in facts} == {"a", "b"}


def test_get_returns_fact_by_id(ltm, user_id):
    fact = ltm.remember(content="x", user_id=user_id)
    assert ltm.get(fact.id) == fact


def test_get_unknown_returns_none(ltm):
    assert ltm.get("nope") is None


def test_clear_user_deletes_all_user_facts(ltm, user_id, other_user_id):
    ltm.remember(content="a", user_id=user_id)
    ltm.remember(content="b", user_id=user_id)
    ltm.remember(content="c", user_id=other_user_id)
    deleted = ltm.clear_user(user_id=user_id)
    assert deleted == 2
    assert ltm.list_all(user_id=user_id) == []
    assert len(ltm.list_all(user_id=other_user_id)) == 1
