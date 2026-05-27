import pytest

from dynamiq.memory.long_term import LongTermMemory, RememberOutcome
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.memory.long_term.long_term_memory import LongTermMemoryError


@pytest.fixture
def ltm(fake_embedder):
    return LongTermMemory(
        backend=InMemoryLongTermMemoryBackend(),
        embedder=fake_embedder,
    )


# --- remember ---

def test_remember_returns_a_fact_and_persists_it(ltm, user_id):
    fact, outcome = ltm.remember(content="User likes pizza", user_id=user_id)
    assert outcome == RememberOutcome.CREATED
    assert fact.id
    assert fact.content == "User likes pizza"
    assert fact.user_id == user_id
    assert ltm.backend.get(fact.id) == fact


def test_remember_exact_duplicate_returns_unchanged(ltm, user_id):
    first, first_outcome = ltm.remember(content="User likes pizza", user_id=user_id)
    second, second_outcome = ltm.remember(content="User likes pizza", user_id=user_id)
    assert first_outcome == RememberOutcome.CREATED
    assert second_outcome == RememberOutcome.UNCHANGED
    assert first.id == second.id


def test_remember_does_not_dedup_across_users(ltm, user_id, other_user_id):
    a, _ = ltm.remember(content="User likes pizza", user_id=user_id)
    b, b_outcome = ltm.remember(content="User likes pizza", user_id=other_user_id)
    assert b_outcome == RememberOutcome.CREATED
    assert a.id != b.id
    assert a.user_id != b.user_id


def test_remember_normalises_whitespace_for_dedup(ltm, user_id):
    a, _ = ltm.remember(content="  User likes pizza  ", user_id=user_id)
    b, b_outcome = ltm.remember(content="USER LIKES PIZZA", user_id=user_id)
    assert b_outcome == RememberOutcome.UNCHANGED
    assert a.id == b.id


def test_remember_paraphrase_upserts_existing(fake_embedder, user_id):
    """With a low threshold, a near-similar fact replaces the earlier one in place."""
    ltm = LongTermMemory(
        backend=InMemoryLongTermMemoryBackend(),
        embedder=fake_embedder,
        upsert_threshold=0.0,
    )
    original, _ = ltm.remember(content="User likes pizza", user_id=user_id)
    updated, outcome = ltm.remember(content="User loves pizza", user_id=user_id)

    assert outcome == RememberOutcome.UPDATED
    assert updated.id == original.id
    assert updated.content == "User loves pizza"
    assert ltm.backend.get(original.id).content == "User loves pizza"
    assert len(ltm.list_all(user_id=user_id)) == 1


def test_remember_distinct_content_inserts_new_when_threshold_high(ltm, user_id):
    """Default high threshold (0.85) keeps unrelated facts separate."""
    a, _ = ltm.remember(content="User likes pizza", user_id=user_id)
    b, outcome = ltm.remember(content="User dislikes mushrooms", user_id=user_id)
    assert outcome == RememberOutcome.CREATED
    assert a.id != b.id
    assert len(ltm.list_all(user_id=user_id)) == 2


def test_remember_rejects_empty_content(ltm, user_id):
    with pytest.raises(LongTermMemoryError):
        ltm.remember(content="   ", user_id=user_id)


def test_remember_stores_metadata(ltm, user_id):
    fact, _ = ltm.remember(content="x", user_id=user_id, metadata={"category": "preference"})
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


# --- forget (programmatic API; not exposed to agents) ---

def test_forget_deletes_known_fact(ltm, user_id):
    fact, _ = ltm.remember(content="x", user_id=user_id)
    assert ltm.forget(fact_id=fact.id, user_id=user_id) == "deleted"
    assert ltm.backend.get(fact.id) is None


def test_forget_unknown_returns_not_found(ltm, user_id):
    assert ltm.forget(fact_id="does-not-exist", user_id=user_id) == "not_found"


def test_forget_cross_user_returns_forbidden(ltm, user_id, other_user_id):
    fact, _ = ltm.remember(content="x", user_id=user_id)
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
    fact, _ = ltm.remember(content="x", user_id=user_id)
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
