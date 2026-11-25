from datetime import datetime, timezone

import pytest

from dynamiq.evaluations.memory.identity import MemoryBackendContractError, MemoryIdentityRegistry
from dynamiq.evaluations.memory.suite import Remember
from dynamiq.memory.long_term.schemas import Fact
from dynamiq.memory.long_term.types import RememberOutcome


def fact(text="tea", *, user="scope-alice", id="backend-secret"):
    now = datetime.now(timezone.utc)
    return Fact(
        id=id,
        content=text,
        user_id=user,
        hash="not-exported",
        created_at=now,
        updated_at=now,
        metadata={"secret": "never-export"},
    )


def test_alias_versions_survive_updates_and_do_not_follow_mutable_facts():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    first = fact()
    old = registry.capture_remember(
        Remember(id="w1", actor="alice", alias="old", content="tea"), (first, RememberOutcome.CREATED)
    )
    first.content = "mutated externally"
    new = registry.capture_remember(
        Remember(id="w2", actor="alice", alias="new", content="coffee"), (fact("coffee"), RememberOutcome.UPDATED)
    )
    assert old.version.object_token == new.version.object_token
    assert old.version.version_token != new.version.version_token
    hits = registry.capture_recall("alice", [(fact("tea"), 0.8), (fact("coffee"), 0.7)], limit=5).hits
    assert hits[0].version_token == old.version.version_token
    assert hits[1].duplicate


def test_returned_corruption_fails_write_integrity_even_if_retrievable():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    observed = registry.capture_remember(
        Remember(id="w", actor="alice", alias="x", content="intended"), (fact("wrong"), RememberOutcome.CREATED)
    )
    assert observed.write_matches is False


def test_foreign_recall_keeps_rank_but_exports_no_foreign_payload():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    result = registry.capture_recall("alice", [(fact("secret text", user="other-secret-user"), 0.8)], limit=1)
    assert result.hits[0].rank == 1
    assert not result.hits[0].scope_valid
    assert result.hits[0].object_token is None
    assert "secret" not in result.model_dump_json()


def test_model_copy_cannot_bypass_returned_fact_validation():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    bad = fact().model_copy(update={"id": 17})
    with pytest.raises(MemoryBackendContractError):
        registry.capture_remember(
            Remember(id="w", actor="alice", alias="x", content="tea"), (bad, RememberOutcome.CREATED)
        )
    result = registry.capture_recall("alice", [(bad, 0.5), (fact(), float("nan"))], limit=2)
    assert [hit.rank for hit in result.hits] == [1, 2]
    assert all(not hit.output_valid for hit in result.hits)


def test_forget_tombstones_every_version_of_an_object():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    registry.capture_remember(
        Remember(id="w", actor="alice", alias="old", content="tea"), (fact(), RememberOutcome.CREATED)
    )
    registry.capture_remember(
        Remember(id="w2", actor="alice", alias="new", content="coffee"), (fact("coffee"), RememberOutcome.UPDATED)
    )
    registry.mark_deleted("old")
    hit = registry.capture_recall("alice", [(fact("coffee"), 1)], limit=1).hits[0]
    assert hit.deleted


def test_unchanged_must_match_current_version_not_any_historical_version():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    for index, (content, outcome) in enumerate((("tea", "created"), ("coffee", "updated"))):
        registry.capture_remember(
            Remember(id=f"w{index}", actor="alice", alias=f"a{index}", content=content), (fact(content), outcome)
        )
    result = registry.capture_remember(
        Remember(id="stale", actor="alice", alias="stale", content="tea"), (fact("tea"), "unchanged")
    )
    assert not result.identity_consistent


def test_foreign_forbidden_match_retains_no_foreign_identifiers():
    registry = MemoryIdentityRegistry({"alice": "scope-alice", "bob": "scope-bob"})
    foreign = fact("PRIVATE", user="scope-bob")
    registry.capture_remember(Remember(id="w", actor="bob", alias="secret", content="PRIVATE"), (foreign, "created"))
    result = registry.capture_recall("alice", [(foreign, 1.0)], limit=1, forbidden=("secret",))
    assert result.hits[0].forbidden_match
    assert result.hits[0].version_token is None
    assert "PRIVATE" not in result.model_dump_json()


def test_incomplete_constructed_fact_is_a_contract_violation():
    registry = MemoryIdentityRegistry({"alice": "scope-alice"})
    incomplete = Fact.model_construct(content="tea")
    with pytest.raises(MemoryBackendContractError):
        registry.capture_remember(Remember(id="w", actor="alice", alias="a", content="tea"), (incomplete, "created"))
