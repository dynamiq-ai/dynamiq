from datetime import UTC, datetime

from dynamiq.memory.long_term.schemas import Fact


def test_fact_round_trip():
    now = datetime.now(UTC)
    fact = Fact(
        id="f1",
        content="User prefers terse responses",
        hash="abcd1234",
        user_id="u1",
        metadata={"category": "preference"},
        created_at=now,
        updated_at=now,
    )
    dumped = fact.model_dump()
    assert Fact(**dumped) == fact


def test_fact_metadata_defaults_to_empty_dict():
    now = datetime.now(UTC)
    fact = Fact(id="f1", content="x", hash="h", user_id="u",
                created_at=now, updated_at=now)
    assert fact.metadata == {}
