from datetime import datetime, timezone

import pytest

from dynamiq.ontology import Entity, OntologyEntityType, TemporalFact


def test_entity_defaults_ontology_uri_from_type_and_id():
    entity = Entity(id="user-1", label="Ada", entity_type=OntologyEntityType.USER)

    assert entity.ontology_uri == "dynamiq://User/user-1"


def test_temporal_fact_requires_object_target():
    with pytest.raises(ValueError, match="object_id or object_value"):
        TemporalFact(subject_id="user-1", predicate="has_preference")


def test_temporal_fact_accepts_time_fields():
    valid_at = datetime(2026, 5, 14, 10, 0, tzinfo=timezone.utc)
    invalid_at = datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc)

    fact = TemporalFact(
        subject_id="user-1",
        predicate="has_preference",
        object_value="concise answers",
        valid_at=valid_at,
        invalid_at=invalid_at,
    )

    assert fact.valid_at == valid_at
    assert fact.invalid_at == invalid_at
