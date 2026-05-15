from __future__ import annotations

from dynamiq.ontology.models import TemporalFact


class OntologyValidationError(ValueError):
    """Raised when ontology memory data fails validation."""


def validate_temporal_fact(fact: TemporalFact) -> TemporalFact:
    if fact.invalid_at and fact.invalid_at < fact.valid_at:
        raise OntologyValidationError("TemporalFact.invalid_at cannot be earlier than valid_at.")
    if fact.expired_at and fact.expired_at < fact.created_at:
        raise OntologyValidationError("TemporalFact.expired_at cannot be earlier than created_at.")
    return fact
