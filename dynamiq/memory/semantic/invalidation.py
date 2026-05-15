from __future__ import annotations

from dynamiq.ontology import TemporalFact

CONFLICTING_PREDICATES = {
    "has_preference",
    "works_at",
    "assigned_to",
}


def predicate_supports_invalidation(predicate: str) -> bool:
    return predicate in CONFLICTING_PREDICATES


def facts_conflict(existing: TemporalFact, incoming: TemporalFact) -> bool:
    if existing.id == incoming.id:
        return False
    if existing.subject_id != incoming.subject_id or existing.predicate != incoming.predicate:
        return False
    if existing.object_id and incoming.object_id:
        return existing.object_id != incoming.object_id
    return existing.object_value != incoming.object_value
