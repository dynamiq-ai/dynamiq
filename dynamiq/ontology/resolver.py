from __future__ import annotations

import re
from dataclasses import dataclass

from dynamiq.ontology.models import Entity

_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_entity_label(value: str) -> str:
    normalized = _NORMALIZE_PATTERN.sub(" ", value.strip().lower())
    return " ".join(normalized.split())


@dataclass(slots=True)
class EntityResolutionResult:
    entity: Entity
    matched_on: str
    confidence: float


class EntityResolver:
    """Resolve extracted labels to canonical entities using exact and alias matching."""

    def resolve(
        self,
        *,
        label: str,
        candidates: list[Entity],
    ) -> EntityResolutionResult | None:
        target = normalize_entity_label(label)
        if not target:
            return None

        for entity in candidates:
            if normalize_entity_label(entity.label) == target:
                return EntityResolutionResult(entity=entity, matched_on="label", confidence=1.0)

        for entity in candidates:
            for alias in entity.aliases:
                if normalize_entity_label(alias) == target:
                    return EntityResolutionResult(entity=entity, matched_on="alias", confidence=0.92)

        return None
