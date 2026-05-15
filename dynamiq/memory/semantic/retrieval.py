from __future__ import annotations

from enum import Enum


class ContextRetrievalMode(str, Enum):
    CURRENT = "current"
    HISTORICAL = "historical"
    AUDIT = "audit"
    OPS = "ops"
    SELF = "self"


def lexical_score(*, query: str | None, haystacks: list[str]) -> float:
    if not query:
        return 0.0
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return 0.0

    corpus = " ".join(haystacks).lower()
    matches = sum(1 for term in query_terms if term in corpus)
    return matches / len(query_terms)
