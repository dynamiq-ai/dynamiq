"""Rank-preserving retrieval metrics and lifecycle requirements on safe evidence."""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated

from pydantic import Field, StrictInt

from dynamiq.evaluations.memory.evidence import CapturedVersion, ExpectedRecall, RecallObservation
from dynamiq.evaluations.memory.suite import MemoryMetric, MemoryModel, Name, OperationStatus, UnitValue, Verdict


class MemoryAssertion(MemoryModel):
    """One explicit invariant or caller-provided lifecycle expectation."""

    name: Name
    verdict: Verdict
    reason: Name


class MemoryMetricValue(MemoryModel):
    """An observed query/cutoff score or an explicit undefined result."""

    metric: MemoryMetric
    cutoff: Annotated[StrictInt, Field(ge=1, le=1000)]
    value: UnitValue | None
    reason: Name | None = None


class MemoryQueryEvaluation(MemoryModel):
    """All cutoff scores for a single expected recall slot."""

    step_id: Name
    actor: Name
    status: OperationStatus
    metrics: tuple[MemoryMetricValue, ...]
    assertions: tuple[MemoryAssertion, ...]
    positive_versions: Annotated[StrictInt, Field(ge=0)] | None
    returned_hits: Annotated[StrictInt, Field(ge=0)] | None


def assertion(name: str, satisfied: bool, *, success: str, failure: str) -> MemoryAssertion:
    return MemoryAssertion(
        name=name,
        verdict=Verdict.PASS if satisfied else Verdict.FAIL,
        reason=success if satisfied else failure,
    )


def resolve_judgments(
    expected: ExpectedRecall,
    aliases: Mapping[str, CapturedVersion],
) -> tuple[dict[str, int], set[str]]:
    """Coalesce aliases after native dedup/upsert resolves their true identities.

    A declaration can be valid syntactically while two aliases resolve to the
    same captured version at runtime. Contradictory grades cannot be averaged
    or silently overwritten: the trial has an invalid reference specification.
    """
    grades: dict[str, int] = {}
    positive_objects: dict[str, str] = {}
    for item in expected.relevance:
        if item.alias not in aliases:
            raise ValueError("relevance references an unavailable captured alias")
        version = aliases[item.alias]
        if version.actor != expected.actor:
            raise ValueError("relevance references another actor")
        prior = grades.get(version.version_token)
        if prior is not None and prior != item.grade:
            raise ValueError("conflicting grades resolve to the same captured version")
        grades[version.version_token] = item.grade
        if item.grade > 0:
            previous_version = positive_objects.get(version.object_token)
            if previous_version is not None and previous_version != version.version_token:
                raise ValueError("multiple positive versions of one object cannot be recalled uniquely")
            positive_objects[version.object_token] = version.version_token
    forbidden = set()
    for alias in expected.forbidden:
        if alias not in aliases:
            raise ValueError("forbidden judgment references an unavailable alias")
        forbidden.add(aliases[alias].version_token)
    if any(grades.get(version, 0) > 0 for version in forbidden):
        raise ValueError("conflicting positive and forbidden aliases resolve to the same version")
    return grades, forbidden


def _dcg(grades: Sequence[int]) -> float:
    return math.fsum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def score_recall(
    expected: ExpectedRecall,
    observed: RecallObservation,
    aliases: Mapping[str, CapturedVersion],
    cutoffs: Sequence[int],
) -> MemoryQueryEvaluation:
    """Score original ranks; invalid, foreign and duplicate rows never earn credit.

    Precision uses k, even when fewer than k results are returned. A relevance
    set with no positive grades leaves recall/MRR/NDCG/hit rate undefined.
    Adverse rates are one-or-zero query indicators, not fractions of hit rows.
    """
    grades, forbidden = resolve_judgments(expected, aliases)
    positive = {version for version, grade in grades.items() if grade > 0}
    if tuple(hit.rank for hit in observed.hits) != tuple(range(1, len(observed.hits) + 1)):
        raise ValueError("recall ranks must be contiguous and preserve backend order")
    if len(observed.hits) > expected.limit:
        raise ValueError("recall evidence exceeds the declared request limit")
    known_versions = {version.version_token: version for version in aliases.values()}
    for hit in observed.hits:
        if hit.version_token is not None:
            version = known_versions.get(hit.version_token)
            if version is None or version.object_token != hit.object_token or version.actor != expected.actor:
                raise ValueError("recall references an impossible captured version")
    assertions = [
        assertion(
            "valid_output",
            all(hit.output_valid for hit in observed.hits) and not observed.excess_results,
            success="valid_ranked_result",
            failure="invalid_backend_output",
        ),
        assertion(
            "scope_isolation",
            all(hit.scope_valid for hit in observed.hits),
            success="all_hits_in_scope",
            failure="foreign_scope_returned",
        ),
        assertion(
            "forgotten_objects_absent",
            not any(hit.deleted for hit in observed.hits),
            success="no_deleted_object_returned",
            failure="deleted_object_returned",
        ),
    ]
    if expected.forbidden:
        assertions.append(
            assertion(
                "forbidden_versions_absent",
                not any(hit.forbidden_match or hit.version_token in forbidden for hit in observed.hits),
                success="no_forbidden_version_returned",
                failure="forbidden_version_returned",
            )
        )
    if expected.expect_empty is not None:
        assertions.append(
            assertion(
                "expected_empty",
                (len(observed.hits) == 0 and not observed.excess_results) == expected.expect_empty,
                success="empty_expectation_met",
                failure="empty_expectation_not_met",
            )
        )
    metrics = []
    for cutoff in cutoffs:
        if type(cutoff) is not int or cutoff < 1 or cutoff > expected.limit:
            raise ValueError("invalid or unsupported metric cutoff")
        selected = observed.hits[:cutoff]
        earned: set[str] = set()
        seen_objects: set[str] = set()
        gains = []
        first = None
        duplicate_query = False
        for hit in selected:
            repeated = hit.duplicate or (hit.object_token is not None and hit.object_token in seen_objects)
            duplicate_query = duplicate_query or repeated
            if hit.object_token is not None:
                seen_objects.add(hit.object_token)
            grade = 0
            if hit.output_valid and hit.scope_valid and not hit.deleted and not repeated:
                grade = grades.get(hit.version_token, 0)
                if grade > 0 and hit.version_token not in earned:
                    earned.add(hit.version_token)
                    if first is None:
                        first = hit.rank
                else:
                    grade = 0
            gains.append(grade)
        ideal = _dcg(sorted(grades.values(), reverse=True)[:cutoff])
        ndcg = _dcg(gains) / ideal if ideal > 0 else None
        if ndcg is not None:
            # Roundoff in otherwise identical DCG paths must not exceed Score's range.
            ndcg = min(1.0, max(0.0, ndcg))
        values = {
            MemoryMetric.PRECISION: len(earned) / cutoff,
            MemoryMetric.RECALL: len(earned) / len(positive) if positive else None,
            MemoryMetric.RECIPROCAL_RANK: (1 / first if first else 0.0) if positive else None,
            MemoryMetric.NDCG: ndcg,
            MemoryMetric.HIT_RATE: float(bool(earned)) if positive else None,
            MemoryMetric.FORBIDDEN_QUERY_RATE: float(
                any(hit.forbidden_match or hit.version_token in forbidden for hit in selected)
            ),
            MemoryMetric.DUPLICATE_QUERY_RATE: float(duplicate_query),
            MemoryMetric.UNKNOWN_QUERY_RATE: float(
                any(hit.scope_valid and hit.output_valid and hit.version_token is None for hit in selected)
            ),
            MemoryMetric.SCOPE_VIOLATION_QUERY_RATE: float(any(not hit.scope_valid for hit in selected)),
        }
        metrics.extend(
            MemoryMetricValue(
                metric=metric,
                cutoff=cutoff,
                value=value,
                reason="no_positive_judgments" if value is None else None,
            )
            for metric, value in values.items()
        )
    return MemoryQueryEvaluation(
        step_id=expected.id,
        actor=expected.actor,
        status=OperationStatus.COMPLETED,
        metrics=tuple(metrics),
        assertions=tuple(assertions),
        positive_versions=len(positive),
        returned_hits=len(observed.hits),
    )


def unavailable_recall(
    expected: ExpectedRecall,
    cutoffs: Sequence[int],
    *,
    status: OperationStatus,
    reason: str,
) -> MemoryQueryEvaluation:
    """Retain an expected query slot after factory, operation or input failure."""
    if status == OperationStatus.COMPLETED:
        raise ValueError("unavailable recall cannot be completed")
    return MemoryQueryEvaluation(
        step_id=expected.id,
        actor=expected.actor,
        status=status,
        metrics=tuple(
            MemoryMetricValue(metric=metric, cutoff=cutoff, value=None, reason=reason)
            for cutoff in cutoffs
            for metric in MemoryMetric
        ),
        assertions=(MemoryAssertion(name="query_available", verdict=Verdict.INCONCLUSIVE, reason=reason),),
        positive_versions=None,
        returned_hits=None,
    )
