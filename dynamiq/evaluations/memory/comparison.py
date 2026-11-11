"""Descriptive paired regression checks over compatible memory evidence."""

import math
from collections.abc import Sequence
from typing import Annotated

from pydantic import Field, StrictInt

from dynamiq.evaluations.memory.evidence import CapturedVersion, ExpectedRecall, MemoryEvidence
from dynamiq.evaluations.memory.metrics import MemoryAssertion, MemoryQueryEvaluation, assertion
from dynamiq.evaluations.memory.replay import replay_memory
from dynamiq.evaluations.memory.results import MemoryEvaluationReport, combine_verdicts
from dynamiq.evaluations.memory.suite import (
    Count,
    MemoryEvaluationLimits,
    MemoryMetric,
    MemoryModel,
    Name,
    OperationStatus,
    UnitValue,
    Verdict,
)

Difference = Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)]
Nonnegative = Annotated[StrictInt, Field(ge=0)]


class MemoryComparisonGate(MemoryModel):
    """An explicit observed improvement bound, not a significance test.

    A negative minimum permits that much degradation. Positive improvement
    means better for every metric, including lower adverse query rates.
    """

    metric: MemoryMetric
    cutoff: Annotated[StrictInt, Field(ge=1, le=1000)]
    group: Name = "overall"
    minimum_improvement: Difference
    minimum_support: Count = 1
    minimum_coverage: UnitValue = 1.0


class MemoryPairedMetric(MemoryModel):
    """Both means use exactly the same comparable, defined query pairs."""

    group: Name
    metric: MemoryMetric
    cutoff: Annotated[StrictInt, Field(ge=1, le=1000)]
    baseline_mean: UnitValue | None
    candidate_mean: UnitValue | None
    delta: Difference | None
    improvement: Difference | None
    expected_queries: Nonnegative
    baseline_available_queries: Nonnegative
    candidate_available_queries: Nonnegative
    baseline_defined_queries: Nonnegative
    candidate_defined_queries: Nonnegative
    baseline_undefined_queries: Nonnegative
    candidate_undefined_queries: Nonnegative
    unavailable_queries: Nonnegative
    undefined_queries: Nonnegative
    incomparable_queries: Nonnegative
    paired_queries: Nonnegative
    expected_trials: Nonnegative
    paired_trials: Nonnegative
    expected_cases: Nonnegative
    paired_cases: Nonnegative
    coverage: UnitValue | None
    reason: Name | None = None


class MemoryComparisonGateEvaluation(MemoryModel):
    """A comparison gate with independent support and coverage diagnostics."""

    gate: MemoryComparisonGate
    improvement: Difference | None
    assertions: tuple[MemoryAssertion, ...]
    verdict: Verdict


class MemoryComparisonReport(MemoryModel):
    """Full inputs plus reproducible paired summaries and explicit decisions.

    The shared revision is a caller assertion about scenario inputs. It does
    not prove identical embedding models, randomness or backend configuration.
    Repetition indices name corresponding planned slots, not shared draws.
    """

    baseline: MemoryEvidence
    candidate: MemoryEvidence
    dataset_revision: Name
    metrics: tuple[MemoryPairedMetric, ...]
    gates: tuple[MemoryComparisonGateEvaluation, ...]
    candidate_lifecycle_verdict: Verdict
    baseline_absolute_verdict: Verdict
    candidate_absolute_verdict: Verdict
    verdict: Verdict


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _compatible(baseline: MemoryEvidence, candidate: MemoryEvidence) -> str:
    revision = baseline.specification.dataset_revision
    if revision is None or revision != candidate.specification.dataset_revision:
        raise ValueError("comparison requires the same explicit dataset_revision assertion")
    # Absolute acceptance gates and display names are not execution semantics.
    # Source gate results remain visible separately in the comparison report.
    left = baseline.specification.model_dump(exclude={"name", "gates"})
    right = candidate.specification.model_dump(exclude={"name", "gates"})
    if left != right:
        raise ValueError("memory comparison requires matching scenario execution and reference specifications")
    return revision


def _partition(expected: ExpectedRecall, aliases: dict[str, CapturedVersion]) -> tuple[tuple[int, int, str], ...]:
    """Canonicalize equivalence, never compare run-local opaque token strings."""
    names = list(dict.fromkeys([item.alias for item in expected.relevance] + list(expected.forbidden)))
    objects: dict[str, int] = {}
    versions: dict[str, int] = {}
    signature = []
    for name in names:
        version = aliases[name]
        object_index = objects.setdefault(version.object_token, len(objects))
        version_index = versions.setdefault(version.version_token, len(versions))
        signature.append((object_index, version_index, version.actor))
    return tuple(signature)


def _partitions(evidence: MemoryEvidence) -> dict[tuple[str, int, str], tuple[tuple[int, int, str], ...]]:
    cases = {case.id: case for case in evidence.specification.cases}
    result = {}
    for trial in evidence.trials:
        observed = {step.step_id: step for step in trial.operations}
        aliases = {}
        for expected in cases[trial.case_id].steps:
            observation = observed.get(expected.id)
            if observation is None or observation.status != OperationStatus.COMPLETED:
                break
            if observation.remember is not None:
                aliases[expected.alias] = observation.remember.version
            elif isinstance(expected, ExpectedRecall):
                result[(trial.case_id, trial.repetition, expected.id)] = _partition(expected, aliases)
    return result


def _queries(report: MemoryEvaluationReport) -> dict[tuple[str, int, str], MemoryQueryEvaluation]:
    return {
        (trial.case_id, trial.repetition, operation.step_id): operation.query
        for trial in report.trials
        for operation in trial.operations
        if operation.query is not None
    }


def _paired_rows(
    baseline: MemoryEvaluationReport,
    candidate: MemoryEvaluationReport,
) -> tuple[MemoryPairedMetric, ...]:
    specification = baseline.evidence.specification
    groups = {"overall": tuple(case.id for case in specification.cases)}
    groups.update({group.name: group.cases for group in specification.groups})
    queries_by_case = {
        case.id: tuple(step for step in case.steps if isinstance(step, ExpectedRecall)) for case in specification.cases
    }
    left_queries, right_queries = _queries(baseline), _queries(candidate)
    left_partitions, right_partitions = _partitions(baseline.evidence), _partitions(candidate.evidence)
    left_scores = {
        key: {(score.metric, score.cutoff): score.value for score in query.metrics}
        for key, query in left_queries.items()
    }
    right_scores = {
        key: {(score.metric, score.cutoff): score.value for score in query.metrics}
        for key, query in right_queries.items()
    }
    rows = []
    for group, case_ids in groups.items():
        for cutoff in specification.cutoffs:
            for metric in MemoryMetric:
                left_cases, right_cases = [], []
                counts = dict.fromkeys(
                    (
                        "expected_queries",
                        "baseline_available_queries",
                        "candidate_available_queries",
                        "baseline_defined_queries",
                        "candidate_defined_queries",
                        "baseline_undefined_queries",
                        "candidate_undefined_queries",
                        "unavailable_queries",
                        "undefined_queries",
                        "incomparable_queries",
                        "paired_queries",
                        "paired_trials",
                    ),
                    0,
                )
                for case_id in case_ids:
                    if not queries_by_case[case_id]:
                        continue
                    left_repetitions, right_repetitions = [], []
                    for repetition in range(specification.repetitions):
                        left_values, right_values = [], []
                        for step in queries_by_case[case_id]:
                            key = (case_id, repetition, step.id)
                            left_query, right_query = left_queries[key], right_queries[key]
                            left = left_scores[key][(metric, cutoff)]
                            right = right_scores[key][(metric, cutoff)]
                            left_available = left_query.status == OperationStatus.COMPLETED
                            right_available = right_query.status == OperationStatus.COMPLETED
                            counts["expected_queries"] += 1
                            counts["baseline_available_queries"] += left_available
                            counts["candidate_available_queries"] += right_available
                            counts["baseline_defined_queries"] += left is not None
                            counts["candidate_defined_queries"] += right is not None
                            counts["baseline_undefined_queries"] += left_available and left is None
                            counts["candidate_undefined_queries"] += right_available and right is None
                            if not left_available or not right_available:
                                counts["unavailable_queries"] += 1
                            elif left is None or right is None:
                                counts["undefined_queries"] += 1
                            elif left_partitions[key] != right_partitions[key]:
                                counts["incomparable_queries"] += 1
                            else:
                                counts["paired_queries"] += 1
                                left_values.append(left)
                                right_values.append(right)
                        if left_values:
                            counts["paired_trials"] += 1
                            left_repetitions.append(_mean(left_values))
                            right_repetitions.append(_mean(right_values))
                    if left_repetitions:
                        left_cases.append(_mean(left_repetitions))
                        right_cases.append(_mean(right_repetitions))
                left_mean, right_mean = _mean(left_cases), _mean(right_cases)
                delta = right_mean - left_mean if left_mean is not None and right_mean is not None else None
                expected = counts["expected_queries"]
                rows.append(
                    MemoryPairedMetric(
                        group=group,
                        metric=metric,
                        cutoff=cutoff,
                        baseline_mean=left_mean,
                        candidate_mean=right_mean,
                        delta=delta,
                        improvement=delta * metric.improvement_sign if delta is not None else None,
                        expected_trials=len(case_ids) * specification.repetitions,
                        expected_cases=len(case_ids),
                        paired_cases=len(left_cases),
                        coverage=counts["paired_queries"] / expected if expected else None,
                        reason=(
                            "no_expected_queries" if not expected else "no_comparable_pairs" if delta is None else None
                        ),
                        **counts,
                    )
                )
    return tuple(rows)


def _check_gates(
    gates: Sequence[MemoryComparisonGate],
    metrics: Sequence[MemoryPairedMetric],
) -> tuple[MemoryComparisonGateEvaluation, ...]:
    index = {(row.group, row.metric, row.cutoff): row for row in metrics}
    evaluated = []
    used = set()
    for gate in gates:
        key = (gate.group, gate.metric, gate.cutoff)
        if key in used or key not in index:
            raise ValueError("unknown or repeated comparison metric/cutoff/group gate")
        used.add(key)
        row = index[key]
        checks = []
        requirements = (
            ("minimum_support", row.paired_queries >= gate.minimum_support),
            ("minimum_coverage", row.coverage is not None and row.coverage >= gate.minimum_coverage),
        )
        for name, enough in requirements:
            checks.append(
                MemoryAssertion(
                    name=name,
                    verdict=Verdict.PASS if enough else Verdict.INCONCLUSIVE,
                    reason="requirement_met" if enough else "insufficient_paired_evidence",
                )
            )
        if row.improvement is None:
            checks.append(
                MemoryAssertion(name="improvement_defined", verdict=Verdict.INCONCLUSIVE, reason="no_comparable_pairs")
            )
        else:
            checks.append(
                assertion(
                    "minimum_improvement",
                    row.improvement >= gate.minimum_improvement,
                    success="improvement_bound_met",
                    failure="improvement_below_bound",
                )
            )
        evaluated.append(
            MemoryComparisonGateEvaluation(
                gate=gate,
                improvement=row.improvement,
                assertions=tuple(checks),
                verdict=combine_verdicts([check.verdict for check in checks]),
            )
        )
    return tuple(evaluated)


def compare_memory(
    baseline: MemoryEvidence,
    candidate: MemoryEvidence,
    *,
    gates: Sequence[MemoryComparisonGate] = (),
    limits: MemoryEvaluationLimits | None = None,
) -> MemoryComparisonReport:
    """Pair compatible planned query slots and check explicit regression bounds.

    Missing results never disappear from coverage. Changed runtime alias
    equivalence makes a pair incomparable, preventing native dedup/upsert from
    changing relevance denominators and manufacturing an apparent improvement.
    No inference about a population or shared random sampling is performed.
    """
    trusted = MemoryEvaluationLimits.model_validate((limits or MemoryEvaluationLimits()).model_dump())
    if len(gates) > 1000:
        raise ValueError("too many memory comparison gates")
    gates = tuple(MemoryComparisonGate.model_validate(gate.model_dump()) for gate in gates)
    left = replay_memory(baseline, limits=trusted)
    right = replay_memory(candidate, limits=trusted)
    revision = _compatible(left.evidence, right.evidence)
    metrics = _paired_rows(left, right)
    evaluated = _check_gates(gates, metrics)
    lifecycle = combine_verdicts([trial.verdict for trial in right.trials])
    verdicts = [gate.verdict for gate in evaluated]
    if lifecycle in (Verdict.FAIL, Verdict.INCONCLUSIVE):
        verdicts.append(lifecycle)
    return MemoryComparisonReport(
        baseline=left.evidence,
        candidate=right.evidence,
        dataset_revision=revision,
        metrics=metrics,
        gates=evaluated,
        candidate_lifecycle_verdict=lifecycle,
        baseline_absolute_verdict=left.verdict,
        candidate_absolute_verdict=right.verdict,
        verdict=combine_verdicts(verdicts),
    )
