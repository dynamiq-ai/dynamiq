"""Coverage-aware, hierarchically weighted summaries of a fixed query panel."""

import math
from collections.abc import Sequence

from dynamiq.evaluations.memory.evidence import MemorySpecification
from dynamiq.evaluations.memory.metrics import MemoryAssertion, assertion
from dynamiq.evaluations.memory.results import (
    MemoryAggregate,
    MemoryGateEvaluation,
    MemoryTrialEvaluation,
    combine_verdicts,
)
from dynamiq.evaluations.memory.suite import MemoryMetric, OperationStatus, Verdict


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def aggregate_memory(
    specification: MemorySpecification,
    trials: Sequence[MemoryTrialEvaluation],
) -> tuple[MemoryAggregate, ...]:
    """Weight cases equally, retaining missing trials in coverage counts.

    An observed mean is conditional on available, defined scores. It does not
    impute a failed query as zero or silently claim that incomplete support is
    representative. Full-coverage gates are the default acceptance mechanism.
    """
    groups = {"overall": tuple(case.id for case in specification.cases)}
    groups.update({group.name: group.cases for group in specification.groups})
    # Index query slots once. Write-heavy scenarios must not rescan their full
    # lifecycle for every group, metric and cutoff.
    indexed = {
        (trial.case_id, trial.repetition): tuple(
            operation.query for operation in trial.operations if operation.query is not None
        )
        for trial in trials
    }
    scores = {
        (case_id, repetition, query.step_id): {(score.metric, score.cutoff): score.value for score in query.metrics}
        for (case_id, repetition), queries in indexed.items()
        for query in queries
    }
    query_cases = {case_id for (case_id, _), queries in indexed.items() if queries}
    rows = []
    for group, case_ids in groups.items():
        for cutoff in specification.cutoffs:
            for metric in MemoryMetric:
                case_values = []
                expected = available = supported = undefined = supported_trials = 0
                for case_id in case_ids:
                    if case_id not in query_cases:
                        continue
                    repetition_values = []
                    for repetition in range(specification.repetitions):
                        query_values = []
                        for query in indexed[(case_id, repetition)]:
                            expected += 1
                            if query.status == OperationStatus.COMPLETED:
                                available += 1
                            value = scores[(case_id, repetition, query.step_id)][(metric, cutoff)]
                            if value is not None:
                                supported += 1
                                query_values.append(value)
                            elif query.status == OperationStatus.COMPLETED:
                                undefined += 1
                        trial_value = _mean(query_values)
                        if trial_value is not None:
                            supported_trials += 1
                            repetition_values.append(trial_value)
                    case_value = _mean(repetition_values)
                    if case_value is not None:
                        case_values.append(case_value)
                value = _mean(case_values)
                rows.append(
                    MemoryAggregate(
                        group=group,
                        metric=metric,
                        cutoff=cutoff,
                        value=value,
                        expected_queries=expected,
                        available_queries=available,
                        supported_queries=supported,
                        undefined_queries=undefined,
                        expected_trials=len(case_ids) * specification.repetitions,
                        supported_trials=supported_trials,
                        expected_cases=len(case_ids),
                        supported_cases=len(case_values),
                        coverage=available / expected if expected else None,
                        reason=(
                            ("no_expected_queries" if not expected else "no_defined_scores") if value is None else None
                        ),
                    )
                )
    return tuple(rows)


def evaluate_gates(
    specification: MemorySpecification,
    aggregates: Sequence[MemoryAggregate],
) -> tuple[MemoryGateEvaluation, ...]:
    """Check observed bounds and minimum evidence without population claims."""
    index = {(row.group, row.metric, row.cutoff): row for row in aggregates}
    evaluated = []
    for gate in specification.gates:
        observed = index[(gate.group, gate.metric, gate.cutoff)]
        checks = []
        enough_support = observed.supported_queries >= gate.minimum_support
        enough_coverage = observed.coverage is not None and observed.coverage >= gate.minimum_coverage
        for name, satisfied in (("minimum_support", enough_support), ("minimum_coverage", enough_coverage)):
            checks.append(
                MemoryAssertion(
                    name=name,
                    verdict=Verdict.PASS if satisfied else Verdict.INCONCLUSIVE,
                    reason="requirement_met" if satisfied else "insufficient_evidence",
                )
            )
        if observed.value is None:
            checks.append(
                MemoryAssertion(name="metric_defined", verdict=Verdict.INCONCLUSIVE, reason="no_defined_scores")
            )
        else:
            if gate.minimum is not None:
                checks.append(
                    assertion("minimum", observed.value >= gate.minimum, success="bound_met", failure="below_minimum")
                )
            if gate.maximum is not None:
                checks.append(
                    assertion("maximum", observed.value <= gate.maximum, success="bound_met", failure="above_maximum")
                )
        evaluated.append(
            MemoryGateEvaluation(
                gate=gate,
                observed=observed.value,
                coverage=observed.coverage,
                support=observed.supported_queries,
                assertions=tuple(checks),
                verdict=combine_verdicts([check.verdict for check in checks]),
            )
        )
    return tuple(evaluated)
