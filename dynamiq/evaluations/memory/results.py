"""Derived results kept separate from the portable execution evidence."""

from typing import Annotated, Literal

from pydantic import Field, StrictInt

from dynamiq.evaluations.memory.evidence import MemoryEvidence
from dynamiq.evaluations.memory.metrics import MemoryAssertion, MemoryQueryEvaluation
from dynamiq.evaluations.memory.suite import (
    MemoryGate,
    MemoryMetric,
    MemoryModel,
    Name,
    OperationStatus,
    UnitValue,
    Verdict,
)

Nonnegative = Annotated[StrictInt, Field(ge=0)]


class MemoryOperationEvaluation(MemoryModel):
    """Assertions for an expected operation, including unavailable slots."""

    step_id: Name
    kind: Name
    status: OperationStatus
    assertions: tuple[MemoryAssertion, ...]
    query: MemoryQueryEvaluation | None = None
    verdict: Verdict


class MemoryTrialEvaluation(MemoryModel):
    """A complete case/repetition denominator with separate cleanup results."""

    case_id: Name
    repetition: Nonnegative
    operations: tuple[MemoryOperationEvaluation, ...]
    cleanup: tuple[MemoryAssertion, ...]
    expected_operations: Nonnegative
    completed_operations: Nonnegative
    verdict: Verdict


class MemoryAggregate(MemoryModel):
    """Equal case, then repetition, then query weighted observed metric.

    Availability measures completed queries, including structurally undefined
    scores. Support counts only defined query scores. Both retain the original
    expected panel so short or failed trials cannot disappear from coverage.
    """

    group: Name
    metric: MemoryMetric
    cutoff: Annotated[StrictInt, Field(ge=1, le=1000)]
    value: UnitValue | None
    expected_queries: Nonnegative
    available_queries: Nonnegative
    supported_queries: Nonnegative
    undefined_queries: Nonnegative
    expected_trials: Nonnegative
    supported_trials: Nonnegative
    expected_cases: Nonnegative
    supported_cases: Nonnegative
    coverage: UnitValue | None
    reason: Name | None = None


class MemoryGateEvaluation(MemoryModel):
    """Observed bound decisions with independent coverage/support requirements."""

    gate: MemoryGate
    observed: UnitValue | None
    coverage: UnitValue | None
    support: Nonnegative
    assertions: tuple[MemoryAssertion, ...]
    verdict: Verdict


class MemoryEvaluationReport(MemoryModel):
    """Recomputed results and the complete privacy-filtered replay evidence."""

    format: Literal["dynamiq.memory.report"] = "dynamiq.memory.report"
    version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    evidence: MemoryEvidence
    trials: tuple[MemoryTrialEvaluation, ...]
    aggregates: tuple[MemoryAggregate, ...]
    gates: tuple[MemoryGateEvaluation, ...]
    expected_operations: Nonnegative
    completed_operations: Nonnegative
    verdict: Verdict


def combine_verdicts(verdicts: list[Verdict]) -> Verdict:
    """A known violation wins even when other requirements lack observations."""
    for verdict in (Verdict.FAIL, Verdict.INCONCLUSIVE, Verdict.PASS):
        if verdict in verdicts:
            return verdict
    return Verdict.NOT_EVALUATED
