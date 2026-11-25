"""Text-free expected ledgers and selected backend observations for replay."""

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from dynamiq.evaluations.memory.suite import (
    MemoryGate,
    MemoryGroup,
    MemoryModel,
    MemorySuite,
    Name,
    OperationStatus,
    Relevance,
)
from dynamiq.memory.long_term.types import ForgetStatus, RememberOutcome

Seconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
ScoreValue = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class ExpectedRemember(MemoryModel):
    """Write requirements without the requested memory text."""

    kind: Literal["remember"] = "remember"
    id: Name
    actor: Name
    alias: Name
    write_match: Literal["native", "exact"] = "native"
    expected_outcome: RememberOutcome | None = None
    expected_same_as: Name | None = None


class ExpectedRecall(MemoryModel):
    """Rank requirements without the query text."""

    kind: Literal["recall"] = "recall"
    id: Name
    actor: Name
    limit: Annotated[StrictInt, Field(ge=1, le=1000)]
    relevance: tuple[Relevance, ...] = Field(default=(), max_length=1000)
    forbidden: tuple[Name, ...] = Field(default=(), max_length=1000)
    expect_empty: Annotated[bool, Field(strict=True)] | None = None


class ExpectedForget(MemoryModel):
    """Delete the current object represented by the named earlier alias."""

    kind: Literal["forget"] = "forget"
    id: Name
    actor: Name
    alias: Name
    expected_status: ForgetStatus


ExpectedStep = Annotated[ExpectedRemember | ExpectedRecall | ExpectedForget, Field(discriminator="kind")]


class ExpectedCase(MemoryModel):
    """Complete planned operation order, independent of any successful results."""

    id: Name
    actors: tuple[Name, ...] = Field(min_length=1, max_length=32)
    steps: tuple[ExpectedStep, ...] = Field(min_length=1, max_length=1000)


class MemorySpecification(MemoryModel):
    """Sanitized suite embedded in replay evidence, including failed trial slots."""

    name: Name
    dataset_revision: Name | None = None
    cases: tuple[ExpectedCase, ...] = Field(min_length=1, max_length=2000)
    cutoffs: tuple[Annotated[StrictInt, Field(ge=1, le=1000)], ...] = Field(min_length=1, max_length=32)
    repetitions: Annotated[StrictInt, Field(ge=1, le=100)]
    groups: tuple[MemoryGroup, ...] = Field(default=(), max_length=50)
    gates: tuple[MemoryGate, ...] = Field(default=(), max_length=1000)

    @classmethod
    def from_suite(cls, suite: MemorySuite) -> "MemorySpecification":
        data = suite.model_dump()
        cases = []
        for case in data["cases"]:
            steps = []
            for step in case["steps"]:
                steps.append({key: value for key, value in step.items() if key not in ("content", "query")})
            cases.append({**case, "steps": steps})
        return cls.model_validate({**data, "cases": cases})

    def as_validation_suite(self) -> MemorySuite:
        """Reuse scenario validation with inert placeholders, never real text."""
        data = self.model_dump()
        for case in data["cases"]:
            for step in case["steps"]:
                if step["kind"] == "remember":
                    step["content"] = "validation placeholder"
                elif step["kind"] == "recall":
                    step["query"] = "validation placeholder"
        return MemorySuite.model_validate(data)


class CapturedVersion(MemoryModel):
    """Report-local opaque identities, never actual provider IDs or text hashes."""

    object_token: Name
    version_token: Name
    actor: Name


class RememberObservation(MemoryModel):
    """A captured write result with independent content and identity assertions."""

    version: CapturedVersion
    outcome: RememberOutcome
    write_matches: Annotated[bool, Field(strict=True)]
    identity_consistent: Annotated[bool, Field(strict=True)]


class RecallHit(MemoryModel):
    """One original rank; malformed and foreign rows are retained as violations."""

    rank: Annotated[StrictInt, Field(ge=1, le=1001)]
    object_token: Name | None = None
    version_token: Name | None = None
    score: ScoreValue | None = None
    scope_valid: Annotated[bool, Field(strict=True)] = True
    output_valid: Annotated[bool, Field(strict=True)] = True
    duplicate: Annotated[bool, Field(strict=True)] = False
    deleted: Annotated[bool, Field(strict=True)] = False
    forbidden_match: Annotated[bool, Field(strict=True)] = False

    @model_validator(mode="after")
    def _safe_identity(self) -> Self:
        if self.version_token is not None and self.object_token is None:
            raise ValueError("a captured version requires an object token")
        if not self.scope_valid and (self.object_token is not None or self.version_token is not None):
            raise ValueError("foreign scope cannot expose captured identities")
        if self.output_valid and self.score is None:
            raise ValueError("a valid hit requires a finite score")
        return self


class RecallObservation(MemoryModel):
    """A bounded ranked response and any extra-row contract violation."""

    hits: tuple[RecallHit, ...] = Field(default=(), max_length=1000)
    excess_results: Annotated[bool, Field(strict=True)] = False


class StepObservation(MemoryModel):
    """One slot in a trial ledger, including errors and unexecuted operations."""

    step_id: Name
    kind: Literal["remember", "recall", "forget"]
    status: OperationStatus
    duration_seconds: Seconds = 0.0
    remember: RememberObservation | None = None
    recall: RecallObservation | None = None
    forget: ForgetStatus | None = None
    error_type: Name | None = None
    error_category: Literal["backend", "contract"] | None = None
    unavailable_reason: Literal["factory_error", "prior_operation_error", "missing_trial", "missing_step"] | None = None

    @model_validator(mode="after")
    def _one_payload(self) -> Self:
        payloads = {"remember": self.remember, "recall": self.recall, "forget": self.forget}
        present = {key for key, value in payloads.items() if value is not None}
        if self.status == OperationStatus.COMPLETED:
            if (
                present != {self.kind}
                or self.error_type is not None
                or self.error_category is not None
                or self.unavailable_reason is not None
            ):
                raise ValueError("completed operation requires only its matching payload")
        elif present:
            raise ValueError("unavailable or error operation cannot contain a success payload")
        elif self.status == OperationStatus.ERROR:
            if self.error_type is None or self.error_category is None or self.unavailable_reason is not None:
                raise ValueError("error operation requires only an exception type")
        elif self.unavailable_reason is None or self.error_type is not None or self.error_category is not None:
            raise ValueError("unavailable operation requires only an unavailable reason")
        return self


class CleanupObservation(MemoryModel):
    """Cleanup acknowledgement and bounded emptiness verification per actor."""

    actor: Name
    status: Literal["verified_empty", "not_empty", "error", "disabled", "not_started"]
    deleted_count: Annotated[StrictInt, Field(ge=0)] | None = None
    error_type: Name | None = None


class TrialObservation(MemoryModel):
    """Selected observations from one case/repetition; no raw provider objects."""

    case_id: Name
    repetition: Annotated[StrictInt, Field(ge=0, le=99)]
    operations: tuple[StepObservation, ...] = Field(default=(), max_length=1000)
    cleanup: tuple[CleanupObservation, ...] = Field(default=(), max_length=32)
    factory_error_type: Name | None = None
    backend_calls: Annotated[StrictInt, Field(ge=0, le=1000)] = 0
    cleanup_calls: Annotated[StrictInt, Field(ge=0, le=64)] = 0


class MemoryEvidence(MemoryModel):
    """Portable normalized observations; replay recomputes all derived verdicts."""

    format: Literal["dynamiq.memory.evidence"] = "dynamiq.memory.evidence"
    version: Annotated[StrictInt, Field(ge=1, le=1)] = 1
    run_id: Name
    specification: MemorySpecification
    cleanup_enabled: Annotated[bool, Field(strict=True)] = True
    trials: tuple[TrialObservation, ...] = Field(default=(), max_length=10000)
