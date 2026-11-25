"""Strict scenarios for evaluating the public long-term memory lifecycle."""

from enum import Enum
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from dynamiq.memory.long_term.types import ForgetStatus, RememberOutcome


def _portable(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("strings must contain portable Unicode") from exc
    return value


Name = Annotated[StrictStr, Field(min_length=1, max_length=256), AfterValidator(_portable)]
Text = Annotated[StrictStr, Field(min_length=1, max_length=1000000), AfterValidator(_portable)]
Count = Annotated[StrictInt, Field(ge=1, le=1000000)]
UnitValue = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class MemoryModel(BaseModel):
    """Immutable snapshots reject unknown fields instead of ignoring typos."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class Verdict(str, Enum):
    """Requirements on supplied observations, not a population guarantee."""

    PASS = "pass"  # nosec B105 - evaluation verdict, not a password
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    NOT_EVALUATED = "not_evaluated"


class OperationStatus(str, Enum):
    """Execution availability remains separate from assertion verdicts."""

    COMPLETED = "completed"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class MemoryMetric(str, Enum):
    """Cutoff-specific metrics; adverse rates explicitly name their query unit."""

    PRECISION = "precision"
    RECALL = "recall"
    RECIPROCAL_RANK = "reciprocal_rank"
    NDCG = "ndcg"
    HIT_RATE = "hit_rate"
    FORBIDDEN_QUERY_RATE = "forbidden_query_rate"
    DUPLICATE_QUERY_RATE = "duplicate_query_rate"
    UNKNOWN_QUERY_RATE = "unknown_query_rate"
    SCOPE_VIOLATION_QUERY_RATE = "scope_violation_query_rate"

    @property
    def improvement_sign(self) -> int:
        """Orient candidate-minus-baseline so positive always means better."""
        return -1 if self.value.endswith("_query_rate") else 1


class Relevance(MemoryModel):
    """An earlier captured memory version and its predeclared relevance grade."""

    alias: Name
    grade: Annotated[StrictInt, Field(ge=0, le=3)]


class Remember(MemoryModel):
    """Remember text and bind a new immutable alias to the returned object/version.

    Native matching permits the earlier spelling on case-insensitive dedup;
    exact matching requires the returned text to equal the stripped request.
    """

    kind: Literal["remember"] = "remember"
    id: Name
    actor: Name
    alias: Name
    content: Text
    write_match: Literal["native", "exact"] = "native"
    expected_outcome: RememberOutcome | None = None
    expected_same_as: Name | None = None

    @model_validator(mode="after")
    def _nonblank_content(self) -> Self:
        if not self.content.strip():
            raise ValueError("memory content must contain non-whitespace text")
        return self


class Recall(MemoryModel):
    """Search one actor's memories without reordering backend results."""

    kind: Literal["recall"] = "recall"
    id: Name
    actor: Name
    query: Text
    limit: Annotated[StrictInt, Field(ge=1, le=1000)] = 10
    relevance: tuple[Relevance, ...] = Field(default=(), max_length=1000)
    forbidden: tuple[Name, ...] = Field(default=(), max_length=1000)
    expect_empty: Annotated[bool, Field(strict=True)] | None = None

    @model_validator(mode="after")
    def _valid_judgments(self) -> Self:
        if not self.query.strip():
            raise ValueError("recall query must contain non-whitespace text")
        aliases = [item.alias for item in self.relevance]
        if len(set(aliases)) != len(aliases):
            raise ValueError("duplicate relevance alias")
        if len(set(self.forbidden)) != len(self.forbidden):
            raise ValueError("duplicate forbidden alias")
        positive = {item.alias for item in self.relevance if item.grade > 0}
        if positive.intersection(self.forbidden):
            raise ValueError("a positive relevance alias cannot be forbidden")
        if self.expect_empty is True and positive:
            raise ValueError("empty retrieval cannot also require relevant memories")
        return self


class Forget(MemoryModel):
    """Delete the current object at an earlier alias's ID, not only its version.

    An alias from a different actor can exercise an explicit forbidden-delete
    control. It still targets only data created inside the evaluation trial.
    """

    kind: Literal["forget"] = "forget"
    id: Name
    actor: Name
    alias: Name
    expected_status: ForgetStatus = ForgetStatus.DELETED


MemoryStep = Annotated[Remember | Recall | Forget, Field(discriminator="kind")]


class MemoryCase(MemoryModel):
    """An ordered lifecycle with logical actors isolated into generated scopes."""

    id: Name
    actors: tuple[Name, ...] = Field(default=("subject",), min_length=1, max_length=32)
    steps: tuple[MemoryStep, ...] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def _validate_sequence(self) -> Self:
        if len(set(self.actors)) != len(self.actors):
            raise ValueError("duplicate actor")
        seen_steps: set[str] = set()
        aliases: dict[str, str] = {}
        for step in self.steps:
            if step.id in seen_steps:
                raise ValueError("duplicate step ID")
            seen_steps.add(step.id)
            if step.actor not in self.actors:
                raise ValueError("step references an undeclared actor")
            if isinstance(step, Remember):
                if step.alias in aliases:
                    raise ValueError("duplicate memory alias")
                if step.expected_same_as is not None:
                    if step.expected_same_as not in aliases:
                        raise ValueError("expected_same_as must reference an earlier alias")
                    if aliases[step.expected_same_as] != step.actor:
                        raise ValueError("expected_same_as must belong to the same actor")
                aliases[step.alias] = step.actor
            elif isinstance(step, Forget):
                if step.alias not in aliases:
                    raise ValueError("forget must reference an earlier alias")
            else:
                for judgment in step.relevance:
                    if judgment.alias not in aliases:
                        raise ValueError("relevance must reference an earlier alias")
                    if aliases[judgment.alias] != step.actor:
                        raise ValueError("positive or zero relevance must belong to the recall actor")
                if any(alias not in aliases for alias in step.forbidden):
                    raise ValueError("forbidden judgments must reference earlier aliases")
        return self


class MemoryGroup(MemoryModel):
    """A fixed subset of cases, selected before any outcomes are observed."""

    name: Name
    cases: tuple[Name, ...] = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _unique_cases(self) -> Self:
        if self.name == "overall":
            raise ValueError("overall is a reserved group name")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("duplicate group case")
        return self


class MemoryGate(MemoryModel):
    """Explicit observed metric bounds with availability and support requirements."""

    metric: MemoryMetric
    cutoff: Annotated[StrictInt, Field(ge=1, le=1000)]
    group: Name = "overall"
    minimum: UnitValue | None = None
    maximum: UnitValue | None = None
    minimum_support: Count = 1
    minimum_coverage: UnitValue = 1.0

    @model_validator(mode="after")
    def _valid_bounds(self) -> Self:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a gate requires a minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        return self


class MemoryEvaluationLimits(MemoryModel):
    """Trusted application ceilings, never controlled by a portable suite."""

    max_cases: Count = 2000
    max_trials: Count = 10000
    max_operations: Count = 100000
    max_cleanup_calls: Count = 100000
    max_returned_hits: Count = 1000000
    max_group_observations: Count = 1000000
    max_metric_values: Count = 100000
    max_bytes: Annotated[StrictInt, Field(ge=1, le=2**53 - 1)] = 64 * 1024 * 1024
    max_json_nodes: Count = 1000000
    max_json_depth: Annotated[StrictInt, Field(ge=1, le=100)] = 30
    max_detail_rows: Count = 10000


class MemorySuite(MemoryModel):
    """Declare the complete expected ledger before creating backend instances."""

    name: Name
    dataset_revision: Name | None = None
    cases: tuple[MemoryCase, ...] = Field(min_length=1, max_length=2000)
    cutoffs: tuple[Annotated[StrictInt, Field(ge=1, le=1000)], ...] = Field(default=(1, 5), min_length=1, max_length=32)
    repetitions: Annotated[StrictInt, Field(ge=1, le=100)] = 1
    groups: tuple[MemoryGroup, ...] = Field(default=(), max_length=50)
    gates: tuple[MemoryGate, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def _valid_panel(self) -> Self:
        case_ids = {case.id for case in self.cases}
        if len(case_ids) != len(self.cases):
            raise ValueError("duplicate case ID")
        if len(set(self.cutoffs)) != len(self.cutoffs):
            raise ValueError("duplicate cutoff")
        names = {group.name for group in self.groups}
        if len(names) != len(self.groups):
            raise ValueError("duplicate group name")
        for group in self.groups:
            if not set(group.cases).issubset(case_ids):
                raise ValueError("group references an unknown case")
        maximum = max(self.cutoffs)
        for case in self.cases:
            for step in case.steps:
                if isinstance(step, Recall) and step.limit < maximum:
                    raise ValueError("recall limit must cover every configured cutoff")
        keys = set()
        for gate in self.gates:
            if gate.group != "overall" and gate.group not in names:
                raise ValueError("gate references an unknown group")
            if gate.cutoff not in self.cutoffs:
                raise ValueError("gate references an unconfigured cutoff")
            key = (gate.group, gate.metric, gate.cutoff)
            if key in keys:
                raise ValueError("duplicate metric/cutoff/group gate")
            keys.add(key)
        return self

    def check_limits(self, limits: MemoryEvaluationLimits) -> None:
        """Reject expensive declarations before reading input or calling a factory."""
        operations = sum(len(case.steps) for case in self.cases) * self.repetitions
        cleanup_calls = sum(len(case.actors) for case in self.cases) * self.repetitions * 2
        hits = sum(step.limit for case in self.cases for step in case.steps if isinstance(step, Recall))
        metric_values = sum(isinstance(step, Recall) for case in self.cases for step in case.steps)
        metric_values *= self.repetitions * len(self.cutoffs) * len(MemoryMetric)
        memberships = {case.id: 1 for case in self.cases}
        for group in self.groups:
            for case_id in group.cases:
                memberships[case_id] += 1
        observations = (
            sum(
                max(1, sum(isinstance(step, Recall) for step in case.steps)) * memberships[case.id]
                for case in self.cases
            )
            * self.repetitions
            * len(self.cutoffs)
            * len(MemoryMetric)
        )
        for actual, maximum, label in (
            (len(self.cases), limits.max_cases, "case"),
            (len(self.cases) * self.repetitions, limits.max_trials, "trial"),
            (operations, limits.max_operations, "operation"),
            (cleanup_calls, limits.max_cleanup_calls, "cleanup"),
            (hits * self.repetitions, limits.max_returned_hits, "returned hit"),
            (observations, limits.max_group_observations, "group observation"),
            (metric_values, limits.max_metric_values, "metric value"),
        ):
            if actual > maximum:
                raise ValueError(f"memory evaluation {label} limit exceeded")
        if len(self.model_dump_json().encode("utf-8")) > limits.max_bytes:
            raise ValueError("memory evaluation suite byte limit exceeded")
