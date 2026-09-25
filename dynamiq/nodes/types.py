from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger


class NodeGroup(str, Enum):
    """
    Enumeration of node groups that categorize different types of nodes.

    Each group represents a collection of related node types, providing a higher-level
    classification of the system's components.
    """

    LLMS = "llms"
    OPERATORS = "operators"
    EMBEDDERS = "embedders"
    RANKERS = "rankers"
    CONVERTERS = "converters"
    RETRIEVERS = "retrievers"
    SPLITTERS = "splitters"
    WRITERS = "writers"
    UTILS = "utils"
    TOOLS = "tools"
    AGENTS = "agents"
    AUDIO = "audio"
    VALIDATORS = "validators"
    IMAGES = "images"
    DETECTORS = "detectors"
    TRANSFORMERS = "transformers"
    EXTRACTORS = "extractors"
    FILTERS = "filters"
    PARSERS = "parsers"


class InferenceMode(str, Enum):
    """
    Enumeration of inference types.
    """

    DEFAULT = "DEFAULT"
    XML = "XML"
    FUNCTION_CALLING = "FUNCTION_CALLING"
    STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT"


class InputParamMode(str, Enum):
    """Per-field override mode for a node's optional input parameters.

    Applied to the input_schema model that drives both the agent-facing tool schema
    and execution-time validation (see ``apply_param_modes``).
    """

    REQUIRED = "required"
    HIDDEN = "hidden"


class ActionType(str, Enum):
    """
    Enumeration of action types for tools and nodes used as tools.
    Classifies what kind of action is performed.

    Note: This is distinct from NodeGroup which classifies the node category.
    ActionType indicates the behavior/action being performed.
    """

    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    DATABASE_QUERY = "database_query"
    COMPUTER_USE = "computer_use"
    SEMANTIC_SEARCH = "semantic_search"
    PARALLEL_EXECUTION = "parallel_execution"
    JUDGEMENT = "judgement"


class Behavior(str, Enum):
    RAISE = "raise"
    RETURN = "return"


class ConditionOperator(str, Enum):
    """Enum representing various condition operators."""

    OR = "or"
    AND = "and"
    BOOLEAN_EQUALS = "boolean-equals"
    BOOLEAN_EQUALS_PATH = "boolean-equals-path"
    NUMERIC_EQUALS = "numeric-equals"
    NUMERIC_EQUALS_PATH = "numeric-equals-path"
    NUMERIC_GREATER_THAN = "numeric-greater-than"
    NUMERIC_GREATER_THAN_PATH = "numeric-greater-than-path"
    NUMERIC_GREATER_THAN_OR_EQUALS = "numeric-greater-than-or-equals"
    NUMERIC_GREATER_THAN_OR_EQUALS_PATH = "numeric-greater-than-or-equals-path"
    NUMERIC_LESS_THAN = "numeric-less-than"
    NUMERIC_LESS_THAN_PATH = "numeric-less-than-path"
    NUMERIC_LESS_THAN_OR_EQUALS = "numeric-less-than-or-equals"
    NUMERIC_LESS_THAN_OR_EQUALS_PATH = "numeric-less-than-or-equals-path"
    STRING_EQUALS = "string-equals"
    STRING_EQUALS_PATH = "string-equals-path"
    STRING_GREATER_THAN = "string-greater-than"
    STRING_GREATER_THAN_PATH = "string-greater-than-path"
    STRING_GREATER_THAN_OR_EQUALS = "string-greater-than-or-equals"
    STRING_GREATER_THAN_OR_EQUALS_PATH = "string-greater-than-or-equals-path"
    STRING_LESS_THAN = "string-less-than"
    STRING_LESS_THAN_PATH = "string-less-than-path"
    STRING_LESS_THAN_OR_EQUALS = "string-less-than-or-equals"
    STRING_LESS_THAN_OR_EQUALS_PATH = "string-less-than-or-equals-path"
    STRING_STARTS_WITH = "string-starts-with"
    STRING_ENDS_WITH = "string-ends-with"
    STRING_CONTAINS = "string-contains"
    STRING_REGEXP = "string-regexp"


class ChoiceCondition(BaseModel):
    """Represents a condition."""

    variable: str | None = None
    operator: ConditionOperator | None = None
    value: Any = None
    is_not: bool = False
    operands: list["ChoiceCondition"] | None = None


class ChoiceHitPolicy(str, Enum):
    """Which options of a Choice run: the first whose condition holds, or every one that holds."""

    FIRST = "first"
    ALL = "all"


class DecisionHitPolicy(str, Enum):
    """Which matching rules of a DecisionTable produce its output."""

    FIRST = "first"
    UNIQUE = "unique"
    COLLECT = "collect"


class DecisionAggregation(str, Enum):
    """How a collect DecisionTable folds the outputs of every matching rule."""

    LIST = "list"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    COUNT = "count"


class Authored(BaseModel):
    """A model whose id the user wrote: a rule code, a row id, a field id.

    Findings, matched rules and test coverage are keyed by that id, so a clone keeps it where a node or a
    Choice option gets a new one.
    """

    keeps_id: ClassVar[bool] = True


class NamedField(Authored):
    """A field the user defines by name. `type` uses the Input node vocabulary: string, int, float, bool, Any."""

    id: str = Field(default_factory=generate_uuid)
    name: str
    type: str = "Any"


class SubWorkflowField(NamedField):
    """An Input or Output field of the flow a SubWorkflow runs, as captured when the flow was chosen."""

    required: bool = False


class DecisionRule(Authored):
    """One row of a DecisionTable: a condition cell per input column and a value cell per output column."""

    id: str = Field(default_factory=generate_uuid)
    name: str = ""
    when: list[str | int | float | bool | None] = []
    then: list[str | int | float | bool | None] = []
    enabled: bool = True


class ExpressionItem(Authored):
    """One output of an Expression node: the key it is returned under and the expression that computes it."""

    id: str = Field(default_factory=generate_uuid)
    key: str
    expression: str


class RuleSeverity(str, Enum):
    """What a check that does not hold means: `fail` blocks, `warn` flags for attention, `info` notes."""

    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


class RuleMissingPolicy(str, Enum):
    """What a rule reports when a value it reads is missing.

    `not_evaluated` holds the rule as a finding to review. `fail` reports the rule's own severity. `not_applicable`
    skips the rule for a record without the value, as `applies_when` would, so the rule needs no presence guard.
    Only data the record lacks is skipped. A value that is there but cannot be read, a lookup that found nothing, a
    name the node does not declare, a call of a name no helper has or a filter or a test no sandbox has is still
    `not_evaluated`, or the severity under `fail`, whatever the policy; under `not_applicable` the finding's reason
    says why it was not skipped.
    """

    NOT_EVALUATED = "not_evaluated"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class DerivedValue(Authored):
    """A value a Rules node computes once per record, before its rules run, and exposes to them by name.

    One that cannot be computed comes out as `None`; the node's `derived_errors` names the reason when the
    cause is the expression's own rather than a value the record lacks.
    """

    id: str = Field(default_factory=generate_uuid)
    name: str
    expression: str


class Rule(Authored):
    """One check of a Rules node.

    `check` is an expression that must hold for the rule to pass; `applies_when` is an optional precondition,
    and a rule that does not apply reports `not_applicable`. `message` is a template rendered with the whole
    record when the check does not hold. `effective_from` and `effective_until` are ISO dates; outside the
    window the rule is not applicable for the record's `as_of` date. `on_missing` says what a missing value
    means for this rule, overriding the node's policy; unset, or saved empty, it leaves that to the node, and so
    does a value that is no policy, with a warning. Unset, it is left out when the rule is serialized, so a rule
    saved before rules had a policy of their own serializes as it did.
    """

    id: str = Field(default_factory=generate_uuid)
    name: str = ""
    category: str = ""
    severity: RuleSeverity = RuleSeverity.FAIL
    applies_when: str | None = None
    check: str = ""
    message: str | None = None
    reason_code: str | None = None
    references: list[str] = []
    tags: list[str] = []
    effective_from: str | None = None
    effective_until: str | None = None
    enabled: bool = True
    # Left out of a dump while unset, rather than written as null: every flow saved before the key existed would
    # otherwise read as changed. `exclude_if` keeps the JSON schema as it is, which a model serializer would not.
    on_missing: RuleMissingPolicy | None = Field(default=None, exclude_if=lambda policy: policy is None)

    @field_validator("on_missing", mode="before")
    @classmethod
    def unknown_policy_is_unset(cls, value: Any, info: ValidationInfo) -> Any:
        """A policy left empty, as the editor saves it, leaves the choice to the node, as None does.

        So does a value that is no policy, `skip` or a typo, with a warning naming it: before rules had a policy of
        their own the key was ignored, and a rule that carries one must keep building.
        """
        if isinstance(value, str) and not value.strip():
            return None
        allowed = [policy.value for policy in RuleMissingPolicy]
        # A member of the enum equals its value, so it passes here as well.
        if value is None or value in allowed:
            return value
        logger.warning(
            f"Rule {info.data.get('id')!r}: on_missing {value!r} is not one of {', '.join(allowed)}, "
            "so the node's policy applies."
        )
        return None
