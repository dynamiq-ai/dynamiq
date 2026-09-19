import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, ClassVar, Literal, NamedTuple
from uuid import uuid4

from jinja2 import ChainableUndefined, Template, TemplateSyntaxError, Undefined, nodes
from jinja2.exceptions import UndefinedError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import DerivedValue, NamedField, Rule, RuleMissingPolicy
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.utils import TRUNCATE_LIST_LIMIT

STATUS_PASSED = "pass"
STATUS_FAIL = "fail"
STATUS_WARN = "warn"
STATUS_INFO = "info"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUSES = (STATUS_PASSED, STATUS_FAIL, STATUS_WARN, STATUS_INFO, STATUS_NOT_APPLICABLE, STATUS_NOT_EVALUATED)
AS_OF_KEY = "as_of"

# A check that reaches a missing value, a value of the wrong kind or a helper that refuses its input is a
# finding to review, never a crash of the run: the record is what it is, and the rule says what it needs.
EVALUATION_ERRORS = (UndefinedError, TypeError, ValueError, ArithmeticError, AttributeError, LookupError)
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_INDEX = re.compile(r"^(.*)\[(-?\d+)\]$")
_EXEMPT_TESTS = frozenset({"defined", "undefined", "none", "sameas"})

_MISSING = object()


def _is_missing(value: Any) -> bool:
    return value is None or value is _MISSING or isinstance(value, Undefined)


def has(value: Any) -> bool:
    """True when a value is present: defined and not null."""
    return not _is_missing(value)


def to_date(value: Any) -> date:
    """Reads a date from a date, a datetime, an ISO string or a US `MM/DD/YYYY` string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if _ISO_DATE.match(text):
            return date.fromisoformat(text[:10])
        if us := _US_DATE.match(text):
            month, day, year = (int(part) for part in us.groups())
            return date(year, month, day)
    raise ValueError(f"not a date: {value!r}")


def days_between(start: Any, end: Any) -> int:
    """The number of days from `start` to `end`, negative when `end` comes first."""
    return (to_date(end) - to_date(start)).days


def today() -> date:
    return date.today()


HELPERS: dict[str, Callable[..., Any]] = {
    "has": has,
    "days_between": days_between,
    "date": to_date,
    "today": today,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
}

# One sandbox for every Rules node. A missing attribute stays undefined instead of raising, so a check can ask
# `has(docs.FloodCert)` about a document that is not there; a comparison with an undefined value still raises,
# which the node reports as a finding to review.
_ENVIRONMENT = ImmutableSandboxedEnvironment(undefined=ChainableUndefined)
_ENVIRONMENT.globals.update(HELPERS)


def _path_of(node: nodes.Node) -> str | None:
    """The dotted path an attribute chain reads, or None when it is not a plain chain."""
    if isinstance(node, nodes.Name):
        return node.name
    if isinstance(node, nodes.Getattr):
        base = _path_of(node.node)
        return f"{base}.{node.attr}" if base else None
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        base = _path_of(node.node)
        if base is None:
            return None
        key = node.arg.value
        if isinstance(key, str):
            return f"{base}.{key}"
        if isinstance(key, int) and not isinstance(key, bool):
            return f"{base}[{key}]"
    return None


class Reads(NamedTuple):
    """The paths an expression reads: the ones it needs, and the ones it only asks about."""

    required: list[str]
    optional: list[str]


def _collect_paths(node: nodes.Node, reads: Reads, required: bool) -> None:
    # A value asked about with `has`, `is defined` or `default` is allowed to be missing.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name) and node.node.name == "has":
        for argument in node.args:
            _collect_paths(argument, reads, required=False)
        return
    if isinstance(node, nodes.Test) and node.name in _EXEMPT_TESTS:
        _collect_paths(node.node, reads, required=False)
        return
    if isinstance(node, nodes.Filter) and node.name == "default":
        _collect_paths(node.node, reads, required=False)
        for argument in node.args:
            _collect_paths(argument, reads, required)
        return
    path = _path_of(node)
    if path is not None:
        target = reads.required if required else reads.optional
        if path.split(".")[0].split("[")[0] not in _ENVIRONMENT.globals and path not in target:
            target.append(path)
        return
    for child in node.iter_child_nodes():
        _collect_paths(child, reads, required)


def read_paths(expression: str) -> Reads:
    """The paths an expression reads, in order of appearance.

    A path the expression only asks `has`, `is defined` or `default` about is optional: it may be missing
    without stopping the evaluation. Every other path is required.
    """
    reads = Reads(required=[], optional=[])
    _collect_paths(_ENVIRONMENT.parse("{{ " + expression + " }}"), reads, required=True)
    return reads


def _split_path(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for piece in path.split("."):
        indexes: list[int] = []
        while indexed := _INDEX.match(piece):
            piece = indexed.group(1)
            indexes.insert(0, int(indexed.group(2)))
        if piece:
            parts.append(piece)
        parts.extend(indexes)
    return parts


def resolve_path(context: dict[str, Any], path: str) -> Any:
    """The value at a dotted path in the context, or the missing marker when any step is absent."""
    current: Any = context
    for part in _split_path(path):
        if isinstance(part, int):
            if isinstance(current, (list, tuple)) and -len(current) <= part < len(current):
                current = current[part]
            else:
                return _MISSING
        elif isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif hasattr(current, part) and not part.startswith("_"):
            current = getattr(current, part)
        else:
            return _MISSING
    return _MISSING if isinstance(current, Undefined) else current


def _shown(value: Any) -> Any:
    """A value as a finding shows it: scalars as they are, containers as their size."""
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return f"{{…{len(value)} keys}}"
    if isinstance(value, (list, tuple)):
        return f"[…{len(value)} items]"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class CompiledRule(NamedTuple):
    rule: Rule
    applies: Callable[..., Any] | None
    applies_reads: Reads
    check: Callable[..., Any]
    check_reads: Reads
    message: Template | None
    effective_from: date | None
    effective_until: date | None


class RulesInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Rules(Node):
    """Evaluates every rule against the inputs and returns one finding per rule.

    Inputs arrive by name and rules read them by path (`docs.Note.interest_rate`), so a record of any shape
    needs no mapping beyond naming it. Derived values are computed once per record, in order, before the rules
    run, and are read by name like an input. Expressions use the same sandboxed engine as the Expression node,
    plus `has`, `days_between`, `date`, `today`, `len`, `abs`, `min`, `max`, `sum` and `round`.

    Every enabled rule reports a status: `pass` when its check holds; its severity (`fail`, `warn`, `info`)
    when the check does not; `not_applicable` when `applies_when` does not hold or the record's `as_of` date
    falls outside the rule's effective window; `not_evaluated` when a value the check reads is missing or the
    check cannot be evaluated, unless `on_missing` says to report the rule's severity instead. A missing value
    never passes or fails a rule silently. The message is rendered with the values the check read, which the
    finding also carries under `evaluated`. Rules compile when the node is built, so a malformed expression
    fails then, naming the rule.

    The output holds `findings` in rule order, a `summary` of statuses, `status` (`fail` if any rule failed,
    else `warn` if any warned, else `pass`) and the `derived` values. An optional `as_of` input, an ISO date,
    fixes the date the effective windows are compared with; without it the run date is used.
    """

    name: str | None = "rules"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    input_fields: list[NamedField] = []
    derived_values: list[DerivedValue] = []
    rules: list[Rule] = []
    on_missing: RuleMissingPolicy = RuleMissingPolicy.NOT_EVALUATED
    input_schema: ClassVar[type[RulesInputSchema]] = RulesInputSchema

    _compiled: list[CompiledRule] = PrivateAttr(default_factory=list)
    _derived: list[tuple[str, Callable[..., Any]]] = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._derived = self._compile_derived()
        self._compiled = self._compile_rules()

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"rules": True}

    def to_dict(self, include_secure_params: bool = True, for_tracing: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        A trace keeps the first rules and the total: the findings carry every rule that mattered, and a large
        rule set would otherwise be copied into every run it takes part in.
        """
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        rules = self.rules[:TRUNCATE_LIST_LIMIT] if for_tracing else self.rules
        data["rules"] = [rule.model_dump(**kwargs) for rule in rules]
        if for_tracing:
            data["rules_count"] = len(self.rules)
        return data

    def _compile_expression(self, text: str, where: str) -> Callable[..., Any]:
        try:
            return _ENVIRONMENT.compile_expression(text, undefined_to_none=True)
        except TemplateSyntaxError as e:
            raise ValueError(f"{where} is not a valid expression: {e}") from e

    def _compile_derived(self) -> list[tuple[str, Callable[..., Any]]]:
        taken = {field.name for field in self.input_fields}
        compiled = []
        for value in self.derived_values:
            label = f"Rules '{self.name}': derived value {value.name!r}"
            if not value.name.isidentifier():
                raise ValueError(f"{label} is not a valid identifier")
            if value.name in taken or value.name in HELPERS:
                raise ValueError(f"{label} is already the name of an input or a helper")
            if not value.expression.strip():
                raise ValueError(f"{label} has no expression")
            taken.add(value.name)
            compiled.append((value.name, self._compile_expression(value.expression, label)))
        return compiled

    def _compile_rules(self) -> list[CompiledRule]:
        ids: set[str] = set()
        compiled = []
        for index, rule in enumerate(self.rules, start=1):
            label = f"Rules '{self.name}', rule {index}" + (f" ({rule.name})" if rule.name else "")
            if rule.id in ids:
                raise ValueError(f"{label}: id {rule.id!r} is used twice")
            ids.add(rule.id)
            if not rule.enabled:
                continue
            if not rule.check.strip():
                raise ValueError(f"{label}: the check is empty")
            applies = rule.applies_when.strip() if rule.applies_when else ""
            effective_from = self._effective_date(rule.effective_from, f"{label}: effective_from")
            effective_until = self._effective_date(rule.effective_until, f"{label}: effective_until")
            if effective_from and effective_until and effective_until < effective_from:
                raise ValueError(f"{label}: the effective window ends before it starts")
            try:
                message = _ENVIRONMENT.from_string(rule.message) if rule.message and rule.message.strip() else None
            except TemplateSyntaxError as e:
                raise ValueError(f"{label}: the message is not a valid template: {e}") from e
            compiled.append(
                CompiledRule(
                    rule=rule,
                    applies=self._compile_expression(applies, f"{label}: applies_when") if applies else None,
                    applies_reads=read_paths(applies) if applies else Reads(required=[], optional=[]),
                    check=self._compile_expression(rule.check, f"{label}: the check"),
                    check_reads=read_paths(rule.check),
                    message=message,
                    effective_from=effective_from,
                    effective_until=effective_until,
                )
            )
        return compiled

    @staticmethod
    def _effective_date(text: str | None, where: str) -> date | None:
        if not text or not text.strip():
            return None
        try:
            return to_date(text)
        except ValueError as e:
            raise ValueError(f"{where} is not a date: {text!r}") from e

    def execute(self, input_data: RulesInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Evaluates every rule against the inputs and returns the findings."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **{**kwargs, "parent_run_id": kwargs.get("run_id", uuid4())})

        context = input_data.model_dump()
        as_of = self._as_of(context.get(AS_OF_KEY))
        derived: dict[str, Any] = {}
        for name, expression in self._derived:
            try:
                derived[name] = expression(**context, **derived)
            except EVALUATION_ERRORS:
                derived[name] = None
        scope = {**context, **derived}

        findings = [self._evaluate(compiled, scope, as_of) for compiled in self._compiled]
        summary = {status: 0 for status in STATUSES}
        for finding in findings:
            summary[finding["status"]] += 1
        status = STATUS_FAIL if summary[STATUS_FAIL] else STATUS_WARN if summary[STATUS_WARN] else STATUS_PASSED
        return {"status": status, "summary": summary, "findings": findings, "derived": derived}

    @staticmethod
    def _as_of(value: Any) -> date:
        if _is_missing(value) or value == "":
            return date.today()
        try:
            return to_date(value)
        except ValueError as e:
            raise ValueError(f"Rules: '{AS_OF_KEY}' is not a date: {value!r}") from e

    def _evaluate(self, compiled: CompiledRule, scope: dict[str, Any], as_of: date) -> dict[str, Any]:
        rule = compiled.rule
        finding: dict[str, Any] = {
            "rule_id": rule.id,
            "name": rule.name,
            "category": rule.category,
            "severity": rule.severity.value,
            "status": STATUS_PASSED,
            "message": None,
            "reason_code": rule.reason_code,
            "references": list(rule.references),
            "tags": list(rule.tags),
            "evaluated": {},
        }
        if (compiled.effective_from and as_of < compiled.effective_from) or (
            compiled.effective_until and as_of > compiled.effective_until
        ):
            finding["status"] = STATUS_NOT_APPLICABLE
            return finding

        status, reason = self._status(compiled, scope)
        finding["status"] = status
        if status != STATUS_NOT_APPLICABLE:
            reads = compiled.check_reads.required + compiled.check_reads.optional
            finding["evaluated"] = {path: _shown(resolve_path(scope, path)) for path in reads}
        if status == STATUS_NOT_EVALUATED:
            finding["message"] = reason
        elif status != STATUS_PASSED and status != STATUS_NOT_APPLICABLE:
            finding["message"] = self._render(compiled, scope, reason)
        return finding

    def _status(self, compiled: CompiledRule, scope: dict[str, Any]) -> tuple[str, str | None]:
        if compiled.applies is not None:
            if missing := self._missing(compiled.applies_reads.required, scope):
                return self._missing_status(compiled, f"missing value for {missing}")
            try:
                applies = bool(compiled.applies(**scope))
            except EVALUATION_ERRORS as e:
                return self._missing_status(compiled, f"applies_when could not be evaluated: {e}")
            if not applies:
                return STATUS_NOT_APPLICABLE, None
        if missing := self._missing(compiled.check_reads.required, scope):
            return self._missing_status(compiled, f"missing value for {missing}")
        try:
            holds = bool(compiled.check(**scope))
        except EVALUATION_ERRORS as e:
            return self._missing_status(compiled, f"check could not be evaluated: {e}")
        return (STATUS_PASSED, None) if holds else (compiled.rule.severity.value, None)

    def _missing_status(self, compiled: CompiledRule, reason: str) -> tuple[str, str]:
        if self.on_missing == RuleMissingPolicy.FAIL:
            return compiled.rule.severity.value, reason
        return STATUS_NOT_EVALUATED, reason

    @staticmethod
    def _missing(paths: list[str], scope: dict[str, Any]) -> str | None:
        for path in paths:
            if _is_missing(resolve_path(scope, path)):
                return path
        return None

    @staticmethod
    def _render(compiled: CompiledRule, scope: dict[str, Any], reason: str | None) -> str | None:
        if compiled.message is None:
            return reason
        try:
            rendered = compiled.message.render(**scope).strip()
        except EVALUATION_ERRORS:
            rendered = compiled.rule.message.strip() if compiled.rule.message else ""
        if reason:
            return f"{rendered} ({reason})" if rendered else reason
        return rendered or None
