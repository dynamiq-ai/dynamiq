import re
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import date, datetime
from typing import Any, ClassVar, Literal, NamedTuple
from uuid import uuid4

from jinja2 import ChainableUndefined, Template, TemplateSyntaxError, Undefined, nodes
from jinja2.exceptions import TemplateRuntimeError
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
# Jinja's own runtime errors count as well: an undefined value used, a filter given the wrong argument, and
# an attribute the sandbox refuses, `append` on a list member say, whose use raises a SecurityError.
EVALUATION_ERRORS = (TemplateRuntimeError, TypeError, ValueError, ArithmeticError, AttributeError, LookupError)
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# One path segment: a name, a numeric index, or a quoted key with escapes.
_SEGMENT = re.compile(r"""\.?([^.\[\]]+)|\[(?:(-?\d+)|'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\]""")
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


class RuleUndefined(ChainableUndefined):
    """A value that is not there.

    Attribute and item access chain, so `has(docs.FloodCert.pages)` can ask about a document that is missing; a
    comparison, a truth test, a count or a loop over the value raises instead, so a lookup such as
    `limits[loan.program]` for a program the table lacks makes the rule not evaluated rather than quietly true or
    false. Rendering it in a message gives an empty string.
    """

    __slots__ = ()
    __eq__ = __ne__ = __bool__ = __hash__ = Undefined._fail_with_undefined_error
    __len__ = __iter__ = __contains__ = Undefined._fail_with_undefined_error


class RecordSandbox(ImmutableSandboxedEnvironment):
    """The immutable sandbox with a record's keys read before an object's attributes.

    Jinja reads `a.b` attribute first, so over a dict record `invoice.items` would be the dict's method and
    `ticket.update`, a mutating method the sandbox refuses, an undefined that raises on use. A record is data:
    a dotted read takes the key the mapping holds, the way `resolve_path` and a finding's `evaluated` read it,
    and reaches an attribute only for a key the mapping lacks, which is what `invoice.get('vat_rate', 0)`
    relies on.
    """

    def getattr(self, obj: Any, attribute: str) -> Any:
        if isinstance(obj, Mapping):
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                pass
        return super().getattr(obj, attribute)


# One sandbox for every Rules node; the expressions it compiles are stateless.
_ENVIRONMENT = RecordSandbox(undefined=RuleUndefined)


def concrete(value: Any) -> Any:
    """Returns the value with every undefined member replaced by None and every lazy iterable materialized.

    Jinja turns a result into None only when the whole result is undefined. A list or a dict the expression
    builds keeps the undefined objects inside it, `map(attribute=...)` over items that lack the attribute above
    all, and such an object is not serializable and raises on its first use downstream. `map`, `select`,
    `selectattr`, `reject` and `rejectattr` return generators, which the first rule to read one exhausts for
    every rule after it, and which no encoder can record, so an iterator, a dict view, a range or a set becomes
    a list; a string and an object that merely iterates, a document say, stay what they are.
    """
    if isinstance(value, Undefined):
        return None
    if isinstance(value, dict):
        return {key: concrete(item) for key, item in value.items()}
    if isinstance(value, list):
        return [concrete(item) for item in value]
    if isinstance(value, tuple):
        return tuple(concrete(item) for item in value)
    if isinstance(value, (Iterator, KeysView, ValuesView, ItemsView, range, set, frozenset)):
        return [concrete(item) for item in value]
    return value


def holds(value: Any) -> bool:
    """The truth of a check or a condition.

    A lazy result is judged by the list it yields, since a generator is true whatever it would yield; an
    undefined result keeps raising, which is what makes a lookup that found nothing `not_evaluated`.
    """
    if isinstance(value, Undefined):
        return bool(value)
    return bool(concrete(value))


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
            # A key that is not a plain name, one with a dot in it say, keeps the subscript form, which the
            # splitter reads back as one segment, so `docs['Flood.Cert']` is not read as `docs.Flood.Cert`.
            if _PLAIN_KEY.match(key):
                return f"{base}.{key}"
            escaped = key.replace("\\", "\\\\").replace("'", "\\'")
            return f"{base}['{escaped}']"
        if isinstance(key, int) and not isinstance(key, bool):
            return f"{base}[{key}]"
    return None


# The names the sandbox provides on its own: the helpers and Jinja's own globals (`range`, `dict`, ...).
GLOBAL_NAMES = frozenset(_ENVIRONMENT.globals)


class Reads(NamedTuple):
    """The paths an expression reads: the ones it needs, the ones it only asks about, and the global names it
    calls and reads as values, which a record key of the same name would shadow."""

    required: list[str]
    optional: list[str]
    helpers_called: tuple[str, ...] = ()
    helpers_read: tuple[str, ...] = ()


class _Collected(NamedTuple):
    required: list[str]
    optional: list[str]
    called: list[str]


def _root(path: str) -> str:
    return path.split(".")[0].split("[")[0]


def _collect_paths(node: nodes.Node, collected: _Collected, required: bool) -> None:
    # A call of a helper reads its arguments, never a member of the helper's name; a value asked about with
    # `has`, `is defined` or `default` is allowed to be missing.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name) and node.node.name in GLOBAL_NAMES:
        collected.called.append(node.node.name)
        for child in node.iter_child_nodes(exclude=("node",)):
            _collect_paths(child, collected, required and node.node.name != "has")
        return
    if isinstance(node, nodes.Test) and node.name in _EXEMPT_TESTS:
        _collect_paths(node.node, collected, required=False)
        return
    if isinstance(node, nodes.Filter) and node.name == "default":
        _collect_paths(node.node, collected, required=False)
        for argument in node.args:
            _collect_paths(argument, collected, required)
        return
    # A method call reads the object it is called on, not a member of the method's name: `invoice.get('vat_rate')`
    # needs `invoice`, and a dict holds no key called `get`.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Getattr):
        _collect_paths(node.node.node, collected, required)
        for child in node.iter_child_nodes(exclude=("node",)):
            _collect_paths(child, collected, required)
        return
    path = _path_of(node)
    if path is not None:
        target = collected.required if required else collected.optional
        if path not in target:
            target.append(path)
        return
    for child in node.iter_child_nodes():
        _collect_paths(child, collected, required)


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def _reads_of(parsed: nodes.Template) -> Reads:
    collected = _Collected(required=[], optional=[], called=[])
    _collect_paths(parsed, collected, required=True)
    required: list[str] = []
    optional = list(collected.optional)
    for path in collected.required:
        if any(_is_under(path, guarded) for guarded in collected.optional):
            if path not in optional:
                optional.append(path)
        else:
            required.append(path)
    roots = dict.fromkeys(_root(path) for path in required + optional)
    return Reads(
        required=required,
        optional=optional,
        helpers_called=tuple(dict.fromkeys(collected.called)),
        helpers_read=tuple(root for root in roots if root in GLOBAL_NAMES),
    )


def read_paths(expression: str) -> Reads:
    """The paths an expression reads, in order of appearance.

    A path the expression only asks `has`, `is defined` or `default` about is optional: it may be missing
    without stopping the evaluation, and so may anything read under it, since `has(docs.FloodCert) and
    docs.FloodCert.zone == 'A'` is how a check guards a read; the guard decides, not a pre-check. Every other
    path is required. A helper's name is never a read: `days_between(a, b)` reads `a` and `b`, while a bare
    `date` is a member of the record, whatever the record holds under it.
    """
    return _reads_of(_ENVIRONMENT.parse("{{ " + expression + " }}"))


def read_template(template: str) -> Reads:
    """The paths a message template reads, the way `read_paths` reads an expression."""
    return _reads_of(_ENVIRONMENT.parse(template))


def scope_for(reads: Reads, scope: dict[str, Any], undefined: type[Undefined]) -> dict[str, Any]:
    """The scope one expression evaluates in.

    A record key named like a helper (`date`, say) is an ordinary member where the expression reads it as a
    value and stays out of the way where the expression calls the helper, so a record that carries `date` and a
    sibling rule that calls `date(...)` both work. A member the expression reads but the record lacks is undefined
    rather than the helper, so `has(date)` never passes on the helper's presence.
    """
    shadowed = [name for name in reads.helpers_called if name in scope]
    absent = [name for name in reads.helpers_read if name not in scope]
    if not shadowed and not absent:
        return scope
    scoped = {name: value for name, value in scope.items() if name not in shadowed}
    for name in absent:
        scoped[name] = undefined(name=name)
    return scoped


RESERVED_ROOT = "self"


def reserved_read(reads: Reads) -> str | None:
    """Returns the first path read from the one name an expression cannot read.

    Jinja binds `self` to its template reference inside every compiled expression and template, so a top-level
    key of that name is never the value handed over: `has(self)` finds the reference and passes on a record
    without the key, and `self.id` raises about a Jinja internal. A `self` nested inside a record reads like
    any other key.
    """
    return next((path for path in reads.required + reads.optional if _root(path) == RESERVED_ROOT), None)


def refuse_reserved_read(reads: Reads, where: str) -> None:
    """Raises when the expression reads a top-level `self`, naming the fix."""
    if reserved := reserved_read(reads):
        raise ValueError(
            f"{where} reads {reserved!r}: Jinja reserves the name 'self' inside an expression, so a top-level key "
            "of that name cannot be read; nest it inside a record or rename the input"
        )


def _private_segment(paths: list[str]) -> str | None:
    """The first path with a Python-internal segment (`__class__`), which the sandbox refuses on any object.

    A single leading underscore is left alone: `_id` or `_source` are ordinary keys of a record from a
    document store, and the sandbox reads them from a dict as it reads any key.
    """
    for path in paths:
        if any(isinstance(segment, str) and segment.startswith("__") for segment in _split_path(path)):
            return path
    return None


def _split_path(path: str) -> list[str | int]:
    parts: list[str | int] = []
    position = 0
    while position < len(path):
        match = _SEGMENT.match(path, position)
        if match is None:
            # An unreadable remainder stays one segment, so a message still names the path as written.
            parts.append(path[position:])
            break
        position = match.end()
        name, index, single, double = match.groups()
        if name is not None:
            parts.append(name)
        elif index is not None:
            parts.append(int(index))
        else:
            parts.append(re.sub(r"\\(.)", r"\1", single if single is not None else double))
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
    message_reads: Reads
    effective_from: date | None
    effective_until: date | None


class RulesInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Rules(Node):
    """Evaluates every rule against the inputs and returns one finding per rule.

    Inputs arrive by name and rules read them by path (`docs.Note.interest_rate`), so a record of any shape
    needs no mapping beyond naming it. Derived values are computed once per record, in order, before the rules
    run, and are read by name like an input. Expressions use the same sandboxed engine as the Expression node,
    plus `has`, `days_between`, `date`, `today`, `len`, `abs`, `min`, `max`, `sum` and `round`. A record member
    named like a helper is the member where a rule reads it as a value and the helper where a rule calls it. A
    member named like a method of the record, `items` or `update`, is the member: the method is reached only
    for a key the record lacks, and a read the sandbox refuses holds that rule as `not_evaluated` rather than
    failing the run.

    Every enabled rule reports a status: `pass` when its check holds; its severity (`fail`, `warn`, `info`)
    when the check does not; `not_applicable` when `applies_when` does not hold or the record's `as_of` date
    falls outside the rule's effective window; `not_evaluated` when a value the check reads is missing or the
    check cannot be evaluated, unless `on_missing` says to report the rule's severity instead. A missing value
    never passes or fails a rule silently. The message is rendered with the whole record, and the finding
    carries the values the check read under `evaluated`. Rules compile when the node is built, so a malformed
    expression fails then, naming the rule.

    The output holds `findings` in rule order, a `summary` of statuses, `status` and the `derived` values.
    The status is `fail` if any rule failed, else `warn` if any warned, else `not_evaluated` if any check
    did not run, else `pass`; a check that read a missing value did not run under either policy, so a record
    is never `pass` while a value was missing, whatever its finding reports. An optional `as_of` input, an
    ISO date, fixes the date the effective windows are compared with; without it the run date is used.
    """

    name: str | None = "rules"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    input_fields: list[NamedField] = []
    derived_values: list[DerivedValue] = []
    rules: list[Rule] = []
    on_missing: RuleMissingPolicy = RuleMissingPolicy.NOT_EVALUATED
    input_schema: ClassVar[type[RulesInputSchema]] = RulesInputSchema

    _compiled: list[CompiledRule] = PrivateAttr(default_factory=list)
    _derived: list[tuple[str, Callable[..., Any], Reads]] = PrivateAttr(default_factory=list)

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

    def _compile_expression(self, text: str, where: str) -> tuple[Callable[..., Any], Reads]:
        try:
            reads = read_paths(text)
            # A lookup that finds nothing must come back as RuleUndefined, whose truth test raises, rather
            # than be turned into None on the way out: a bare `limits[program]` is then not evaluated
            # instead of read as false and reported as a verdict.
            compiled = _ENVIRONMENT.compile_expression(text, undefined_to_none=False)
        except TemplateSyntaxError as e:
            raise ValueError(f"{where} is not a valid expression: {e}") from e
        # The sandbox refuses these at run time; refusing them at build time names the rule instead of holding it.
        if private := _private_segment(reads.required + reads.optional):
            raise ValueError(f"{where} reads a private attribute ({private})")
        refuse_reserved_read(reads, where)
        # One name cannot be both: the record's member would shadow the helper, or the helper stand in for the member.
        if clash := next((name for name in reads.helpers_read if name in reads.helpers_called), None):
            raise ValueError(f"{where} reads {clash!r} as a value and calls it as a helper")
        return compiled, reads

    def _compile_derived(self) -> list[tuple[str, Callable[..., Any], Reads]]:
        taken = {field.name for field in self.input_fields}
        compiled = []
        for value in self.derived_values:
            label = f"Rules '{self.name}': derived value {value.name!r}"
            if not value.name.isidentifier():
                raise ValueError(f"{label} is not a valid identifier")
            if value.name in taken or value.name in HELPERS:
                raise ValueError(f"{label} is already the name of an input or a helper")
            if value.name == RESERVED_ROOT:
                raise ValueError(f"{label} could not be read by a rule: Jinja reserves the name inside an expression")
            if not value.expression.strip():
                raise ValueError(f"{label} has no expression")
            taken.add(value.name)
            compiled.append((value.name, *self._compile_expression(value.expression, label)))
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
            message, message_reads = None, Reads(required=[], optional=[])
            if rule.message and rule.message.strip():
                try:
                    message = _ENVIRONMENT.from_string(rule.message)
                    message_reads = read_template(rule.message)
                except TemplateSyntaxError as e:
                    raise ValueError(f"{label}: the message is not a valid template: {e}") from e
                refuse_reserved_read(message_reads, f"{label}: the message")
            applies_compiled, applies_reads = None, Reads(required=[], optional=[])
            if applies:
                applies_compiled, applies_reads = self._compile_expression(applies, f"{label}: applies_when")
            check, check_reads = self._compile_expression(rule.check, f"{label}: the check")
            compiled.append(
                CompiledRule(
                    rule=rule,
                    applies=applies_compiled,
                    applies_reads=applies_reads,
                    check=check,
                    check_reads=check_reads,
                    message=message,
                    message_reads=message_reads,
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
        for name, expression, reads in self._derived:
            # One mapping, derived winning, passed positionally: an undeclared key the upstream payload carries
            # under a derived value's name would otherwise clash as a duplicate keyword argument, and a key
            # named `self` would collide with the compiled expression's own bound argument. Such a key is never
            # read: Jinja binds the name inside the expression, so a read of it is refused at build.
            try:
                # A value the expression could not find is missing, inside a list or a dict it built as well,
                # and the output stays serializable.
                derived[name] = concrete(expression(scope_for(reads, {**context, **derived}, RuleUndefined)))
            except EVALUATION_ERRORS:
                derived[name] = None
        scope = {**context, **derived}

        findings: list[dict[str, Any]] = []
        screened = True
        for compiled in self._compiled:
            finding, evaluated = self._evaluate(compiled, scope, as_of)
            findings.append(finding)
            screened = screened and evaluated
        summary = {status: 0 for status in STATUSES}
        for finding in findings:
            summary[finding["status"]] += 1
        return {
            "status": self._overall(summary, screened),
            "summary": summary,
            "findings": findings,
            "derived": derived,
        }

    @staticmethod
    def _overall(summary: dict[str, int], screened: bool) -> str:
        # A record passes only when every rule that applied was evaluated and held: a check that could not run
        # is not a pass, or a caller routing on `status == "pass"` would clear a file whose screening never ran.
        # Under the strict policy such a check reports the rule's severity, which for an info rule counts for
        # nothing here, so the record reads not evaluated rather than pass.
        for status in (STATUS_FAIL, STATUS_WARN):
            if summary[status]:
                return status
        if summary[STATUS_NOT_EVALUATED] or not screened:
            return STATUS_NOT_EVALUATED
        return STATUS_PASSED

    @staticmethod
    def _as_of(value: Any) -> date:
        if _is_missing(value) or value == "":
            return date.today()
        try:
            return to_date(value)
        except ValueError as e:
            raise ValueError(f"Rules: '{AS_OF_KEY}' is not a date: {value!r}") from e

    def _evaluate(self, compiled: CompiledRule, scope: dict[str, Any], as_of: date) -> tuple[dict[str, Any], bool]:
        """The finding for one rule, and whether its check ran: a missing value or an error means it did not."""
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
            finding["message"] = f"not in force on {as_of.isoformat()}: effective {self._window(rule)}"
            return finding, True

        status, reason, evaluated = self._status(compiled, scope)
        finding["status"] = status
        if status != STATUS_NOT_APPLICABLE:
            reads = compiled.check_reads.required + compiled.check_reads.optional
            finding["evaluated"] = {path: _shown(resolve_path(scope, path)) for path in reads}
        # A finding that did not fire still says why: a reviewer reading the list should never have
        # to re-run the record to learn which condition or date kept a rule out.
        if status in (STATUS_NOT_EVALUATED, STATUS_NOT_APPLICABLE):
            finding["message"] = reason
        elif status != STATUS_PASSED:
            finding["message"] = self._render(compiled, scope, reason)
        return finding, evaluated

    @staticmethod
    def _window(rule: Rule) -> str:
        if rule.effective_from and rule.effective_until:
            return f"from {rule.effective_from} to {rule.effective_until}"
        return f"from {rule.effective_from}" if rule.effective_from else f"until {rule.effective_until}"

    def _status(self, compiled: CompiledRule, scope: dict[str, Any]) -> tuple[str, str | None, bool]:
        """The rule's status, the reason when it did not run or did not apply, and whether its check ran."""
        if compiled.applies is not None:
            if missing := self._missing(compiled.applies_reads.required, scope):
                return self._missing_status(compiled, f"missing value for {missing}")
            try:
                applies = holds(compiled.applies(scope_for(compiled.applies_reads, scope, RuleUndefined)))
            except EVALUATION_ERRORS as e:
                return self._missing_status(compiled, f"applies_when could not be evaluated: {e}")
            if not applies:
                return STATUS_NOT_APPLICABLE, f"does not apply: {compiled.rule.applies_when.strip()}", True

        if missing := self._missing(compiled.check_reads.required, scope):
            return self._missing_status(compiled, f"missing value for {missing}")
        try:
            held = holds(compiled.check(scope_for(compiled.check_reads, scope, RuleUndefined)))
        except EVALUATION_ERRORS as e:
            return self._missing_status(compiled, f"check could not be evaluated: {e}")
        return (STATUS_PASSED, None, True) if held else (compiled.rule.severity.value, None, True)

    def _missing_status(self, compiled: CompiledRule, reason: str) -> tuple[str, str, bool]:
        if self.on_missing == RuleMissingPolicy.FAIL:
            return compiled.rule.severity.value, reason, False
        return STATUS_NOT_EVALUATED, reason, False

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
            rendered = compiled.message.render(scope_for(compiled.message_reads, scope, RuleUndefined)).strip()
        except EVALUATION_ERRORS:
            rendered = compiled.rule.message.strip() if compiled.rule.message else ""
        if reason:
            return f"{rendered} ({reason})" if rendered else reason
        return rendered or None
