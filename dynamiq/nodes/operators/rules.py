import functools
import math
import numbers
import re
from collections.abc import Callable, ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar, Literal, NamedTuple, NoReturn
from uuid import uuid4

from jinja2 import ChainableUndefined, Template, TemplateSyntaxError, Undefined, nodes, pass_environment
from jinja2.exceptions import TemplateRuntimeError, UndefinedError
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
# Jinja's own runtime errors count as well: an undefined value used, a filter given the wrong argument, an
# attribute the sandbox refuses, `append` on a list member say, whose use raises a SecurityError, and a value
# `number()` or `date()` could not read.
EVALUATION_ERRORS = (TemplateRuntimeError, TypeError, ValueError, ArithmeticError, AttributeError, LookupError)
_US_DATE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_YEAR_FIRST_DATE = re.compile(r"^([0-9]{4})/([0-9]{1,2})/([0-9]{1,2})$")
_NAMED_DATE = re.compile(r"^(?P<month>[A-Za-z]+)\.?\s+(?P<day>[0-9]{1,2})(?:,\s*|\s+)(?P<year>[0-9]{4})$")
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
# English month names, in full or cut to three letters, whatever the process's locale: `%B` in `strptime` follows the
# locale, and a rule must read the same date on every machine.
_MONTHS = {name: index for index, full in enumerate(_MONTH_NAMES, start=1) for name in (full, full[:3])} | {"sept": 9}
# The number `number()` accepts once the text is cleaned: ASCII digits, a sign and a decimal point with digits after
# it. Python's own readers take more, `1e5`, `1_000`, `nan` and digits of any script, which no document means.
_PLAIN_NUMBER = re.compile(r"[+-]?[0-9]+(?:\.[0-9]+)?")
# Thousands grouped by the separator that is not the decimal one: `1,234,567` or, with a decimal comma, `1.234.567`.
_GROUPED_THOUSANDS = {
    ",": re.compile(r"[1-9][0-9]{0,2}(?:,[0-9]{3})+"),
    ".": re.compile(r"[1-9][0-9]{0,2}(?:\.[0-9]{3})+"),
}
_CURRENCY = re.compile(r"[$€£]")
# Whitespace beside a comma or a point: `100, 200` may be two amounts as much as one.
_SPACE_BESIDE_SEPARATOR = re.compile(r"\s[.,]|[.,]\s")
# Whitespace between two digits, which groups thousands the way a grouping separator does: `1 234`.
_SPACE_BETWEEN_DIGITS = re.compile(r"(?<=[0-9])\s+(?=[0-9])")
_SPACE = re.compile(r"\s+")
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# One path segment: a name, a numeric index, or a quoted key with escapes.
_SEGMENT = re.compile(r"""\.?([^.\[\]]+)|\[(?:(-?\d+)|'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\]""")
_EXEMPT_TESTS = frozenset({"defined", "undefined", "none", "sameas", "present", "blank"})

_MISSING = object()


class MissingValue(UndefinedError):
    """A value a rule needs is missing or blank, so the rule could not decide.

    Jinja raises an undefined value's error with the message alone, so the path is optional; a raiser that knows
    which read found nothing builds the error with `for_path`, which names the path in the message and on `path`.
    """

    def __init__(self, message: str | None = None, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    @classmethod
    def for_path(cls, path: str) -> "MissingValue":
        return cls(f"missing value for {path}", path=path)


class Blank:
    """Marks the undefined value a helper returns for blank input: `text('  ')`, a `first_present` that finds nothing.

    Each sandbox derives its blank from its own undefined (`RecordSandbox.blank`), so a blank is as missing as
    anything undefined there: in a rule it raises on use, in an expression it comes out as None. The marker is how a
    text filter tells a blank, which it passes on, from any other undefined, which it keeps reading as ''.
    """

    __slots__ = ()

    def _raise_missing(self, *args: Any, **kwargs: Any) -> NoReturn:
        self._fail_with_undefined_error()  # type: ignore[attr-defined]

    # The numeric hooks Jinja's undefined lacks, where Python would raise a type error instead: a blank `number()`
    # rounded, `round(n, 2)` or `| abs`, is as missing as one added to.
    __round__ = __abs__ = __trunc__ = __floor__ = __ceil__ = __index__ = _raise_missing


class UnreadableValue(TemplateRuntimeError):
    """A value is there but cannot be read as what the expression asks for: `number('TBD')`, `date('March')`.

    A runtime error rather than a ValueError: Jinja's `| float` and `| int` take a ValueError or a TypeError for "no
    number" and return 0, the silent answer this error exists to prevent. The value that could not be read comes with
    the error, so whoever catches it can keep what the record said.
    """

    def __init__(self, message: str | None = None, *, value: Any = None) -> None:
        super().__init__(message)
        self.value = value


class Unreadable:
    """A value `number()` or `date()` could not read, with the reason: `TBD` where an amount goes.

    It is there, so it is present rather than missing, and `first_present` stops at it instead of letting a fallback
    speak over it. Every use raises `UnreadableValue` with the reason: a comparison, arithmetic, a truth test, a count,
    a hash, a member or an item, the conversions behind `| float` and `| int`, which would otherwise read it as 0, and
    the conversion to text behind `| string`, the text filters, `~` and `join`, which would hand a check back the
    text the reader refused. Only a rule's message prints it, as the text the record holds (`_rendered`).
    """

    __slots__ = ("value", "reason")

    def __init__(self, value: Any, reason: str) -> None:
        self.value = value
        self.reason = reason

    def error(self) -> UnreadableValue:
        """The error every use of the value raises."""
        return UnreadableValue(self.reason, value=self.value)

    def _refuse(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise self.error()

    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __hash__ = _refuse
    __bool__ = __len__ = __iter__ = __contains__ = __getitem__ = _refuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __pow__ = __rpow__ = _refuse
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = _refuse
    __divmod__ = __rdivmod__ = __neg__ = __pos__ = __abs__ = _refuse
    __int__ = __float__ = __complex__ = __index__ = __round__ = __trunc__ = __floor__ = __ceil__ = _refuse
    __str__ = __format__ = _refuse

    def __repr__(self) -> str:
        return f"Unreadable({self.value!r}, {self.reason!r})"


def _is_missing(value: Any) -> bool:
    return value is None or value is _MISSING or isinstance(value, Undefined)


def has(value: Any) -> bool:
    """True when a value is present: defined and not null."""
    return not _is_missing(value)


def is_blank(value: Any) -> bool:
    """`x is blank`: missing or null, or holding nothing, as text of spaces only or an empty list or mapping does.

    A form or an extraction says "no answer" in each of these ways, and a rule should read them alike; `0` and
    `false` are answers, so they are present. A path that ends at a method, `invoice.items` over a record without an
    `items` key say, found the record's method rather than a value, so it is blank too. An undefined is never
    compared, counted or tested for truth here, since a rule's undefined raises on each.
    """
    if _is_missing(value) or callable(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, Mapping)):
        return not value
    return False


def is_present(value: Any) -> bool:
    """`x is present`: anything that is not blank."""
    return not is_blank(value)


def to_date(value: Any, format: str | None = None) -> date:
    """Reads a date from a date, a datetime or text written the way documents write one.

    The text may be ISO (`2026-08-07`, a time after it allowed), US month first (`08/07/2026`), year first with
    slashes (`2026/08/07`) or an English month name (`Aug 7, 2026`, `August 7 2026`), read in English whatever the
    process's locale. Given a `format`, the text is read as `datetime.strptime` reads that format, and nothing else.

    Raises when there is no date to read: a value already missing raises its own undefined error and a value already
    unreadable its own error, so `days_between(date(a), b)` reports what `date()` found; anything else raises a
    ValueError naming the value.
    """
    if isinstance(value, Undefined):
        value._fail_with_undefined_error()
    if isinstance(value, Unreadable):
        raise value.error()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        written = value.strip()
        if format is not None:
            try:
                return datetime.strptime(written, format).date()
            except ValueError as e:
                raise ValueError(f"not a date: {value!r} (format {format!r})") from e
        if _ISO_DATE.match(written):
            return date.fromisoformat(written[:10])
        if us := _US_DATE.match(written):
            month, day, year = (int(part) for part in us.groups())
            return date(year, month, day)
        if year_first := _YEAR_FIRST_DATE.match(written):
            year, month, day = (int(part) for part in year_first.groups())
            return date(year, month, day)
        if (named := _NAMED_DATE.match(written)) and (month := _MONTHS.get(named["month"].lower())):
            return date(int(named["year"]), month, int(named["day"]))
    raise ValueError(f"not a date: {value!r}")


def days_between(start: Any, end: Any) -> int:
    """The number of days from `start` to `end`, negative when `end` comes first."""
    return (to_date(end) - to_date(start)).days


def today() -> date:
    return date.today()


@pass_environment
def text(environment: "RecordSandbox", value: Any) -> Any:
    """The value as text without the spaces around it, or a blank when there is no text.

    `app.purpose | trim == 'purchase'` fails a purpose nobody gave, since blank text trims to '' and a null reads as
    'None'; `text(app.purpose) == 'purchase'` holds the rule as missing instead. A value that is not there at all
    becomes a blank as well, so it stays missing through a text filter, which reads any other undefined as ''. A
    blank passes through as it is, and so does a value `number()` or `date()` could not read: as text it would hand
    a comparison the text the reader refused.
    """
    if isinstance(value, (Blank, Unreadable)):
        return value
    if is_blank(value):
        return environment.blank(hint="missing value: text() found no text", exc=MissingValue)
    return str(value).strip()


def _read_number(written: str, decimal: str) -> int | float | None:
    """The number the text writes, or None when it writes none, or writes one only a guess could read."""
    grouping = "," if decimal == "." else "."
    # A currency sign says nothing about the amount; read as a space, it cannot join the digits on either side of it.
    body = _CURRENCY.sub(" ", written)
    if _SPACE_BESIDE_SEPARATOR.search(body):
        return None
    # Spaces group thousands in some documents, `1 234`, and count only where a grouping separator could stand; a
    # number grouped both ways, `1 234,56`, is more likely written with a decimal comma than grouped twice.
    if _SPACE_BETWEEN_DIGITS.search(body):
        if grouping in body:
            return None
        body = _SPACE_BETWEEN_DIGITS.sub(grouping, body)
    # Any other space, at either end or beside a sign, a parenthesis or `%`, says nothing about the amount either.
    body = _SPACE.sub("", body)
    # An amount in parentheses is negative, as an account writes a debit.
    negative = body.startswith("(") and body.endswith(")")
    if negative:
        body = body[1:-1]
    body = body.removesuffix("%")
    sign = body[:1] if body[:1] in ("+", "-") else ""
    if negative and sign:
        return None
    whole, point, fraction = body[len(sign) :].partition(decimal)
    # `12,5` is 12.5 to one writer and 125 to another, so a grouping separator counts only where it groups thousands.
    if grouping in whole:
        if not _GROUPED_THOUSANDS[grouping].fullmatch(whole):
            return None
        whole = whole.replace(grouping, "")
    plain = f"{sign}{whole}.{fraction}" if point else f"{sign}{whole}"
    if not _PLAIN_NUMBER.fullmatch(plain):
        return None
    try:
        read: int | float = float(plain) if point else int(plain)
    except ValueError:  # more digits than Python converts to an int
        return None
    # More digits than a float holds read as infinity, which is no amount.
    if isinstance(read, float) and not math.isfinite(read):
        return None
    return -read if negative else read


def _number_of(value: Any, decimal: str) -> int | float | None:
    """The number a present value holds, or None when it holds none that can be read without guessing."""
    if isinstance(value, bool):
        # True is an answer, not an amount, though Python counts it as 1.
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        read = float(value)
        return read if math.isfinite(read) else None
    if isinstance(value, Decimal):
        # Written out in full, a Decimal reads like any amount: an int without a point, a float with one.
        return _read_number(format(value, "f"), ".")
    if isinstance(value, str):
        return _read_number(value, decimal)
    return None


@pass_environment
def number(environment: "RecordSandbox", value: Any, decimal: str = ".") -> Any:
    """The value as a number: an int when it has no decimal point, a float when it has one.

    Text is read the way a document writes an amount. Currency signs (`$`, `€`, `£`) and the spaces around the
    number say nothing about it; parentheses make it negative, `(1,200.50)`; a `%` after it is dropped, so `6.25%` is
    6.25; commas group thousands, `1,234,567.89`, and so do spaces, `1 234 567`, but only where they group thousands.
    `decimal=','` reads a decimal comma instead, `1.234,56` or `1 234,56`. Anything else is unreadable rather than
    guessed at: `12,5`, `12 5`, `100, 200`, `1e5`, `nan`, `TBD`, `true`. Where `| float` reads `TBD` as 0, a rule that
    uses an unreadable number is not evaluated, naming the value. A blank value is missing, as in `text()`; a value
    already missing or unreadable passes through as it is.
    """
    if decimal not in (".", ","):
        raise ValueError(f"number() reads a decimal point '.' or a decimal comma ',', not {decimal!r}")
    if isinstance(value, (Blank, Unreadable)):
        return value
    if is_blank(value):
        return environment.blank(hint="missing value: number() found no number", exc=MissingValue)
    read = _number_of(value, decimal)
    return Unreadable(value, f"not a number: {value!r}") if read is None else read


@pass_environment
def read_date(environment: "RecordSandbox", value: Any, format: str | None = None) -> Any:
    """`date(x)` in an expression: the value as a date, read as `to_date` reads it (`date(x, format='%d.%m.%Y')`).

    Text that is no date is unreadable, and a rule that uses it is not evaluated, naming the text. A blank value is
    missing, as in `text()`; a value already missing or unreadable passes through as it is.
    """
    if isinstance(value, (Blank, Unreadable)):
        return value
    if is_blank(value):
        return environment.blank(hint="missing value: date() found no date", exc=MissingValue)
    try:
        return to_date(value, format)
    except ValueError as e:
        return Unreadable(value, str(e))


@pass_environment
def first_present(environment: "RecordSandbox", *values: Any) -> Any:
    """The first of the values that is present, or a blank when none is.

    A fallback written out, `a if a is present else b`, names each value twice and grows with every alternative;
    `first_present(a, b, c)` names each once. A value it reads may be missing without holding the rule, since the
    next one stands in for it: only when none is present is the result missing.
    """
    for value in values:
        if is_present(value):
            return value
    return environment.blank(hint="missing value: first_present() found nothing present", exc=MissingValue)


HELPERS: dict[str, Callable[..., Any]] = {
    "has": has,
    "days_between": days_between,
    "date": read_date,
    "today": today,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "text": text,
    "number": number,
    "first_present": first_present,
}

# The helpers there were before the vocabulary grew. A derived value may not take one of these names, as it never
# could; it may take a name added since, so a workflow that already has a derived value called `text` keeps
# building, and the value shadows the helper the way an input of that name does.
RESERVED_NAMES = frozenset({"has", "days_between", "date", "today", "len", "abs", "min", "max", "sum", "round"})

# The tests that ask about a value; like `is defined`, they may read one that is missing.
TESTS: dict[str, Callable[[Any], bool]] = {"present": is_present, "blank": is_blank}

# Jinja's text filters read an undefined value as '', which a comparison takes for an answer: `date(x) | string`
# over a blank `x` as well.
_TEXT_FILTERS = ("lower", "upper", "trim", "title", "capitalize", "replace", "string")


def _keeps_blank(filter_: Callable[..., Any]) -> Callable[..., Any]:
    """The filter, with a blank passed on untouched.

    `text(x) | lower == 'purchase'` would otherwise fail a blank `x` that `text(x) == 'purchase'` holds as missing.
    Only a blank passes: any other undefined still reads as '', as the messages and expressions written before the
    marker expect. `wraps` copies the filter's attributes, Jinja's pass-argument marker among them, so `replace`
    still receives its eval context first and the value second.
    """
    value_at = 0 if getattr(filter_, "jinja_pass_arg", None) is None else 1

    @functools.wraps(filter_)
    def keep_blank(*args: Any, **kwargs: Any) -> Any:
        return args[value_at] if isinstance(args[value_at], Blank) else filter_(*args, **kwargs)

    return keep_blank


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

    Every sandbox, the Rules node's and the Expression node's alike, carries the helpers, the `present` and `blank`
    tests, and a `blank` of its own: its undefined marked `Blank`, so blank input is as missing as anything
    undefined in that sandbox.
    """

    blank: type[Undefined]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.blank = type("Blank", (Blank, self.undefined), {"__slots__": ()})
        self.globals.update(HELPERS)
        self.tests.update(TESTS)
        for name in _TEXT_FILTERS:
            self.filters[name] = _keeps_blank(self.filters[name])

    def getattr(self, obj: Any, attribute: str) -> Any:
        if isinstance(obj, Mapping):
            try:
                return obj[attribute]
            except (TypeError, LookupError):
                pass
        # An unreadable value has no members: `date(x).year` over text that is no date raises why, as its use does.
        elif isinstance(obj, Unreadable):
            raise obj.error()
        return super().getattr(obj, attribute)


def _method_text(value: Any) -> str:
    """A method as text: a path ending at one, `invoice.items.count`, names the method, never the data."""
    return f"{getattr(value, '__name__', type(value).__name__)}(…)"


def _rendered(value: Any) -> Any:
    """A value as a message prints it.

    A message is the one place a value is turned into text, so a method a template names would otherwise
    print a repr carrying an address that differs on every run, against the determinism a finding promises.
    An undefined is callable too, and keeps rendering as the empty string a message expects. A value `number()`
    or `date()` could not read, which refuses to become text anywhere else, prints as the text the record holds,
    so a reviewer reads what the record says.
    """
    if isinstance(value, Unreadable):
        return value.value
    return _method_text(value) if callable(value) and not isinstance(value, Undefined) else value


# One sandbox for every Rules node; the expressions it compiles are stateless. Only a rendered message
# passes through `finalize`; a compiled expression does not, so a check keeps the value it read.
_ENVIRONMENT = RecordSandbox(undefined=RuleUndefined, finalize=_rendered)


def concrete(value: Any) -> Any:
    """Returns the value with every undefined member or method replaced by None and every lazy iterable
    materialized.

    Jinja turns a result into None only when the whole result is undefined. A list or a dict the expression
    builds keeps the undefined objects inside it, `map(attribute=...)` over items that lack the attribute above
    all, and such an object is not serializable and raises on its first use downstream. `map`, `select`,
    `selectattr`, `reject` and `rejectattr` return generators, which the first rule to read one exhausts for
    every rule after it, and which no encoder can record, so an iterator, a dict view, a range or a set becomes
    a list; a string and an object that merely iterates, a document say, stay what they are. A path that ends
    at a method reads the bound method, which no encoder can record and whose repr carries an address, so it
    is no more a value than an undefined is. A value `number()` or `date()` could not read raises its error at
    any depth instead: None would pass it off as missing, where the record says something nobody could read.
    """
    if isinstance(value, Unreadable):
        raise value.error()
    if isinstance(value, Undefined) or callable(value):
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
    undefined result keeps raising, which is what makes a lookup that found nothing `not_evaluated`. A result
    that is a method is no verdict either: every method is truthy, so a check left at one would clear the
    record on a typo, and it raises into `not_evaluated` the way an undefined does.
    """
    if isinstance(value, Undefined):
        return bool(value)
    if callable(value):
        raise TypeError(f"read the method {_method_text(value)}, not a value")
    return bool(concrete(value))


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
    # The fallbacks `first_present` reads: allowed to be missing, yet asked about by nothing, so they guard nothing.
    lenient: list[str]


def _root(path: str) -> str:
    return path.split(".")[0].split("[")[0]


def _collect_paths(node: nodes.Node, collected: _Collected, required: bool, lenient: bool = False) -> None:
    # A call of a helper reads its arguments, never a member of the helper's name; a value asked about with
    # `has`, `is defined`, `is present` or `default` is allowed to be missing, and so is anything read inside the
    # arguments of `first_present`, which skips a missing value rather than asking about it.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name) and node.node.name in GLOBAL_NAMES:
        collected.called.append(node.node.name)
        fallback = lenient or node.node.name == "first_present"
        for child in node.iter_child_nodes(exclude=("node",)):
            _collect_paths(child, collected, required and node.node.name != "has", fallback)
        return
    if isinstance(node, nodes.Test) and node.name in _EXEMPT_TESTS:
        _collect_paths(node.node, collected, required=False, lenient=lenient)
        return
    if isinstance(node, nodes.Filter) and node.name == "default":
        _collect_paths(node.node, collected, required=False, lenient=lenient)
        for argument in node.args:
            _collect_paths(argument, collected, required, lenient)
        return
    # A method call reads the object it is called on, not a member of the method's name: `invoice.get('vat_rate')`
    # needs `invoice`, and a dict holds no key called `get`.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Getattr):
        _collect_paths(node.node.node, collected, required, lenient)
        for child in node.iter_child_nodes(exclude=("node",)):
            _collect_paths(child, collected, required, lenient)
        return
    path = _path_of(node)
    if path is not None:
        target = collected.lenient if lenient else collected.required if required else collected.optional
        if path not in target:
            target.append(path)
        return
    for child in node.iter_child_nodes():
        _collect_paths(child, collected, required, lenient)


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def _reads_of(parsed: nodes.Template) -> Reads:
    collected = _Collected(required=[], optional=[], called=[], lenient=[])
    _collect_paths(parsed, collected, required=True)
    required: list[str] = []
    optional = list(collected.optional)
    for path in collected.required:
        if any(_is_under(path, guarded) for guarded in collected.optional):
            if path not in optional:
                optional.append(path)
        else:
            required.append(path)
    # A fallback is optional unless the expression also reads it on its own, and it guards nothing:
    # `first_present(a.b, c) == 1 and a.b.c > 0` still needs `a.b.c`.
    optional += [path for path in collected.lenient if path not in required and path not in optional]
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


def refuse_clash(compiled: Callable[..., Any], reads: Reads, where: str) -> tuple[Callable[..., Any], Reads]:
    """The compiled expression and its reads, refusing a name the expression both reads as a value and calls.

    One name cannot be both: the record's member would shadow the helper, or the helper stand in for the member.
    Such an expression is refused when the node is built, naming it, unless the name is one the vocabulary added
    since: an expression over an input called `text`, written before `text` was a helper, must keep building. It
    raises the clash whenever it is evaluated instead, an error rather than a missing value, so none of its reads is
    required ahead of it: a missing one would otherwise report the rule as missing before the clash is reached.
    """
    clash = next((name for name in reads.helpers_read if name in reads.helpers_called), None)
    if clash is None:
        return compiled, reads
    message = f"{where} reads {clash!r} as a value and calls it as a helper"
    if clash in RESERVED_NAMES or clash not in HELPERS:
        raise ValueError(message)

    def clashing(*args: Any, **kwargs: Any) -> Any:
        raise TypeError(message)

    return clashing, reads._replace(required=[], optional=reads.required + reads.optional)


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
    """The value at a dotted path in the context, or the missing marker when any step is absent.

    A path that reaches a value nobody could read ends there: the marker's own members, `value` and `reason`,
    are not the record's, and the path is no more missing than the value is, so a finding shows why the value
    could not be read, and a rule that reads the path is not evaluated for that reason rather than as missing.
    """
    current: Any = context
    for part in _split_path(path):
        if isinstance(current, Unreadable):
            return current
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
    """A value as a finding shows it: scalars as they are, containers as their size, a method as a call, and a
    value nobody could read as the reason why."""
    if isinstance(value, Unreadable):
        return f"unreadable: {value.reason}"
    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return f"{{…{len(value)} keys}}"
    if isinstance(value, (list, tuple)):
        return f"[…{len(value)} items]"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    # `invoice.items.count`, the ordinary slip of expecting a count property, resolves to the list's method:
    # the finding names it rather than carrying a bound method the node output could not be serialized with.
    if callable(value):
        return _method_text(value)
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

    The output holds `findings` in rule order, a `summary` of statuses, `status`, the `derived` values and
    `derived_errors`. A derived value computed from a missing value is missing, None under `derived`; one that
    could not be computed from values that are there, a division by zero or `number()` of `TBD`, is None there
    too, with the reason under `derived_errors`, and a rule that reads it is not evaluated, naming the reason.
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
        return refuse_clash(compiled, reads, where)

    def _compile_derived(self) -> list[tuple[str, Callable[..., Any], Reads]]:
        taken = {field.name for field in self.input_fields}
        compiled = []
        for value in self.derived_values:
            label = f"Rules '{self.name}': derived value {value.name!r}"
            if not value.name.isidentifier():
                raise ValueError(f"{label} is not a valid identifier")
            if value.name in taken or value.name in RESERVED_NAMES:
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
        # The derived values as the rules read them, as the output reports them, and why any could not be computed.
        values: dict[str, Any] = {}
        derived: dict[str, Any] = {}
        derived_errors: dict[str, str] = {}
        for name, expression, reads in self._derived:
            # One mapping, derived winning, passed positionally: an undeclared key the upstream payload carries
            # under a derived value's name would otherwise clash as a duplicate keyword argument, and a key
            # named `self` would collide with the compiled expression's own bound argument. Such a key is never
            # read: Jinja binds the name inside the expression, so a read of it is refused at build.
            known = {**context, **values}
            try:
                # A value the expression could not find is missing, inside a list or a dict it built as well,
                # and the output stays serializable.
                value = concrete(expression(scope_for(reads, known, RuleUndefined)))
            except UndefinedError:
                # So is one it used: arithmetic on a key the record lacks, or on a blank `number()` read.
                value = None
            except EVALUATION_ERRORS as e:
                # A failure while a value the expression needs is missing, `amount / value` over a null `value`,
                # is that value missing too. Any other failure is an error: as None it would read as a value
                # nobody gave, one `first_present` or a guard skips. The rules read it as unreadable instead,
                # keeping the value the error names, if any, for a message to print; the output reports None
                # and the reason.
                if self._missing(reads.required, known):
                    value = None
                else:
                    value = Unreadable(e.value if isinstance(e, UnreadableValue) else None, str(e))
                    derived_errors[name] = value.reason
            values[name] = value
            derived[name] = None if isinstance(value, Unreadable) else value
        scope = {**context, **values}

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
            "derived_errors": derived_errors,
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
            except MissingValue as e:
                return self._missing_status(compiled, self._missing_reason(compiled.applies_reads, scope, e))
            except EVALUATION_ERRORS as e:
                return self._missing_status(compiled, f"applies_when could not be evaluated: {e}")
            if not applies:
                return STATUS_NOT_APPLICABLE, f"does not apply: {compiled.rule.applies_when.strip()}", True

        if missing := self._missing(compiled.check_reads.required, scope):
            return self._missing_status(compiled, f"missing value for {missing}")
        try:
            held = holds(compiled.check(scope_for(compiled.check_reads, scope, RuleUndefined)))
        except MissingValue as e:
            return self._missing_status(compiled, self._missing_reason(compiled.check_reads, scope, e))
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
    def _missing_reason(reads: Reads, scope: dict[str, Any], error: MissingValue) -> str:
        """Why an expression that used a blank could not decide: the first value it reads that is missing or blank.

        A helper that returns a blank is handed a value, not the path the value came from, so the reads name it;
        a blank no read accounts for, one made from a literal say, keeps the error's own message.
        """
        for path in reads.required + reads.optional:
            if is_blank(resolve_path(scope, path)):
                return f"missing value for {path}"
        return str(error)

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
