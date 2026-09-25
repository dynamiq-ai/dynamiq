import functools
import math
import numbers
import re
from collections.abc import Callable, Container, ItemsView, Iterator, KeysView, Mapping, ValuesView
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar, Literal, NamedTuple, NoReturn
from uuid import uuid4

import jinja2
from jinja2 import ChainableUndefined, Template, TemplateSyntaxError, Undefined, nodes, pass_context, pass_environment
from jinja2.compiler import CodeGenerator, Frame, optimizeconst
from jinja2.environment import TemplateExpression
from jinja2.exceptions import TemplateRuntimeError, UndefinedError
from jinja2.parser import Parser
from jinja2.runtime import Context
from jinja2.sandbox import ImmutableSandboxedEnvironment
from pydantic import BaseModel, ConfigDict, PrivateAttr

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import DerivedValue, NamedField, Rule, RuleMissingPolicy
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import TRUNCATE_LIST_LIMIT, UntruncatedList

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
# The day first, before a month written as a word, which says which number is the day: `7 Aug 2026`, `07-AUG-2026`.
_DAY_FIRST_DATE = re.compile(
    r"^(?P<day>[0-9]{1,2})(?i:st|nd|rd|th)?(?:\s+|-)(?P<month>[A-Za-z]+)\.?(?:,?(?:\s+|-)|,)(?P<year>[0-9]{4})$"
)
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
# The currency signs a document writes beside an amount, and only signs: `¢`, a hundredth of the unit, would read `¢50`
# as 50, and a code, `USD 1,000`, leaves the amount unreadable.
_CURRENCY = re.compile(r"[$€£¥₹₩₽₺₪₫₱₦₴]")
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

    Jinja raises an undefined value's error with the message alone, so the path is optional; `need` and `blank_at`,
    which know which read found nothing, build the error with `for_path`, which names the path in the message and on
    `path`.
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

    It is there, so it is not missing, and `first_present` stops at it instead of letting a fallback speak over it.
    The uses a check makes of a value raise `UnreadableValue` with the reason: a comparison, arithmetic, a truth
    test, a count, a hash, a member or an item, the conversions behind `| float` and `| int`, which would otherwise
    read it as 0, and the conversion to text behind `| string`, the text filters, `~` and `join`, which would hand a
    check back the text the reader refused. So does a question about it, `has()` or any test, `is present` and `is
    none` among them, which would otherwise answer on a value nobody read. A rule's message, which decides nothing,
    reads a derived value or an input that is itself such a value as None, as `derived` shows it (`Rules.execute`),
    and prints one it reaches through `number()`, `date()` or a member as the text the record holds (`_rendered`).
    Only its repr is not refused: `| pprint` and a `'%r'` format print `Unreadable('TBD', ...)`, the marker rather
    than a value; and a list that holds one can still be counted, which uses the list, not it.
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


# The most of a value a reason quotes. Every finding that reads a value nobody could read carries the reason, so a page
# of text in an amount field would otherwise be copied into each of them.
_QUOTED_LENGTH = 80


def _quoted(value: Any) -> str:
    """The value as a reason quotes it: its repr, cut to about 80 characters with `…` where it is longer."""
    text = repr(value)
    return text if len(text) <= _QUOTED_LENGTH else f"{text[:_QUOTED_LENGTH - 1]}…"


def has(value: Any) -> bool:
    """True when a value is present: defined and not null.

    A value nobody could read raises its error instead of answering: it is there, yet a verdict on it, `has(ltv)`
    over a ratio divided by zero say, would be a verdict on nothing anyone read.
    """
    if isinstance(value, Unreadable):
        raise value.error()
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
    """`x is present`: anything that is not blank. `first_present` asks it too, and stops at a value nobody could
    read, which is not blank; the test itself refuses such a value, as every test in a sandbox does."""
    return not is_blank(value)


def to_date(value: Any, format: str | None = None) -> date:
    """Reads a date from a date, a datetime or text written the way documents write one.

    The text may be ISO (`2026-08-07`, a time after it allowed), US month first (`08/07/2026`), year first with
    slashes (`2026/08/07`) or an English month name, month first or day first: `Aug 7, 2026`, `7 Aug 2026`, read in
    English whatever the process's locale. Given a `format`, the text is read as `datetime.strptime` reads that
    format, and nothing else.

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
                raise ValueError(f"not a date: {_quoted(value)} (format {format!r})") from e
        try:
            if _ISO_DATE.match(written):
                return date.fromisoformat(written[:10])
            if us := _US_DATE.match(written):
                month, day, year = (int(part) for part in us.groups())
                return date(year, month, day)
            if year_first := _YEAR_FIRST_DATE.match(written):
                year, month, day = (int(part) for part in year_first.groups())
                return date(year, month, day)
            named = _NAMED_DATE.match(written) or _DAY_FIRST_DATE.match(written)
            if named and (month := _MONTHS.get(named["month"].lower())):
                return date(int(named["year"]), month, int(named["day"]))
        except ValueError as e:
            # Text in a date's shape that names no day of the calendar, `02/29/2025`: Python says which part is out of
            # range, and the reason names the text as well.
            raise ValueError(f"not a date: {_quoted(value)} ({e})") from e
    raise ValueError(f"not a date: {_quoted(value)}")


def days_between(start: Any, end: Any) -> int:
    """The number of days from `start` to `end`, negative when `end` comes first.

    A value `date()` could not read raises its error, `start`'s before `end`'s, as `date()` raised them while it read
    the arguments left to right; raw text is then read `end` first, as it always was, so `days_between(a, b)` over
    two texts that are no dates names `b`. Blank text is missing where it is read, as what `date()` makes of it is:
    `days_between(a, b)` over blank text in `b` is `days_between(a, date(b))`.
    """
    for value in (start, end):
        if isinstance(value, Unreadable):
            raise value.error()
    return (_day(end) - _day(start)).days


def _day(value: Any) -> date:
    """A date `days_between` is handed, read as `to_date` reads it; blank text is missing, as `date()` makes it."""
    if isinstance(value, str) and not value.strip():
        raise MissingValue("missing value: days_between() found no date")
    return to_date(value)


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
    # The minus sign a PDF prints, `−5`, is the hyphen a keyboard writes.
    body = _CURRENCY.sub(" ", written.replace("\u2212", "-"))
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

    Text is read the way a document writes an amount. Currency signs (`$`, `€`, `£`, `¥`, `₹` and the like) and the
    spaces around the number say nothing about it; parentheses make it negative, `(1,200.50)`, as a minus sign does,
    `-5` or the `−5` a PDF prints; a `%` after it is dropped, so `6.25%` is 6.25; commas group thousands,
    `1,234,567.89`, and so do spaces, `1 234 567`, but only where they group thousands. `decimal=','` reads a decimal
    comma instead, `1.234,56` or `1 234,56`. Anything else is unreadable rather than guessed at: `12,5`, `12 5`,
    `100, 200`, `1,23,456`, `USD 1,000`, `1e5`, `nan`, `TBD`, `true`. Where `| float` reads `TBD` as 0, a rule that
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
    return Unreadable(value, f"not a number: {_quoted(value)}") if read is None else read


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
    next one stands in for it: only when none is present is the result missing. A value nobody could read counts
    as present too, so `first_present` stops there rather than falling back past it.
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


def _value_at(function: Callable[..., Any]) -> int:
    """Where a filter or a test receives the value it is applied to: second for one that asks Jinja for its
    environment or context first, as `replace` asks for its eval context, else first."""
    return 0 if getattr(function, "jinja_pass_arg", None) is None else 1


def _keeps_blank(filter_: Callable[..., Any]) -> Callable[..., Any]:
    """The filter, with a blank passed on untouched.

    `text(x) | lower == 'purchase'` would otherwise fail a blank `x` that `text(x) == 'purchase'` holds as missing.
    Only a blank passes: any other undefined still reads as '', as the messages and expressions written before the
    marker expect. `wraps` copies the filter's attributes, Jinja's pass-argument marker among them, so `replace`
    still receives its eval context first and the value second.
    """
    value_at = _value_at(filter_)

    @functools.wraps(filter_)
    def keep_blank(*args: Any, **kwargs: Any) -> Any:
        return args[value_at] if isinstance(args[value_at], Blank) else filter_(*args, **kwargs)

    return keep_blank


def _refuses_unreadable(test: Callable[..., Any]) -> Callable[..., Any]:
    """The test, raising the error of a value nobody could read instead of answering about it.

    `date(doc.closing) is defined` over `March` would otherwise pass on a date nobody could read, and `ltv is none`
    over a ratio divided by zero pass or fail on a figure nobody computed; `select('defined')` and `reject('none')`
    call the same tests. `wraps` keeps the pass-argument marker, as for the filters.
    """
    value_at = _value_at(test)

    @functools.wraps(test)
    def refuse_unreadable(*args: Any, **kwargs: Any) -> Any:
        if len(args) > value_at and isinstance(args[value_at], Unreadable):
            raise args[value_at].error()
        return test(*args, **kwargs)

    return refuse_unreadable


class QuotedUndefined(Undefined):
    """Jinja's undefined, naming what it was looked up by as a reason quotes a value (`_quoted`): a lookup by a page of
    text, `limits[doc.note]`, would otherwise copy the page into every reason that names it. A shorter name prints as
    Jinja prints it."""

    __slots__ = ()

    @property
    def _undefined_message(self) -> str:
        return super()._undefined_message.replace(repr(self._undefined_name), _quoted(self._undefined_name))


class RuleUndefined(QuotedUndefined, ChainableUndefined):
    """A value that is not there.

    Attribute and item access chain, so `has(docs.FloodCert.pages)` can ask about a document that is missing;
    `is present` and `is blank` ask about it the same way. A comparison, a truth test, a count or a loop over
    the value raises instead, so a lookup such as `limits[loan.program]` for a program the table lacks makes
    the rule not evaluated rather than quietly true or false. Rendering it in a message gives an empty string.
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
    undefined in that sandbox. Its undefined is a `QuotedUndefined` unless it is given one. Every test in it,
    Jinja's own as well, refuses a value nobody could read, as `has()` does: a rule that asks about one is held for
    the value's error, and an expression fails its run, as using the value would.
    """

    blank: type[Undefined]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **{"undefined": QuotedUndefined, **kwargs})
        self.blank = type("Blank", (Blank, self.undefined), {"__slots__": ()})
        self.globals.update(HELPERS)
        self.tests.update(TESTS)
        self.tests.update({name: _refuses_unreadable(test) for name, test in self.tests.items()})
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


def _decide(decider: bool, left: Callable[[], Any], right: Callable[[], Any]) -> Any:
    """`left or right` where `decider` is true, `left and right` where it is false, for an `and` or an `or` whose truth
    alone a check uses (`_mark_deciding`), each side a function: either side decides where the other stops at a missing
    value.

    Where the left side is there, this is Python's `or` or `and`, judging it by Python's truth. Where it stops at a
    missing value, a right side whose truth is `decider` decides and is returned; it is judged as `holds` judges a
    check, a lazy one, what `select` yields say, by its items, and comes out as the list of them. Any other right side
    cannot decide, and the left side's missing value stands, as where the right side stops at one too: of two, the
    reason names the left one. Only a missing value gives way, a blank's included: a value nobody could read, a lookup
    that found nothing or a name nothing defines is an error on whichever side the evaluation reaches it.
    """
    try:
        value = left()
        true = bool(value)
    except MissingValue as missing:
        try:
            other = right()
            if isinstance(other, Iterator):
                other = list(other)
            decides = bool(other) == decider
        except MissingValue:
            raise missing from None
        if decides:
            return other
        raise missing from None
    return value if true == decider else right()


# The helpers a check's `or` and `and` call (`RuleCodeGenerator`). A chain of `or`s nests a call and a side's function
# for each `or`, so a partial, which adds no frame, and a truth judged inline keep a long chain as far from the
# recursion limit as it can be.
rule_or = functools.partial(_decide, True)
rule_and = functools.partial(_decide, False)


def _deciding(operator: str, helper: str) -> Callable[[CodeGenerator, nodes.BinExpr, Frame], None]:
    """The visitor that compiles an `and` or an `or` whose truth alone the check uses (`_mark_deciding`) to a call of
    the environment's `helper`, each side a function, and any other one to Python's `operator`, as Jinja writes it.

    It is built the way Jinja builds its own operator visitors (`_make_binop`), so compiling a long chain recurses no
    deeper than Jinja's own would, and the code it writes nests one call where Jinja nests one parenthesis. `need()`
    nests a read the check needs one call deeper than Jinja would, though, so a chain whose first term reads one nests a
    level deeper than Jinja's own: the longest chain of `app.aN == N` Python compiles is 196 terms, where Jinja alone
    compiles 197, and a longer one fails the build with Python's `SyntaxError: too many nested parentheses`, as one past
    Jinja's own limit always did. An `and` or an `or` Jinja folds to a constant when the check is built, `false and …`
    say, is still folded, and the side after the constant is never looked at (`optimizeconst`).
    """

    @optimizeconst
    def visitor(self: CodeGenerator, node: nodes.BinExpr, frame: Frame) -> None:
        if getattr(node, "deciding", False):
            self.write(f"environment.{helper}(lambda: ")
            self.visit(node.left, frame)
            self.write(", lambda: ")
            self.visit(node.right, frame)
        else:
            # Python's own `and` or `or`, which stops at a missing value on any side it reads, written as Jinja writes
            # an operator this sandbox does not intercept (`intercepted_binops`).
            self.write("(")
            self.visit(node.left, frame)
            self.write(f" {operator} ")
            self.visit(node.right, frame)
        self.write(")")

    return visitor


class RuleCodeGenerator(CodeGenerator):
    """Jinja's code generator, with each `and` and `or` whose truth alone a check uses (`_mark_deciding`) compiled to a
    call of the environment's `rule_and` or `rule_or` that hands it each side as a function, so the helper evaluates
    the sides and either can decide when the other stops at a missing value; any other `and` or `or` is Python's, as
    Jinja compiles it. A side reads the names its expression's frame resolved, as it would inline, and runs before the
    helper returns. The sandbox is synchronous, as it must be: Jinja writes `await` only into an asynchronous
    environment's code, and a lambda cannot hold one.

    It also calls this module's own `need()`, `callee()` and `blank_at()` directly, `need()` with the context it asks
    for, where Jinja would call them through the sandbox, which has nothing to check in them: a check calls one for
    every read it needs, and an `and` or an `or` that gives way reads the other side too. Anything else a check calls
    still goes through the sandbox.
    """

    visit_And = _deciding("and", "rule_and")
    visit_Or = _deciding("or", "rule_or")

    def visit_Call(self, node: nodes.Call, frame: Frame, forward_caller: bool = False) -> None:
        helper = node.node.importname if isinstance(node.node, nodes.ImportedName) else None
        if helper not in (_NEED, _CALLEE, _BLANK_AT):
            super().visit_Call(node, frame, forward_caller=forward_caller)
            return
        self.visit(node.node, frame)
        self.write("(context, " if helper == _NEED else "(")
        for index, argument in enumerate(node.args):
            if index:
                self.write(", ")
            self.visit(argument, frame)
        self.write(")")


class RuleSandbox(RecordSandbox):
    """The sandbox a check and an `applies_when` compile in (`_compile_lazy`): a `RecordSandbox` in which either side of
    an `and` or an `or` whose truth alone the expression uses, as `_mark_deciding` marks it, decides when the other
    stops at a missing value (`rule_or`, `rule_and`), and any other `and` or `or` is Python's. A derived value, a
    message and the Expression node compile in a plain `RecordSandbox`, where every `and` and `or` is Python's."""

    code_generator_class = RuleCodeGenerator
    rule_and = staticmethod(rule_and)
    rule_or = staticmethod(rule_or)


def _method_text(value: Any) -> str:
    """A method as text: a path ending at one, `invoice.items.count`, names the method, never the data."""
    return f"{getattr(value, '__name__', type(value).__name__)}(…)"


def _rendered(value: Any) -> Any:
    """A value as a message prints it.

    A message is the one place a value is turned into text, so a method a template names would otherwise
    print a repr carrying an address that differs on every run, against the determinism a finding promises.
    An undefined is callable too, and keeps rendering as the empty string a message expects. A value `number()`
    or `date()` could not read, which refuses to become text anywhere else, prints as the text the record holds,
    so a reviewer reads what the record says: one the message reads through them itself, or one inside an input,
    since a derived value nobody could read reaches a message as None already.
    """
    if isinstance(value, Unreadable):
        return value.value
    return _method_text(value) if callable(value) and not isinstance(value, Undefined) else value


# The sandbox every Rules node compiles its derived values and messages in, and reads every expression's paths with;
# what it compiles is stateless. Only a rendered message passes through `finalize`; a compiled derived value does not,
# so it keeps the value it read.
_ENVIRONMENT = RecordSandbox(undefined=RuleUndefined, finalize=_rendered)
# The sandbox every check and `applies_when` compiles in: the same, with either side of an `and` or an `or` whose truth
# alone counts deciding when the other is missing. A check renders nothing, so it needs no `finalize`.
_CHECK_ENVIRONMENT = RuleSandbox(undefined=RuleUndefined)


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


class Reader(NamedTuple):
    """A path an expression reads as a number or a date: the helper that reads it, `number` or `date`, and the
    arguments after the value, where they are constants (`date(doc.issued, format='%d.%m.%Y')`)."""

    path: str
    helper: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()


class Reads(NamedTuple):
    """The paths an expression reads: the ones it needs, the ones it only asks about, the global names it calls
    and reads as values, which a record key of the same name would shadow, the paths it calls as if they were
    helpers, a mistyped `firstpresent(x)` say, which are read like any member as well, the paths it reads as a
    number or a date, and the filters and tests it uses that the sandbox does not have, each as `(kind, name)`:
    `("filter", "lowr")`."""

    required: list[str]
    optional: list[str]
    helpers_called: tuple[str, ...] = ()
    helpers_read: tuple[str, ...] = ()
    unknown_calls: tuple[str, ...] = ()
    readers: tuple[Reader, ...] = ()
    unknown_names: tuple[tuple[str, str], ...] = ()


class _Collected(NamedTuple):
    required: list[str]
    optional: list[str]
    called: list[str]
    # The fallbacks `first_present` reads: allowed to be missing, yet asked about by nothing, so they guard nothing.
    lenient: list[str]
    unknown_calls: list[str]
    readers: list[Reader]
    unknown_names: list[tuple[str, str]]
    # Each node that reads a path where the expression needs the value, with the path: `need()` wraps those whose
    # path no guard elsewhere makes optional (`_compile_lazy`).
    needed: list[tuple[nodes.Node, str]]


def _readers_of(call: nodes.Call) -> list[Reader]:
    """The paths a call of `number()`, `date()` or `days_between()` reads as a number or a date.

    Only a path passed as it is, with constant arguments after it, is kept: that much can be read again without
    evaluating the expression. `days_between(a, b)` reads each of its values as `date()` does.
    """
    name = call.node.name if isinstance(call.node, nodes.Name) else None
    if call.dyn_args or call.dyn_kwargs:
        return []
    if name == "days_between":
        return [Reader(path, "date") for argument in call.args if (path := _path_of(argument)) is not None]
    if name not in ("number", "date") or not call.args or (path := _path_of(call.args[0])) is None:
        return []
    extra = call.args[1:]
    if not all(isinstance(argument, nodes.Const) for argument in [*extra, *(item.value for item in call.kwargs)]):
        return []
    return [
        Reader(
            path,
            name,
            tuple(argument.value for argument in extra),
            tuple((item.key, item.value.value) for item in call.kwargs),
        )
    ]


def _root(path: str) -> str:
    return path.split(".")[0].split("[")[0]


def _collect_paths(node: nodes.Node, collected: _Collected, required: bool, lenient: bool = False) -> None:
    # A filter or a test the sandbox does not have is refused when the expression is built, unless it sits in a
    # conditional, where Jinja leaves it to raise when that branch runs: it is noted, as a helper nobody has is.
    if isinstance(node, (nodes.Filter, nodes.Test)):
        kind, known = (
            ("filter", _ENVIRONMENT.filters) if isinstance(node, nodes.Filter) else ("test", _ENVIRONMENT.tests)
        )
        if node.name not in known and (kind, node.name) not in collected.unknown_names:
            collected.unknown_names.append((kind, node.name))
    # A call of a helper reads its arguments, never a member of the helper's name; a value asked about with
    # `has`, `is defined`, `is present` or `default` is allowed to be missing, and so is anything read inside the
    # arguments of `first_present`, which skips a missing value rather than asking about it.
    if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name) and node.node.name in GLOBAL_NAMES:
        collected.called.append(node.node.name)
        collected.readers.extend(reader for reader in _readers_of(node) if reader not in collected.readers)
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
    # Any other call of a path calls what the record holds there as if it were a helper: `firstpresent(x)` reads
    # `firstpresent` like any member, and the path is noted, since a record holds data, never a helper to call.
    if isinstance(node, nodes.Call) and (callee := _path_of(node.node)) and callee not in collected.unknown_calls:
        collected.unknown_calls.append(callee)
    path = _path_of(node)
    if path is not None:
        target = collected.lenient if lenient else collected.required if required else collected.optional
        if path not in target:
            target.append(path)
        if target is collected.required:
            collected.needed.append((node, path))
        return
    for child in node.iter_child_nodes():
        _collect_paths(child, collected, required, lenient)


def _is_under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def _collected(parsed: nodes.Node) -> _Collected:
    """Every path a parsed expression or template reads, as `_collect_paths` sorts it, before a guard elsewhere
    makes one optional (`_reads_of`)."""
    collected = _Collected(
        required=[],
        optional=[],
        called=[],
        lenient=[],
        unknown_calls=[],
        readers=[],
        unknown_names=[],
        needed=[],
    )
    _collect_paths(parsed, collected, required=True)
    return collected


def _reads_of(collected: _Collected) -> Reads:
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
        unknown_calls=tuple(collected.unknown_calls),
        readers=tuple(collected.readers),
        unknown_names=tuple(collected.unknown_names),
    )


def read_paths(expression: str) -> Reads:
    """The paths an expression reads, in order of appearance.

    A path the expression only asks `has`, `is defined`, `is present`, `is blank` or `default` about, or
    offers to `first_present` as a fallback, is optional: it may be missing without stopping the evaluation,
    and so may anything read under it, since `has(docs.FloodCert) and docs.FloodCert.zone == 'A'` is how a
    check guards a read; the guard decides. Every other path is required: a check or an `applies_when` stops at
    one that is missing where it reads it, and only there (`need`), unless the other side of an `and` or an `or`
    whose truth alone it uses decides in its place (`rule_or`). A helper's name is never a read: `days_between(a, b)`
    reads `a` and `b`, while a bare `date` is a member of the record, whatever the record holds under it.
    """
    return _reads_of(_collected(_ENVIRONMENT.parse("{{ " + expression + " }}")))


def read_template(template: str) -> Reads:
    """The paths a message template reads, the way `read_paths` reads an expression."""
    return _reads_of(_collected(_ENVIRONMENT.parse(template)))


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
    raises the clash whenever it is evaluated instead, an error rather than a missing value, before it reads
    anything, so none of its reads is required: a missing one must not pass the clash off as missing, as it would a
    derived value's other failures.
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

    A step reads what Jinja reads there: a key of any mapping, whatever the key's type (`m[0]` over `{0: 'x'}`), an
    item of a list or a tuple, or a character of a text, by its index (`code[0]`). A key a dict lacks is missing,
    though a method of the dict has its name; an attribute is read only of another object, never behind an
    underscore. A path that reaches a value nobody could read ends there: the marker's own members, `value` and
    `reason`, are not the record's, and the path is no more missing than the value is, so a finding shows why the
    value could not be read, and a rule that reads the path is not evaluated for that reason rather than as missing.
    """
    current: Any = context
    for part in _split_path(path):
        if isinstance(current, Unreadable):
            return current
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(part, int):
            if isinstance(current, (list, tuple, str)) and -len(current) <= part < len(current):
                current = current[part]
            else:
                return _MISSING
        elif isinstance(current, dict):
            return _MISSING
        elif hasattr(current, part) and not part.startswith("_"):
            current = getattr(current, part)
        else:
            return _MISSING
    return _MISSING if isinstance(current, Undefined) else current


@pass_context
def need(context: Context, value: Any, path: str) -> Any:
    """What a read an expression needs found, or a `MissingValue` naming the path where the record holds nothing.

    A rule's compiled expression hands each such read here as Jinja reaches it (`_compile_lazy`), so a missing value
    stops the expression where it is read, and only there; on a side of an `and` or an `or` whose truth alone the
    expression uses, it stops that side, which gives way to the other (`rule_or`). Nothing is a null, or an undefined
    or a method Jinja found in place of a value the record lacks: `ticket.items` over a ticket without items finds the
    mapping's method. At a path the record does hold, a method or an undefined goes on as it is and fails where it is
    used: `invoice.items.count` reads the list's method, `ticket.tags.append` one the sandbox refuses.
    """
    if value is None or (
        (isinstance(value, Undefined) or callable(value)) and _is_missing(resolve_path(context.get_all(), path))
    ):
        raise MissingValue.for_path(path)
    return value


def callee(value: Any) -> Any:
    """What the name a call calls holds, or the error of a name nothing defines, raised before the call's arguments
    are read: a rule's compiled expression hands each such name here (`_compile_lazy`), so `firstpresent(x)` is the
    helper no one has, whatever `x` holds, rather than a missing `x`. A name the record holds goes on as it is."""
    if isinstance(value, Undefined):
        value._fail_with_undefined_error()
    return value


def blank_at(value: Any, path: str, raw: bool = False) -> Any:
    """`value`, or a blank that stops a check naming `path` in place of one the check counts as missing (`_needing`):
    a blank a helper made of the value at `path`, or, where `value` is that value itself (`raw`), blank text, which
    `days_between()` counts as missing."""
    if isinstance(value, Blank) or (raw and isinstance(value, str) and not value.strip()):
        return _CHECK_ENVIRONMENT.blank(exc=lambda _: MissingValue.for_path(path))
    return value


# `need`, `callee` and `blank_at` as the compiled code imports them, under names of Jinja's own that no expression can
# reach.
_NEED = f"{__name__}.need"
_CALLEE = f"{__name__}.callee"
_BLANK_AT = f"{__name__}.blank_at"


# The helpers that turn a blank they are handed into a missing value (`blank_at`).
_BLANK_HELPERS = frozenset({"text", "number", "date", "first_present", "days_between"})


def _call(name: str, node: nodes.Expr, *args: nodes.Expr) -> nodes.Call:
    """A call, on `node`'s line, of the function the compiled code imports under `name`, handed `node` and `args`."""
    return nodes.Call(nodes.ImportedName(name), [node, *args], [], None, None, lineno=node.lineno)


def _needing(node: nodes.Node, needed: Mapping[int, str], names: set[str], called: bool = False) -> nodes.Node:
    """`node` with each read under it that the expression needs, in `needed` by the node's id, handed to `need()`
    with its path, and each name it looks up added to `names`. A call's target is never such a read: a name no helper
    or global has goes to `callee()` instead, so `firstpresent(x)`, where the record does not hold the name either,
    raises that the name is undefined before `x` is read. A blank a helper counts as missing names the path the helper
    was handed as it is: the first one `text()`, `number()`, `date()` or `first_present()` is handed, or each one
    `days_between()` is handed (`blank_at`)."""
    path = needed.get(id(node))
    if path is not None and not called:
        names.add(_root(path))
        return _call(_NEED, node, nodes.Const(path))
    if isinstance(node, nodes.Name):
        names.add(node.name)
        if called and node.name not in GLOBAL_NAMES:
            return _call(_CALLEE, node)
        return node
    helper = node.node.name if isinstance(node, nodes.Call) and isinstance(node.node, nodes.Name) else None
    # Read before `need()` wraps the arguments, which are then no longer paths.
    handed = [_path_of(argument) for argument in node.args] if helper in _BLANK_HELPERS else []
    for field, value in node.iter_fields():
        if isinstance(value, nodes.Node):
            setattr(node, field, _needing(value, needed, names, isinstance(node, nodes.Call) and field == "node"))
        elif isinstance(value, list):
            value[:] = [_needing(item, needed, names) if isinstance(item, nodes.Node) else item for item in value]
    if helper == "days_between":
        node.args = [
            argument if path is None else _call(_BLANK_AT, argument, nodes.Const(path), nodes.Const(True))
            for argument, path in zip(node.args, handed)
        ]
    elif handed and handed[0] is not None:
        return _call(_BLANK_AT, node, nodes.Const(handed[0]))
    return node


def _mark_deciding(node: nodes.Node, truth: bool = True) -> None:
    """Sets `deciding` on each `and` and `or` under `node`: true where the expression uses only its truth, so either
    side decides it where the other stops at a missing value (`RuleCodeGenerator`), false where it uses its value.
    `truth` says whether the expression uses only the truth of `node` itself, as a check and an `applies_when` use
    their own.

    Only the truth of an `and` or an `or` counts where it is the operand of `not`, the test of an `if`, a branch of an
    `if` whose truth alone counts, or a side of an `and` or an `or` whose truth alone counts. Anywhere else the
    expression uses its value: compares it, computes with it, filters or tests it, hands it to a helper or takes it as a
    branch of an `if` whose value it uses. There it stays Python's, whose value is the side that decides it, so a
    missing side it reads holds the rule, since a value in its place could change the result.
    """
    if isinstance(node, (nodes.And, nodes.Or)):
        node.deciding = truth
        _mark_deciding(node.left, truth)
        _mark_deciding(node.right, truth)
        return
    for child in node.iter_child_nodes():
        # A branch of an `if` is used as the `if` is; the operand of `not` and the test of an `if`, for their truth.
        branch = isinstance(node, nodes.CondExpr) and child is not node.test
        _mark_deciding(child, truth if branch else isinstance(node, (nodes.Not, nodes.CondExpr)))


class _Lazy:
    """A rule's expression as `_compile_lazy` compiles it, handed only the names it looks up.

    Jinja copies whatever an expression is handed into a new context on every call, and a rule looks up a few of the
    record's names beside every derived value the node computes: the rest would be copied for nothing, for every
    rule of every record. A name it does not look up is one it cannot reach, so the result is the same.
    """

    __slots__ = ("expression", "names")

    def __init__(self, expression: TemplateExpression, names: frozenset[str]) -> None:
        self.expression = expression
        self.names = names

    def __call__(self, scope: Mapping[str, Any]) -> Any:
        return self.expression({name: scope[name] for name in self.names if name in scope})


def _compile_lazy(text: str) -> tuple[_Lazy, Reads]:
    """A check or an `applies_when` compiled as `compile_expression` compiles it, keeping an undefined result, and
    its reads.

    Each read the expression needs goes through `need()`, so a missing value stops it where it is read and nowhere else:
    Jinja evaluates left to right and stops where the result is decided, so the branch of an `if` it does not take, the
    side of an `and` or an `or` its first side decided and the rest of a comparison chain already false are never read.
    It compiles in `RuleSandbox`, where a side of an `and` or an `or` whose truth alone the expression uses
    (`_mark_deciding`) gives way to the other side where it stops at a missing value, and the other side decides in its
    place where it can (`rule_or`); an `and` or an `or` whose value it uses is Python's. A value it only asks about or
    falls back on (`has`, `is defined`, `| default`, `first_present`) is read as it always was. The name a call calls is
    no read: one no helper or global has goes through `callee()`, which Python evaluates before the call's arguments, so
    `firstpresent(x)`, where nothing defines the name, raises that it is undefined whether or not `x` is there; a name
    the record holds is called as it holds it. An undefined result is kept rather than turned into None, so a bare
    `limits[program]` that finds nothing is not evaluated instead of read as false and reported as a verdict. The text
    is parsed as `compile_expression` parses it, so a syntax error names the text as the author wrote it.
    """
    parser = Parser(_CHECK_ENVIRONMENT, text, state="variable")
    try:
        expression = parser.parse_expression()
        if not parser.stream.eos:
            raise TemplateSyntaxError("chunk after expression", parser.stream.current.lineno, None, None)
    except TemplateSyntaxError:
        _CHECK_ENVIRONMENT.handle_exception(source=text)
    collected = _collected(expression)
    reads = _reads_of(collected)
    required = set(reads.required)
    names: set[str] = set()
    expression = _needing(expression, {id(node): path for node, path in collected.needed if path in required}, names)
    _mark_deciding(expression)
    template = nodes.Template([nodes.Assign(nodes.Name("result", "store"), expression, lineno=1)], lineno=1)
    template.set_environment(_CHECK_ENVIRONMENT)
    return _Lazy(TemplateExpression(_CHECK_ENVIRONMENT.from_string(template), False), frozenset(names)), reads


def _check_and_or_decide() -> bool:
    """Compiles and evaluates one check to make sure either side of its `and` and its `or` decides where the other stops
    at a missing value, and warns once, naming the jinja2 release, where it does not; returns whether it does.

    The code generator that makes them decide relies on Jinja's internals (`RuleCodeGenerator`), and a jinja2 release
    that changed those could compile a check's `and` and `or` as Python's again without any error. A check would then
    read them as main does: a missing value on either side stops the rule where the other side should decide it, which
    holds the rule, or skips it under `not_applicable`, and an error the other side would raise goes unseen. The module
    imports all the same, and the tests pin that either side decides; it runs this once, when it is imported, so the
    log names such a release rather than the change going unnoticed. The release is read from the package, not from
    its metadata, which a bundled application may not ship.
    """
    cause = ""
    try:
        check, _ = _compile_lazy("(a.x or b) and not (a.y and c)")
        decided = holds(check({"a": {}, "b": True, "c": False}))
    except MissingValue:
        decided = False
    except Exception as e:
        decided, cause = False, f" (the check raised {type(e).__name__}: {e})"
    if decided:
        return True
    release = getattr(jinja2, "__version__", None) or "(version unknown)"
    logger.warning(
        f"jinja2 {release} does not compile the `and` and the `or` of a Rules check as dynamiq expects: in a check "
        "they will not decide past a missing value on either side, which then stops the rule under its missing-data "
        f"policy; pin jinja2 to a release this version of dynamiq is tested with{cause}"
    )
    return False


_check_and_or_decide()


def _unreadable_because(path: str, reason: str) -> str:
    """Why the value at a path could not be read, in a sentence: `doc.amount is not a number: 'TBD'`, as `number()`
    and `date()` put it, or `ltv is unreadable: division by zero` for a reason that does not say what the value is."""
    return f"{path} is {reason}" if reason.startswith("not a ") else f"{path} is unreadable: {reason}"


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


class _Hold(NamedTuple):
    """Why no rule may skip a missing value, and whether that is only a lookup that found nothing.

    A lookup that found nothing holds a rule only where the rule needs the value: a fallback, `first_present(limit,
    0)`, stands in for it as its author meant. Anything else is a defect of the expression itself, a typo, a name no
    helper has or a value nobody could read, and holds the rule wherever the value is read.
    """

    reason: str
    lookup: bool = False


# What Python says of code nested deeper than it takes: parentheses past 200, indentation past 100, loops past 20.
_NESTING_LIMITS = frozenset(
    {"too many nested parentheses", "too many levels of indentation", "too many statically nested blocks"}
)


def build_error(where: str, error: RecursionError | SyntaxError, what: str = "expression") -> str:
    """Why text fails the build where Python raised `error` on it, naming `where` it is. Jinja reads each level of
    nesting through a dozen calls, so some 70 parentheses exhaust Python's stack; it compiles a chain of operators to
    code nested a level for each, which Python's parser refuses at 200 levels, and a message's nested blocks to Python
    blocks, refused past 100 levels of indentation or 20 loops: such text is nested too deeply to read, as `workflow
    validate` puts it. Any other `SyntaxError`, from the Python parser Jinja's lexer reads a number
    with (`1١.5`), makes the text no valid `what`, in the parser's words, as a syntax error Jinja finds does."""
    if isinstance(error, RecursionError) or error.msg in _NESTING_LIMITS:
        return f"{where} is nested too deeply to read; split it into smaller expressions"
    return f"{where} is not a valid {what}: {error.msg}"


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

    Inputs arrive by name and rules read them by path (`docs.Note.interest_rate`); derived values are computed once per
    record, in order, before the rules run, and are read by name like an input. Expressions run in a `RecordSandbox`, as
    the Expression node's do, a check and an `applies_when` in a `RuleSandbox`, one where either side of an `and` or an
    `or` whose truth alone counts can decide alone (see Missing values), with the helpers `has`, `days_between`, `date`,
    `today`, `len`, `abs`, `min`, `max`, `sum`, `round`, `text`, `number` and `first_present` and the tests `is present`
    and `is blank`. A record member named like a helper is the member where a rule reads it and the helper where a rule
    calls it; one named like a method of the record, `items` say, is the member, the method reached only for a key the
    record lacks. Rules compile when the node is built, so a malformed expression fails then, naming the rule.

    - Statuses: `pass` when the check holds; the rule's severity (`fail`, `warn`, `info`) when it does not;
      `not_applicable` when `applies_when` does not hold, the record's `as_of` date is outside the rule's effective
      window, or the rule skips a missing value; `not_evaluated` when the check or `applies_when` stops at a value that
      is missing, or cannot be evaluated. A rule reported at its severity gives its own message, where it has one,
      rendered with the whole record, where a derived value nobody could compute reads as None, as `derived` shows it;
      one that did not run or did not apply says why. `evaluated` holds the values the check reads, empty where the rule
      did not apply.
    - Missing values: a check and `applies_when` are evaluated left to right, and a missing value counts only where the
      evaluation reaches it. The branch of an `if` not taken, the side of an `and` or an `or` its first side already
      decided and the rest of a comparison chain already false are never read, so a value missing there never stops the
      rule; where the rule stops at a missing value elsewhere, though, what those parts read can still keep it from
      being skipped, or make it an error (see Never skipped). A missing value the check only asks about, with `has`, a
      test such as `is defined` or `is present` or the `default` filter, does not hold the rule either, since the
      question answers for it, and one it offers to `first_present` holds it only where every value offered is missing.
      Any other value the evaluation reaches counts, even where the rest of the check would decide without it: every
      value a call, a filter or a list is handed is read, so `app.a | default(app.b)` is held without `app.b` though
      `app.a` is there, and `app.a in [app.b, app.c]` without `app.c` whatever `app.b` holds. The one exception is an
      `and` or an `or` whose truth alone counts, as the check or `applies_when` itself, under `not`, as the test of an
      `if`, as a branch of an `if` whose truth alone counts, or as a side of another such `and` or `or`: either side
      decides it where the other stops at a missing value. `a or b` is true where `b` is, and `a and b` false where `b`
      is, whether or not `a` is there, so `app.occupancy == 'primary' or app.purpose == 'purchase'` passes a purchase
      without an occupancy; a lazy side, what `select` yields say, decides by its items. Where the check uses its value
      instead, compares it, computes with it, filters or tests it, hands it to a helper or takes it as a branch of an
      `if` whose value it uses, an `and` or an `or` is Python's: the side that decides it is its value, and a missing
      side it reads holds the rule, since a value in its place could change the verdict. `(app.nickname or app.name) ==
      'Ann'` is held without a nickname, as `(app.fee or 100) > 50` is without a fee, where `first_present(app.nickname,
      app.name) == 'Ann'` falls back as its author means. Where nothing decides without it, the first value needed that
      is missing holds the rule, of two on either side of an `and` or an `or` the left one, and the reason names that
      value; where a helper turned a blank it was handed into a missing value, `text()` of blank text say, the reason
      names the path it was handed, or, where it was handed anything else, a blank value the check reads. An error the
      evaluation reaches is an error, whatever the expression would have read after it, on either side of an `and` or
      an `or` as well: only a missing value gives way to the other side. A call of a name nothing defines, neither a
      helper nor the record, is an error before its arguments are read. A message decides nothing and reads a missing
      value as it always did.
    - Derived values are computed as they always were: a missing value stops one only where the expression uses it, so a
      null it falls back past still computes, `(x or 0) < 3` is true over a null `x`, and a list or a dict it builds
      holds None for a member the record lacks; a failure beside a missing value it needs makes it missing; and `and`
      and `or` are Python's, so a side that is not there, or a blank a helper made, makes the value missing though the
      other side would decide. Naming part of a check as a derived value can therefore change what the rule reports, and
      make it looser as well as stricter: the check `(x or 0) < 3` stops at the null `x`, where a check that reads the
      same text as a derived value decides; the check `a or b` passes where `b` is true, where a derived `a or b` is
      missing without `a`; and a derived value whose lookup found nothing is missing, so it gives way to the other side
      of an `and` or an `or` whose truth alone counts, where the same lookup written in the check is an error: with
      `lim` derived as `limits[app.program]` for a program the table lacks, `lim > 5 or app.a == 1` passes where
      `limits[app.program] > 5 or app.a == 1` is not evaluated.
    - Policies: `on_missing` on the node, which a rule's own overrides, says what a missing value means:
      `not_evaluated`, the default, holds the rule for review; `fail` reports its severity, the reason after its
      message; `not_applicable` skips it ("does not apply: missing value for …"), so the rule needs no presence
      guard. A missing value never passes or fails a rule silently.
    - Never skipped: only data the record lacks is skipped, and only where a read names it. Where a missing value
      stops a rule's expression, the rule is held rather than skipped, `not_evaluated` or its severity under
      `fail`, when that expression
        - reads or asks about a value nobody could read: text `number()`, `date()` or `days_between()` cannot read
          (`TBD` for an amount), or a derived value nobody could compute (`has(ltv)` over a ratio divided by zero);
        - calls a name no helper has (`firstpresent(x)`), or uses a filter or a test no sandbox has, which Jinja
          leaves inside a conditional to raise only when that branch runs;
        - finds a value missing under a name the node does not declare, where it sets `input_fields` (a typo; the
          selector's keys, the derived values and `as_of` are declared as well);
        - needs a lookup that found nothing, `limits[loan.program]` for a program the table lacks, in the check or
          in a derived value it does not only fall back on (`first_present(limit, 500000)`);
        - meets a blank no read accounts for, from a lookup inside `text()` say.
      Once a missing value stops the expression, a value nobody could read, a name no helper has, a filter or a test no
      sandbox has, a value missing under a name the node does not declare and a derived value it needs whose lookup
      found nothing hold the rule wherever the expression has them, in a branch not taken or on a side the evaluation
      never read as well: that holds more rules than the evaluation alone would, never fewer. A derived value nobody
      could compute speaks over the missing value under every policy: `(app.x if app.k else ltv) == 1` with the flag
      set, `app.x` missing and `ltv` a ratio divided by zero is held for the division rather than for `app.x`. A derived
      value that came out missing counts as data the record lacks when a value it reads is missing, whether or not its
      evaluation reached that value. Under `not_applicable` the reason says why the rule was not skipped, `missing value
      for loan.amount (not skipped: lon is not an input or a derived value)`, unless it is already the error of a value
      nobody could read.
    - What a skipped rule cannot see, nor one the other side of an `and` or an `or` decides: an error its check would
      raise after the missing value it stopped at, on values that are there: a zero divisor (`appraisal.max_ltv >=
      loan.amount / appraisal.value` without a limit), a misspelled method or a value of the wrong type; that surfaces
      on the records that carry the missing value. So `app.b / app.q > 1 or app.a ==
      1` passes where `app.b` is missing and `app.a` is 1, though `app.q` is 0: the division fails only where `app.b` is
      there. An error the check reaches before the missing value, or on the other side of an `and` or an `or` the
      missing value gives way to, is reported with or without the value: `loan.amount / appraisal.value <=
      appraisal.max_ltv` over a zero value, with or without the limit, and `app.age >= 18 and
      text(app.name).startwith('A')` without the age as with an age of 18 or more.
    - Overall status: `fail` if any rule failed, else `warn` if any warned, else `not_evaluated` if any check did not
      run, else `pass`. A check held for a missing value did not run, so a record is never `pass` while a check or an
      `applies_when` stopped at a missing value, unless every rule that did was set to skip it.

    The output holds `findings` in rule order, which a trace keeps whole, a `summary` of statuses, `status`, the
    `derived` values and `derived_errors`. A derived value computed from a missing value is missing, None under
    `derived`; one that could not be computed from the values that are there, a division by zero or `number()` of
    `TBD`, or from one nobody could read, is None there too, with the reason under `derived_errors`, and every rule
    that reads it is held, naming the reason. Where a missing value stops the expression before a reader in it runs,
    `doc.rate * number(doc.net)` with the rate missing and the net `TBD`, the value is missing instead, though a rule
    set to skip is still held on it where the reader is handed a path as it is, as here. An optional `as_of` input,
    a date, fixes the day the effective windows are compared with; without it the run date is used.
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
    # The names an expression may read from where the node declares its inputs: those, the keys its input transformer's
    # selector maps, which the record holds whether or not `input_fields` lists them, the derived values and `as_of`.
    # None where it declares no input, since a record read then cannot be told from a typo.
    _declared: set[str] | None = PrivateAttr(default=None)
    # Whether some rule skips a missing value by its own policy, and whether some rule leaves its policy to the node.
    _rules_skip: bool = PrivateAttr(default=False)
    _rules_defer: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._derived = self._compile_derived()
        self._compiled = self._compile_rules()
        if self.input_fields:
            names = {field.name for field in self.input_fields} | set(self.input_transformer.selector or {})
            self._declared = names | {name for name, _, _ in self._derived} | {AS_OF_KEY}
        self._rules_skip = any(item.rule.on_missing == RuleMissingPolicy.NOT_APPLICABLE for item in self._compiled)
        self._rules_defer = any(item.rule.on_missing is None for item in self._compiled)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"rules": True}

    def to_dict(self, include_secure_params: bool = True, for_tracing: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        A trace keeps the first rules and the total: `findings`, kept whole in a trace, already carries every
        rule that mattered, and a large rule set here would otherwise be copied into every run it takes part in.
        """
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        rules = self.rules[:TRUNCATE_LIST_LIMIT] if for_tracing else self.rules
        data["rules"] = [rule.model_dump(**kwargs) for rule in rules]
        if for_tracing:
            data["rules_count"] = len(self.rules)
        return data

    def transform_output(self, output_data: Any, **kwargs) -> Any:
        """The output as the node's output transformer shapes it, with `findings` a list a trace keeps whole.

        `execute` marks the findings so, but an output the node cache replays comes back through JSON, as plain lists,
        and a mocked one was never marked; both pass through here on their way out as well, and are marked again.
        """
        findings = output_data.get("findings") if isinstance(output_data, dict) else None
        if isinstance(findings, list) and not isinstance(findings, UntruncatedList):
            output_data = {**output_data, "findings": UntruncatedList(findings)}
        return super().transform_output(output_data, **kwargs)

    def _compile_expression(self, text: str, where: str, *, lazy: bool = True) -> tuple[Callable[..., Any], Reads]:
        try:
            if lazy:
                # A check and `applies_when` stop at a missing value only where they read it, and either side of
                # an `and` or an `or` whose truth alone they use decides where the other is missing (`RuleSandbox`).
                compiled, reads = _compile_lazy(text)
            else:
                # A derived value is computed as it always was. A lookup that finds nothing must come back as
                # RuleUndefined, whose truth test raises, rather than be turned into None on the way out: a bare
                # `limits[program]` is then not evaluated instead of read as false and reported as a verdict.
                # Compiled before its reads are collected, from the text wrapped in braces, so a syntax error names
                # the text as the author wrote it.
                compiled = _ENVIRONMENT.compile_expression(text, undefined_to_none=False)
                reads = read_paths(text)
        except TemplateSyntaxError as e:
            raise ValueError(f"{where} is not a valid expression: {e}") from e
        except (RecursionError, SyntaxError) as e:
            raise ValueError(build_error(where, e)) from None
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
            compiled.append((value.name, *self._compile_expression(value.expression, label, lazy=False)))
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
                except (RecursionError, SyntaxError) as e:
                    raise ValueError(build_error(f"{label}: the message", e, "template")) from None
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
        # The derived values missing for a reason other than data the record lacks, which no rule may skip, each with
        # the reason: a lookup that found nothing, a typo, a call of a name no helper has. They differ from record to
        # record. Only a rule that may skip a missing value needs them, so under the other policies nothing is judged.
        unskippable: dict[str, _Hold] = {}
        judged = self._rules_skip or (self._rules_defer and self.on_missing == RuleMissingPolicy.NOT_APPLICABLE)
        # The derived values not computed yet, which a derived value can only read as missing.
        pending = {name for name, _, _ in self._derived} if judged else set()
        for name, expression, reads in self._derived:
            # One mapping, derived winning, passed positionally: an undeclared key the upstream payload carries
            # under a derived value's name would otherwise clash as a duplicate keyword argument, and a key
            # named `self` would collide with the compiled expression's own bound argument. Such a key is never
            # read: Jinja binds the name inside the expression, so a read of it is refused at build.
            known = {**context, **values}
            # How the value came out missing, where it did: undefined, as a key the record lacks or a lookup that
            # finds nothing does, and blank, as `number()` of blank text does.
            undefined = blank = False
            try:
                result = expression(scope_for(reads, known, RuleUndefined))
                undefined, blank = isinstance(result, Undefined), isinstance(result, Blank)
                # A value the expression could not find is missing, inside a list or a dict it built as well,
                # and the output stays serializable.
                value = concrete(result)
            except UndefinedError as e:
                # So is one it used, arithmetic on a key the record lacks or on a blank `number()` read, unless
                # it also reads a value nobody could read, which the missing one must not hide: `doc.rate * net`
                # over a missing rate is as unreadable as `net`, for the same reason.
                undefined, blank = True, isinstance(e, MissingValue)
                _, value = self._unreadable(reads.required, known)
            except UnreadableValue as e:
                # A value nobody could read makes what is computed from it unreadable, whatever else is missing.
                value = Unreadable(e.value, str(e))
            except EVALUATION_ERRORS as e:
                # A failure while a value the expression needs is missing, `amount / value` over a null `value`,
                # is that value missing too, unless it also reads a value nobody could read. Any other failure
                # is an error: as None it would read as a value nobody gave, one `first_present` or a guard skips.
                if self._missing(reads.required, known):
                    _, value = self._unreadable(reads.required, known)
                else:
                    value = Unreadable(None, str(e))
            values[name] = value
            if isinstance(value, Unreadable):
                # The rules read it as unreadable, keeping the value the error names, if any, for a message to
                # print; the output reports None and the reason.
                derived[name] = None
                derived_errors[name] = value.reason
            else:
                derived[name] = value
                if (
                    judged
                    and value is None
                    and (hold := self._why_missing(name, reads, known, undefined, blank, pending, unskippable))
                    is not None
                ):
                    unskippable[name] = hold
            pending.discard(name)
        scope = {**context, **values}
        # What a message reads: a value nobody could read as None, as `derived` shows it and as it was before derived
        # values kept their errors, so a message's own guard, `{% if has(ltv) %}`, decides as it did rather than the
        # message come back as written. The rules read the marker itself.
        unreadable = {name: None for name, value in scope.items() if isinstance(value, Unreadable)}
        shown = {**scope, **unreadable} if unreadable else scope

        # A trace keeps this whole rather than cutting it to `TRUNCATE_LIST_LIMIT`: past that many rules,
        # the platform UI still needs every rule's own finding to show its coverage and last result.
        findings: list[dict[str, Any]] = UntruncatedList()
        screened = True
        for compiled in self._compiled:
            finding, evaluated = self._evaluate(compiled, scope, shown, as_of, unskippable)
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
            raise ValueError(f"Rules: '{AS_OF_KEY}' is not a date: {_quoted(value)}") from e

    def _evaluate(
        self,
        compiled: CompiledRule,
        scope: dict[str, Any],
        shown: dict[str, Any],
        as_of: date,
        unskippable: Mapping[str, _Hold],
    ) -> tuple[dict[str, Any], bool]:
        """The finding for one rule, and whether its check ran: a missing value or an error means it did not,
        unless the rule skips the missing value, which counts as a rule that did not apply. The rule reads `scope`;
        its message reads `shown`, where a value nobody could read is None (`execute`)."""
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

        status, reason, evaluated = self._status(compiled, scope, unskippable)
        finding["status"] = status
        if status != STATUS_NOT_APPLICABLE:
            reads = compiled.check_reads.required + compiled.check_reads.optional
            finding["evaluated"] = {path: _shown(resolve_path(scope, path)) for path in reads}
        # A finding that did not fire still says why: a reviewer reading the list should never have
        # to re-run the record to learn which condition or date kept a rule out.
        if status in (STATUS_NOT_EVALUATED, STATUS_NOT_APPLICABLE):
            finding["message"] = reason
        elif status != STATUS_PASSED:
            finding["message"] = self._render(compiled, shown, reason)
        return finding, evaluated

    @staticmethod
    def _window(rule: Rule) -> str:
        if rule.effective_from and rule.effective_until:
            return f"from {rule.effective_from} to {rule.effective_until}"
        return f"from {rule.effective_from}" if rule.effective_from else f"until {rule.effective_until}"

    def _status(
        self, compiled: CompiledRule, scope: dict[str, Any], unskippable: Mapping[str, _Hold]
    ) -> tuple[str, str | None, bool]:
        """The rule's status, the reason when it did not run or did not apply, and whether its check ran.

        `unskippable` names the derived values missing in this record for a reason other than data it lacks.
        """
        if compiled.applies is not None:
            reads = compiled.applies_reads
            try:
                applies = holds(compiled.applies(scope_for(reads, scope, RuleUndefined)))
            except MissingValue as e:
                path, reason = self._missing_reason(reads, scope, e)
                return self._missing_status(compiled, "applies_when", reads, scope, unskippable, path, reason)
            except EVALUATION_ERRORS as e:
                return self._error_status(compiled, f"applies_when could not be evaluated: {e}")
            if not applies:
                return STATUS_NOT_APPLICABLE, f"does not apply: {compiled.rule.applies_when.strip()}", True

        reads = compiled.check_reads
        try:
            held = holds(compiled.check(scope_for(reads, scope, RuleUndefined)))
        except MissingValue as e:
            path, reason = self._missing_reason(reads, scope, e)
            return self._missing_status(compiled, "check", reads, scope, unskippable, path, reason)
        except EVALUATION_ERRORS as e:
            return self._error_status(compiled, f"check could not be evaluated: {e}")
        return (STATUS_PASSED, None, True) if held else (compiled.rule.severity.value, None, True)

    def _policy(self, compiled: CompiledRule) -> RuleMissingPolicy:
        """What a missing value means for the rule: its own policy, or the node's where it sets none."""
        return compiled.rule.on_missing or self.on_missing

    def _missing_status(
        self,
        compiled: CompiledRule,
        where: str,
        reads: Reads,
        scope: dict[str, Any],
        unskippable: Mapping[str, _Hold],
        path: str | None,
        reason: str | None = None,
    ) -> tuple[str, str, bool]:
        """The status of an expression that stopped at a missing value, which `path` names where a read accounts
        for it, under the rule's policy.

        `not_applicable` skips the rule, and a skipped rule counts as having run, so a record whose only unmet rules
        lacked their data can still pass. It skips only data the record lacks: where the expression holds anything
        else no rule may skip (`_why_held`), or no value it reads accounts for the stop, the rule is not evaluated and
        the reason says why it was not skipped. Under `fail` and `not_evaluated` the reason stays as it is. Where the
        expression reads a value nobody could read, whose reason a missing value must not hide (`ltv <=
        appraisal.max_ltv` over a ratio divided by zero), the stop is an error instead, under every policy, and its
        reason already names the cause.

        Both look at every value the expression reads, whether or not the evaluation reached it, in a branch not taken
        say: `(app.x if app.k else ltv) == 1` with the flag set, `app.x` missing and `ltv` a ratio divided by zero is
        the error of `ltv`. That is deliberate, and stricter than the evaluation, never looser: what they find can keep
        a rule from being skipped, or name an error where the missing value would have been named, but never skips a
        rule or gives it a verdict.
        """
        reason = reason or f"missing value for {path}"
        # Compared with None: the marker refuses a truth test, as every other use.
        _, unreadable = self._unreadable(reads.required + reads.optional, scope)
        if unreadable is not None:
            return self._error_status(compiled, f"{where} could not be evaluated: {unreadable.reason}")
        policy = self._policy(compiled)
        if policy == RuleMissingPolicy.FAIL:
            return compiled.rule.severity.value, reason, False
        if policy != RuleMissingPolicy.NOT_APPLICABLE:
            return STATUS_NOT_EVALUATED, reason, False
        hold = self._why_held(reads, scope, self._declared, unskippable)
        if hold is None and path is None:
            hold = _Hold("no field of the record is named")
        if hold is None:
            return STATUS_NOT_APPLICABLE, f"does not apply: {reason}", True
        return STATUS_NOT_EVALUATED, f"{reason} (not skipped: {hold.reason})", False

    def _error_status(self, compiled: CompiledRule, reason: str) -> tuple[str, str, bool]:
        """The status of an expression that could not be evaluated: the rule's severity under `fail`, otherwise not
        evaluated. Never not applicable, whatever the policy: a rule skips only data the record lacks."""
        if self._policy(compiled) == RuleMissingPolicy.FAIL:
            return compiled.rule.severity.value, reason, False
        return STATUS_NOT_EVALUATED, reason, False

    def _why_held(
        self,
        reads: Reads,
        scope: dict[str, Any],
        declared: Container[str] | None,
        unskippable: Mapping[str, _Hold],
        pending: Container[str] = frozenset(),
        reading: str | None = None,
    ) -> _Hold | None:
        """Why what an expression finds missing is not only data the record lacks, so no rule may skip it, or None
        when it is. The reason names the name or the path at fault.

        The expression may call a name that is no helper and that the record does not hold, a mistyped
        `firstpresent`, or use a filter or a test the sandbox does not have, `x | lowr` inside a conditional, which
        Jinja leaves to raise until that branch runs. It may read text through `number()` or `date()` that they
        cannot read, which a missing value must not hide (`number(doc.amount) > doc.limit` over an amount of `TBD`);
        a value that is already one nobody could read, an earlier derived value say, its callers look for first
        (`_missing_status`, `_why_missing`). A value it finds missing may sit under a name outside `declared`, a
        typo, where the node declares its inputs (None where it does not), or under a derived value in `pending`,
        one computed after `reading`, the derived value this expression computes. Or it may read a derived value
        held for any of these defects, wherever it reads it, or one a lookup found nothing for, where it needs the
        value rather than falls back on it: a gap in a table or in the node, never in the record. A defect speaks
        over a lookup that found nothing, whichever the expression reads first.
        """
        if (callee := self._missing(list(reads.unknown_calls), scope)) is not None:
            return _Hold(f"{callee} is not a helper")
        if reads.unknown_names:
            kind, name = reads.unknown_names[0]
            return _Hold(f"{name} is not a {kind}")
        paths = reads.required + reads.optional
        if (misread := self._misread(reads.readers, scope)) is not None:
            return _Hold(misread)
        # A defect holds wherever it is read and speaks over a lookup that found nothing, whichever the expression
        # reads first; a fallback stands in for the lookup, as its author meant (`first_present(limit, 0)`), so that
        # holds only where the value is needed.
        lookup: _Hold | None = None
        for path in paths:
            if not is_blank(resolve_path(scope, path)):
                continue
            root = _root(path)
            if root in pending:
                return _Hold(f"{root} is computed after {reading}")
            if declared is not None and root not in declared:
                return _Hold(f"{root} is not an input or a derived value")
            if (hold := unskippable.get(root)) is None:
                continue
            if not hold.lookup:
                return hold
            if lookup is None and path in reads.required:
                lookup = hold
        return lookup

    def _why_missing(
        self,
        name: str,
        reads: Reads,
        known: dict[str, Any],
        undefined: bool,
        blank: bool,
        pending: Container[str],
        unskippable: Mapping[str, _Hold],
    ) -> _Hold | None:
        """Why a derived value that came out missing is not only data the record lacks, or None when it is, as the
        same expression written in a check would be; `pending` holds the derived values not computed yet.

        There, a blank from `text()`, `number()`, `date()` or `first_present()` is missing when a value it read is
        missing or blank, and an undefined result when a value it needs is missing. With every value it needs
        there, an undefined result is a lookup that found nothing, `limits[loan.program]` for a program the table
        lacks, which a check reports as an error, unless the expression has a defect of its own, a call of a name no
        helper has say (`firstpresent(x) | default(0)`), which is the cause instead. A None the expression computes
        itself, with `else none` say, is its author's answer, and counts as data the record lacks so long as the
        values it reads do (`_why_held`). A value it reads that nobody could read, an earlier derived value say,
        holds it before anything else, as it would the same expression in a check.
        """
        found, unreadable = self._unreadable(reads.required + reads.optional, known)
        if unreadable is not None:
            return _Hold(_unreadable_because(found, unreadable.reason))
        hold = self._why_held(reads, known, self._declared, unskippable, pending, name)
        if hold is not None and not hold.lookup:
            return hold
        if blank:
            accounted = any(is_blank(resolve_path(known, path)) for path in reads.required + reads.optional)
        elif undefined:
            accounted = self._missing(reads.required, known) is not None
        else:
            accounted = True
        return hold if accounted else _Hold(f"the lookup for {name} found nothing", lookup=True)

    @staticmethod
    def _missing(paths: list[str], scope: dict[str, Any]) -> str | None:
        for path in paths:
            if _is_missing(resolve_path(scope, path)):
                return path
        return None

    @staticmethod
    def _misread(readers: tuple[Reader, ...], scope: dict[str, Any]) -> str | None:
        """Why a value an expression reads as a number or a date, and that is there, is one its reader cannot read,
        or cannot read with the arguments it is given (`decimal=';'`), naming its path; None when every one reads.

        The reader runs again on the value alone, as the expression would run it: a missing value stays missing,
        and one already unreadable is left to the scan for those.
        """
        for reader in readers:
            value = resolve_path(scope, reader.path)
            if is_blank(value) or isinstance(value, Unreadable):
                continue
            try:
                read = HELPERS[reader.helper](_ENVIRONMENT, value, *reader.args, **dict(reader.kwargs))
            except EVALUATION_ERRORS as e:
                return f"{reader.path} could not be read: {e}"
            if isinstance(read, Unreadable):
                return _unreadable_because(reader.path, read.reason)
        return None

    @staticmethod
    def _unreadable(paths: list[str], scope: dict[str, Any]) -> tuple[str, Unreadable] | tuple[None, None]:
        """The first path that reaches a value nobody could read, and the value with its reason; (None, None) when
        there is none."""
        for path in paths:
            if isinstance(value := resolve_path(scope, path), Unreadable):
                return path, value
        return None, None

    @staticmethod
    def _missing_reason(reads: Reads, scope: dict[str, Any], error: MissingValue) -> tuple[str | None, str]:
        """The value an expression that stopped at a missing value could not decide on, and why.

        The error names the path the evaluation stopped at, where a read the expression needs found nothing (`need`)
        or a helper turned a blank it was handed as a path into a missing value (`blank_at`). A blank a helper made of
        anything else, a filtered value say, has no path, so the value named is the first the expression reads that is
        missing or blank; a blank no read accounts for, one made by a lookup that found nothing say, keeps the error's
        own message.
        """
        if error.path:
            return error.path, f"missing value for {error.path}"
        for path in reads.required + reads.optional:
            if is_blank(resolve_path(scope, path)):
                return path, f"missing value for {path}"
        return None, str(error)

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
