"""A missing value counts only where a check's evaluation reaches it.

Jinja evaluates an expression left to right and stops where the result is decided: the branch of an `if` it does not
take, the side of an `and` or an `or` its first side already decided and the rest of a comparison chain already false
are never read, so a value missing there cannot change the result. A check and an `applies_when` each stop at a missing
value where they read it, and name the value they stopped at; a call of a name nothing defines, neither a helper nor the
record, is an error before its arguments are read. Where only the truth of an `and` or an `or` counts, as the check
itself, under `not`, as the test of an `if`, as a branch of an `if` whose truth alone counts or as a side of another
such `and` or `or`, either side decides where the other stops at a missing value: `a or b` is true where `b` is, and
`a and b` false where `b` is, whether or not `a` is there; where the side that is there cannot decide, the missing value
stands, the left one where both are missing. Where the check uses its value, compared or computed with, an `and` or an
`or` is Python's, and a missing side it reads holds the rule. Only a missing value gives way: an error on either side is
an error. A path reads what Jinja reads, a character of a text by its index and a key of any mapping. A value an
expression only asks about (`has`, `is present`, `| default`, `first_present`) was never needed, and reads as it always
did; so does everything in a message, which decides nothing, in an Expression node, which reads a missing input as None,
and in a derived value, which is computed as it always was, `and` and `or` included.
"""

import importlib.metadata
import importlib.util
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import MappingProxyType, ModuleType, SimpleNamespace

import jinja2
import pytest
from jinja2 import nodes
from jinja2.compiler import CodeGenerator
from jinja2.visitor import NodeVisitor

from dynamiq.nodes.operators import Expression, Rules
from dynamiq.nodes.operators import rules as rules_module
from dynamiq.nodes.operators.rules import has, read_paths, resolve_path
from dynamiq.nodes.types import DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

POLICIES = [None, "not_evaluated", "fail", "not_applicable"]


def run(node: Rules, data: dict) -> dict:
    result = node.run(input_data=data, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output


def outcome(output: dict) -> tuple[str, str | None]:
    (finding,) = output["findings"]
    return finding["status"], finding["message"]


def screening(
    check: str,
    policy: str | None = None,
    *,
    inputs: tuple[str, ...] = ("app",),
    derived: tuple[tuple[str, str], ...] = (),
    applies_when: str | None = None,
    message: str | None = None,
    node_policy: str = "not_evaluated",
) -> Rules:
    """A node with one `fail` rule, `rule`, reading the named inputs and derived values."""
    return Rules(
        name="screening",
        input_fields=[NamedField(name=name) for name in inputs],
        derived_values=[DerivedValue(name=name, expression=expression) for name, expression in derived],
        on_missing=node_policy,
        rules=[
            Rule(id="rule", check=check, applies_when=applies_when, message=message, on_missing=policy),
        ],
    )


# --- a check reads only what its evaluation reaches -------------------------------------------------------------


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"purpose": "purchase"}, ("pass", None)),
        ({"purpose": "refinance"}, ("not_evaluated", "missing value for app.occupancy")),
        ({"purpose": "refinance", "occupancy": None}, ("not_evaluated", "missing value for app.occupancy")),
        ({"purpose": "refinance", "occupancy": "primary"}, ("pass", None)),
    ],
    ids=["decided-by-the-first-side", "second-side-absent", "second-side-null", "second-side-there"],
)
def test_an_or_decided_by_its_first_side_never_reads_the_second(app, expected):
    node = screening("app.purpose == 'purchase' or app.occupancy == 'primary'")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"amount": 0}, ("fail", None)),
        ({"amount": 5}, ("not_evaluated", "missing value for app.rate")),
        ({"amount": 5, "rate": 7}, ("pass", None)),
    ],
    ids=["decided-by-the-first-side", "second-side-absent", "second-side-there"],
)
def test_an_and_decided_by_its_first_side_never_reads_the_second(app, expected):
    node = screening("app.amount > 0 and app.rate < 10")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"kind": "arm"}, ("not_evaluated", "missing value for app.margin")),
        ({"kind": "arm", "margin": 2}, ("pass", None)),
        ({"kind": "fixed", "rate": 5}, ("pass", None)),
        ({"kind": "fixed"}, ("not_evaluated", "missing value for app.rate")),
    ],
    ids=["else-branch-missing", "else-branch-there", "if-branch-there", "if-branch-missing"],
)
def test_an_if_reads_only_the_branch_it_takes_and_names_the_value_it_stopped_at(app, expected):
    node = screening("app.rate < 7 if app.kind == 'fixed' else app.margin < 3")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"low": 5, "mid": 1}, ("fail", None)),
        ({"low": 1, "mid": 5}, ("not_evaluated", "missing value for app.high")),
        ({"low": 1, "mid": 5, "high": 9}, ("pass", None)),
    ],
    ids=["decided-by-the-first-comparison", "last-operand-absent", "last-operand-there"],
)
def test_a_comparison_chain_already_false_never_reads_the_rest(app, expected):
    node = screening("app.low < app.mid < app.high")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("check", "app", "missing"),
    [
        ("app.a | default(app.b) == 1", {"a": 1}, "app.b"),
        ("app.a in [app.b, app.c]", {"a": 1, "b": 1}, "app.c"),
        ("max(app.a, app.b) > 0", {"a": 1}, "app.b"),
    ],
    ids=["the-fallback-a-filter-is-handed", "a-member-of-a-list", "an-argument-of-a-call"],
)
def test_every_value_a_filter_a_list_or_a_call_is_handed_is_read_though_the_rest_would_decide_without_it(
    check, app, missing
):
    """The evaluation reaches every value a filter, a list or a call is handed before it uses any, so a missing one
    holds the rule, though `app.a` is there for `default` to keep and already in the list."""
    node = screening(check)

    assert outcome(run(node, {"app": app})) == ("not_evaluated", f"missing value for {missing}")


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({}, ("pass", None)),
        ({"limit": 1}, ("fail", None)),
        ({"limit": 4}, ("pass", None)),
    ],
    ids=["the-fallback-decides", "read-alone-as-well", "there"],
)
def test_a_fallback_stands_in_for_a_value_the_check_also_needs_where_it_reads_it_alone(app, expected):
    """The limit is a fallback inside `first_present` and needed where the check reads it alone: missing, it gives
    way to 5 in the one, and the other is never read."""
    node = screening("first_present(app.limit, 5) > 1 or app.limit > 3")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("docs", "expected"),
    [
        ({}, ("fail", None)),
        ({"cert": {"zone": "A"}}, ("pass", None)),
        ({"cert": {}}, ("not_evaluated", "check could not be evaluated: 'dict object' has no attribute 'zone'")),
    ],
    ids=["guard-decides", "there", "missing-under-the-guard"],
)
def test_a_value_under_one_a_guard_asks_about_is_read_as_it_always_was(docs, expected):
    """`has(docs.cert)` guards every read under it, so none of them is a value the check needs: a zone missing from a
    certificate that is there fails its comparison, as it always did."""
    node = screening("has(docs.cert) and docs.cert.zone == 'A'", inputs=("docs",))

    assert outcome(run(node, {"docs": docs})) == expected


@pytest.mark.parametrize(("strict", "expected"), [(True, ("pass", None)), (False, ("fail", None))])
def test_a_name_the_check_looks_up_without_reading_it_as_a_value_still_reaches_it(strict, expected):
    """`strict`, handed to `default` by keyword, is no read the check needs or asks about; it is looked up all the
    same, so the check is handed it with the names it reads."""
    node = screening("app.note | default('none', boolean=strict) == 'none'", inputs=("app", "strict"))

    assert outcome(run(node, {"app": {"note": ""}, "strict": strict})) == expected


def test_applies_when_reads_only_what_decides_it():
    node = screening("app.points <= 2", applies_when="app.kind == 'fixed' and app.rate > 6")

    assert outcome(run(node, {"app": {"kind": "arm"}})) == (
        "not_applicable",
        "does not apply: app.kind == 'fixed' and app.rate > 6",
    )
    assert outcome(run(node, {"app": {"kind": "fixed"}})) == ("not_evaluated", "missing value for app.rate")
    assert outcome(run(node, {"app": {"kind": "fixed", "rate": 7, "points": 1}})) == ("pass", None)


def test_a_value_under_a_key_that_is_no_name_is_named_as_the_check_writes_it():
    node = screening("doc.signed or doc['Issue Date'] == '2026-01-01'", inputs=("doc",))

    assert outcome(run(node, {"doc": {"signed": True}})) == ("pass", None)
    assert outcome(run(node, {"doc": {"signed": False}})) == (
        "not_evaluated",
        "missing value for doc['Issue Date']",
    )


def test_a_key_the_record_lacks_is_missing_though_a_method_of_the_mapping_has_its_name():
    """`ticket.items` over a ticket without items finds the mapping's `items` method, which the record never held."""
    node = screening("ok or ticket.items == 'x'", inputs=("ok", "ticket"))

    assert outcome(run(node, {"ok": True, "ticket": {}})) == ("pass", None)
    assert outcome(run(node, {"ok": False, "ticket": {}})) == ("not_evaluated", "missing value for ticket.items")


@pytest.mark.parametrize(
    ("check", "record", "message"),
    [
        (
            "invoice.items.count",
            {"invoice": {"items": [{"sku": "a"}]}},
            "check could not be evaluated: read the method count(…), not a value",
        ),
        (
            "ticket.tags.append == 1",
            {"ticket": {"tags": ["a"]}},
            "check could not be evaluated: access to attribute 'append' of 'list' object is unsafe.",
        ),
    ],
    ids=["method-of-a-list", "method-the-sandbox-refuses"],
)
def test_a_path_that_reaches_a_method_of_a_value_the_record_holds_reads_the_method_as_before(check, record, message):
    node = screening(check, inputs=tuple(record))

    assert outcome(run(node, record)) == ("not_evaluated", message)


def test_an_input_named_need_is_data():
    node = screening("need > 1", inputs=("need",))

    assert outcome(run(node, {"need": 5})) == ("pass", None)
    assert outcome(run(node, {})) == ("not_evaluated", "missing value for need")


@pytest.mark.parametrize("name", ["need", "callee"])
def test_an_expression_cannot_call_the_guards_the_check_is_compiled_with(name):
    node = screening(f"{name}(order.total, 'order.total') == 5", inputs=("order",))

    assert outcome(run(node, {"order": {"total": 5}})) == (
        "not_evaluated",
        f"check could not be evaluated: '{name}' is undefined",
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("app", [{"a": 1}, {}], ids=["argument-there", "argument-missing"])
def test_a_mistyped_helper_is_an_error_under_every_policy_before_its_arguments_are_read(app, policy):
    """The name a call calls is judged before its arguments: a missing argument does not hide the typo, and a rule
    set to skip missing data does not skip it."""
    node = screening("firstpresent(app.a) == 1", policy)

    assert outcome(run(node, {"app": app})) == (
        "fail" if policy == "fail" else "not_evaluated",
        "check could not be evaluated: 'firstpresent' is undefined",
    )


@pytest.mark.parametrize(
    ("check", "record", "policy", "expected"),
    [
        (
            "code(1) == 1",
            {"code": 5},
            None,
            ("not_evaluated", "check could not be evaluated: 'int' object is not callable"),
        ),
        (
            "code(order.total) == 1",
            {"code": 5, "order": {}},
            "not_applicable",
            ("not_applicable", "does not apply: missing value for order.total"),
        ),
    ],
    ids=["called", "called-over-a-missing-argument"],
)
def test_a_name_the_record_holds_called_as_a_helper_reads_as_it_always_did(check, record, policy, expected):
    node = screening(check, policy, inputs=("code", "order"))

    assert outcome(run(node, record)) == expected


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("check", "record", "message"),
    [
        (
            "firstpresent(app.a) == 1",
            {"firstpresent": None, "app": {"a": 1}},
            "check could not be evaluated: 'NoneType' object is not callable",
        ),
        ("funcs['f'](app.a) == 1", {"app": {"a": 1}}, "check could not be evaluated: 'funcs' is undefined"),
    ],
    ids=["a-null-the-record-holds", "a-member-of-a-name-nothing-defines"],
)
def test_a_call_of_what_is_no_helper_fails_as_an_error_with_the_status_it_always_had(check, record, message, policy):
    """A null under the name is called as the record holds it, and a member of a name nothing defines is looked up,
    then called: either call fails, with the status the rule always had, now naming why the call failed rather than
    a missing value."""
    node = screening(check, policy, inputs=("app", "firstpresent", "funcs"))

    assert outcome(run(node, record)) == ("fail" if policy == "fail" else "not_evaluated", message)


@pytest.mark.parametrize(
    "check",
    [
        "app.purpose == 'purchase' or app.occupancy == 'primary'",
        "app.rate < 7 if app.kind == 'fixed' else app.margin < 3",
        "has(docs.cert) and docs.cert.zone == 'A' or first_present(app.a, app.b) > 1",
        "firstpresent(app.a) == 1 and invoice.get('rate', 0) > 0",
        "doc['Issue Date'] | default(app.date) != '' and text(app.name) is present",
    ],
)
def test_a_check_reads_the_paths_read_paths_names_in_the_same_order(check):
    node = screening(check, inputs=())
    reads = read_paths(check)

    (finding,) = run(node, {})["findings"]

    assert list(finding["evaluated"]) == reads.required + reads.optional


# --- either side of an `and` or an `or` whose truth alone counts decides ----------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("check", "app", "status"),
    [
        ("app.occupancy == 'primary' or app.purpose == 'purchase'", {"purpose": "purchase"}, "pass"),
        ("app.occupancy == 'primary' or app.purpose == 'purchase'", {"occupancy": None, "purpose": "purchase"}, "pass"),
        ("app.rate < 10 and app.amount > 0", {"amount": 0}, "fail"),
        ("app.rate < 10 and app.amount > 0", {"rate": None, "amount": 0}, "fail"),
    ],
    ids=["or-first-side-absent", "or-first-side-null", "and-first-side-absent", "and-first-side-null"],
)
def test_the_other_side_of_an_and_or_an_or_decides_where_the_first_is_missing(check, app, status, policy):
    """The purchase makes the `or` true and the zero amount the `and` false, whatever the missing side would hold, so
    the rule has its verdict under every policy."""
    node = screening(check, policy)

    assert outcome(run(node, {"app": app})) == (status, None)


@pytest.mark.parametrize(
    ("policy", "status", "prefix"),
    [
        (None, "not_evaluated", ""),
        ("fail", "fail", ""),
        ("not_applicable", "not_applicable", "does not apply: "),
    ],
)
@pytest.mark.parametrize(
    ("check", "app", "missing"),
    [
        ("app.occupancy == 'primary' or app.purpose == 'purchase'", {"purpose": "refinance"}, "app.occupancy"),
        ("app.occupancy == 'primary' or app.purpose == 'purchase'", {}, "app.occupancy"),
        ("app.rate < 10 and app.amount > 0", {"amount": 5}, "app.rate"),
        ("app.rate < 10 and app.amount > 0", {}, "app.rate"),
        ("app.a or text(app.b)", {"b": "  "}, "app.a"),
        ("app.a and text(app.b)", {"b": "  "}, "app.a"),
    ],
    ids=[
        "or-other-side-false",
        "or-both-missing",
        "and-other-side-true",
        "and-both-missing",
        "or-both-missing-the-other-a-blank",
        "and-both-missing-the-other-a-blank",
    ],
)
def test_a_missing_value_the_other_side_cannot_decide_without_holds_the_rule_naming_the_left_one_first(
    check, app, missing, policy, status, prefix
):
    """A refinance leaves the `or` to the occupancy, and a positive amount the `and` to the rate: the missing value
    holds the rule under its policy. Where both sides are missing, the reason names the left one, in reading order,
    a right side that is a blank `text()` made, missing only once its truth is tested, as well."""
    node = screening(check, policy)

    assert outcome(run(node, {"app": app})) == (status, f"{prefix}missing value for {missing}")


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"c": 3}, ("pass", None)),
        ({"b": 2}, ("pass", None)),
        ({"a": 0, "c": 3}, ("pass", None)),
        ({"b": 0, "c": 0}, ("not_evaluated", "missing value for app.a")),
        ({"a": 0, "c": 0}, ("not_evaluated", "missing value for app.b")),
        ({"a": 0, "b": 0}, ("not_evaluated", "missing value for app.c")),
        ({"c": 0}, ("not_evaluated", "missing value for app.a")),
        ({}, ("not_evaluated", "missing value for app.a")),
    ],
    ids=[
        "last-decides-the-first-two-missing",
        "middle-decides-the-others-missing",
        "last-decides-the-middle-missing",
        "first-missing-the-others-false",
        "middle-missing-the-others-false",
        "last-missing-the-others-false",
        "first-two-missing-the-last-false",
        "all-missing",
    ],
)
def test_any_side_of_a_three_way_or_decides_and_the_first_missing_value_stands_where_none_does(app, expected):
    node = screening("app.a == 1 or app.b == 2 or app.c == 3")

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"b": 2, "c": 3}, ("pass", None)),
        ({"b": 2, "c": 0}, ("fail", None)),
        ({"b": 0, "c": 0}, ("fail", None)),
        ({"b": 0, "c": 3}, ("not_evaluated", "missing value for app.a")),
        ({"a": 1}, ("not_evaluated", "missing value for app.c")),
        ({}, ("not_evaluated", "missing value for app.a")),
    ],
    ids=[
        "the-or-decided-by-its-second-side",
        "the-and-decided-false-after-the-or",
        "the-and-decided-false-by-its-second-side",
        "neither-decides",
        "the-or-decided-by-its-first-side",
        "all-missing",
    ],
)
def test_an_and_over_an_or_is_decided_by_whichever_side_is_there_to_decide_it(app, expected):
    node = screening("(app.a == 1 or app.b == 2) and app.c == 3")

    assert outcome(run(node, {"app": app})) == expected


def test_either_side_of_an_or_in_applies_when_decides_where_the_other_is_missing():
    node = screening("app.points <= 2", applies_when="app.kind == 'fixed' or app.term > 10")

    assert outcome(run(node, {"app": {"term": 30, "points": 1}})) == ("pass", None)
    assert outcome(run(node, {"app": {"term": 30, "points": 5}})) == ("fail", None)
    assert outcome(run(node, {"app": {"term": 5, "points": 1}})) == ("not_evaluated", "missing value for app.kind")


def test_either_side_of_an_and_in_applies_when_decides_where_the_other_is_missing():
    node = screening("app.points <= 2", applies_when="app.kind == 'fixed' and app.term > 10")

    assert outcome(run(node, {"app": {"term": 5}})) == (
        "not_applicable",
        "does not apply: app.kind == 'fixed' and app.term > 10",
    )
    assert outcome(run(node, {"app": {"term": 30}})) == ("not_evaluated", "missing value for app.kind")


@pytest.mark.parametrize(("b", "expected"), [(1, ("pass", None)), (2, ("not_evaluated", "missing value for app.a"))])
def test_a_blank_a_helper_turned_missing_gives_way_to_the_other_side_as_a_missing_value_does(b, expected):
    node = screening("text(app.a) == 'x' or app.b == 1")

    assert outcome(run(node, {"app": {"a": "  ", "b": b}})) == expected


@pytest.mark.parametrize(
    ("policy", "status", "prefix"),
    [(None, "not_evaluated", ""), ("fail", "fail", ""), ("not_applicable", "not_applicable", "does not apply: ")],
)
@pytest.mark.parametrize(
    ("check", "app", "missing"),
    [
        ("text(app.a) == 'x' if app.kind == 'y' else text(app.b) == 'z'", {"kind": "n", "a": " ", "b": ""}, "app.b"),
        ("(app.kind == 'y' and text(app.a) == 'x') or number(app.b) > 1", {"kind": "n", "a": " ", "b": ""}, "app.b"),
        ("days_between(app.opened, app.closed) > 30", {"opened": "", "closed": " "}, "app.closed"),
        ("days_between(app.opened, date(app.closed)) > 30", {"opened": "", "closed": " "}, "app.closed"),
        (
            "(app.kind == 'y' and text(app.a) == 'x') or days_between(app.opened, app.closed) > 30",
            {"kind": "n", "a": " ", "opened": "2026-08-07", "closed": ""},
            "app.closed",
        ),
    ],
    ids=[
        "if-branch-not-taken",
        "and-side-not-read",
        "days-between-reads-end-first",
        "days-between-over-date",
        "days-between-past-a-side-not-read",
    ],
)
def test_the_reason_names_the_blank_the_evaluation_stopped_at(check, app, missing, policy, status, prefix):
    """A helper that turns a blank it is handed into a missing value, `text()` of blank text say, stops the check
    naming the path it was handed, so the reason names the blank the evaluation stopped at, never one it did not reach
    in the branch of an `if` not taken or on the side of an `and` its first side decided. `days_between` reads `end`
    first, so of two blank dates it stops at `end`, whether it is handed the text or what `date()` made of it."""
    node = screening(check, policy)

    assert outcome(run(node, {"app": app})) == (status, f"{prefix}missing value for {missing}")


@pytest.mark.parametrize(
    ("policy", "status", "prefix"),
    [(None, "not_evaluated", ""), ("fail", "fail", ""), ("not_applicable", "not_applicable", "does not apply: ")],
)
@pytest.mark.parametrize(
    ("check", "app", "missing"),
    [
        ("app.b == '' and first_present(text(app.c), app.a) == 'x'", {"b": "", "c": " ", "a": ""}, "app.c"),
        ("app.b == '' and number(text(app.a)) > 1", {"b": "", "a": " "}, "app.a"),
        ("app.b == '' and text(app.x | trim) == 'y'", {"b": "", "x": "  "}, "app.x"),
        ("app.b == '' and days_between(app.o | trim, app.c) > 1", {"b": "", "o": "  ", "c": "2026-08-07"}, "app.o"),
        ("app.b == '' and days_between(start=app.o, end=app.c) > 1", {"b": "", "o": " ", "c": "2026-08-07"}, "app.o"),
        ("app.b == '' and text(app.x) == 'y'", {"b": ""}, "app.x"),
        ("app.b == '' and text(app.x) == 'y'", {"b": "", "x": " "}, "app.x"),
        (
            "app.b == '' and text(app.nickname or app.name) == 'Ann'",
            {"b": "", "nickname": "", "name": " "},
            "app.nickname",
        ),
        ("app.b == '' and text(app.x if app.k else app.y) == 'z'", {"b": "", "x": " ", "y": "", "k": True}, "app.x"),
        ("app.b == '' and number(app.x.replace('$', '')) > 1", {"b": "", "x": "  "}, "app.x"),
        ("app.b == '' and text(app.x | default(app.y)) == 'z'", {"b": "", "y": " "}, "app.x"),
        ("app.b == '' and text(first_present(app.x, app.y)) == 'z'", {"b": "", "x": " ", "y": ""}, "app.x"),
        (
            "app.b == '' and days_between(first_present(app.x, app.y), app.c) > 1",
            {"b": "", "x": " ", "y": "", "c": "2026-08-07"},
            "app.x",
        ),
        (
            "app.b == '' and first_present(first_present(app.x, app.y), app.z) == 'w'",
            {"b": "", "x": " ", "y": "", "z": "  "},
            "app.x",
        ),
        ("app.b == '' and number(first_present(app.x, app.y | trim)) > 1", {"b": "", "x": " ", "y": "  "}, "app.x"),
        ("app.b == '' and text(app.x | default('')) == 'z'", {"b": ""}, "app.x"),
        ("app.b == '' and text(app.get('F')) == 'z'", {"b": ""}, "app.F"),
        ("app.b == '' and text(app.get('F', '')) == 'z'", {"b": "", "F": "  "}, "app.F"),
        ("app.b == '' and text(app.get('Issue Date', '')) == 'z'", {"b": ""}, "app['Issue Date']"),
        ("app.b == '' and first_present(app.x, '') == 'z'", {"b": ""}, "app.x"),
        ("app.b == '' and number(first_present(app.x, '  ' | trim)) > 1", {"b": "", "x": " "}, "app.x"),
    ],
    ids=[
        "first-present-over-a-helper",
        "number-over-text",
        "a-filter-over-a-blank-path",
        "days-between-over-a-filter",
        "days-between-by-keyword",
        "text-of-an-absent-path",
        "text-of-a-blank-path",
        "an-or-of-two-blanks",
        "an-if-of-two-blanks",
        "a-method-of-a-blank",
        "a-default-that-is-blank",
        "a-helper-over-first-present",
        "days-between-over-first-present",
        "first-present-over-first-present",
        "number-over-first-present-over-a-filter",
        "a-literal-default",
        "a-get-by-a-constant-key",
        "a-get-with-a-literal-default-of-a-blank",
        "a-get-by-a-key-no-name-writes",
        "a-literal-among-fallbacks",
        "a-literal-filtered-among-fallbacks",
    ],
)
def test_a_helpers_blank_names_the_path_it_came_from_through_filters_and_helpers(
    check, app, missing, policy, status, prefix
):
    """A blank a helper makes is data the record lacks where every value it could have come from is blank in the
    record: the path itself, a filter's value and a `default`'s, both sides of an `or` or an `and`, both branches of an
    `if`, the value a method is called on, and every value a helper inside it passes on. It names the first of them, and
    a rule set to skip it does; a literal among them, `default('')` say, adds none, and `x.get('F')` reads `x.F`. The
    blank `app.b` compared before it was never where the check stopped."""
    node = screening(check, policy)

    assert outcome(run(node, {"app": app})) == (status, f"{prefix}missing value for {missing}")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("app", "expected"), [({}, ("fail", None)), ({"F": "x"}, ("pass", None))], ids=["absent", "there"]
)
def test_a_get_a_check_compares_reads_and_decides_as_it_always_did(app, expected, policy):
    """`app.get('F')` is a call on `app`, which the check needs, and hands back None for a key the record lacks: the
    comparison decides on it, as Jinja's does, however `app.F` names a blank a helper makes of it."""
    node = screening("app.get('F') == 'x'", policy)

    assert (read_paths("app.get('F') == 'x'").required, outcome(run(node, {"app": app}))) == (["app"], expected)


NO_TEXT = "missing value: text() found no text"
NO_NUMBER = "missing value: number() found no number"
NOTHING = "missing value: first_present() found nothing present"


@pytest.mark.parametrize(
    ("policy", "status", "suffix"),
    [
        (None, "not_evaluated", ""),
        ("fail", "fail", ""),
        ("not_applicable", "not_evaluated", " (not skipped: no field of the record accounts for it)"),
    ],
)
@pytest.mark.parametrize(
    ("check", "app", "hint"),
    [
        (
            "app.c == '' and text(limits[app.k]) == 'x'",
            {"c": "", "k": "zz"},
            f"{NO_TEXT} ('dict object' has no attribute 'zz')",
        ),
        (
            "app.b == '' and text(app.x | replace('x', '')) == 'y'",
            {"b": "", "x": "x"},
            "missing value: text() found no text",
        ),
        (
            "app.b == '' and first_present(app.a, limits[app.k]) == 'x'",
            {"b": "", "a": "", "k": "zz"},
            "missing value: first_present() found nothing present",
        ),
        (
            "app.b == '' and days_between(app.o | replace('x', ''), app.c) > 1",
            {"b": "", "o": "x", "c": "2026-08-07"},
            "missing value: days_between() found no date",
        ),
        ("app.b == '' and text(app.nickname or app.name) == 'Ann'", {"b": "", "nickname": " ", "name": "Ann"}, NO_TEXT),
        ("app.b == '' and text(app.x if app.k else app.y) == 'z'", {"b": "", "x": "x", "y": " ", "k": False}, NO_TEXT),
        ("app.b == '' and number(app.x.replace('$', '')) > 1", {"b": "", "x": "$"}, NO_NUMBER),
        (
            "app.b == '' and text(app.x | default(limits[app.k])) == 'z'",
            {"b": "", "k": "zz"},
            f"{NO_TEXT} ('dict object' has no attribute 'zz')",
        ),
        ("app.b == '' and text(first_present(app.x, limits[app.k])) == 'z'", {"b": "", "x": " ", "k": "zz"}, NOTHING),
        (
            "app.b == '' and days_between(first_present(app.x, limits[app.k]), app.c) > 1",
            {"b": "", "x": " ", "k": "zz", "c": "2026-08-07"},
            NOTHING,
        ),
        (
            "app.b == '' and first_present(first_present(app.x, limits[app.k]), app.y) == 'z'",
            {"b": "", "x": " ", "y": "", "k": "zz"},
            NOTHING,
        ),
        ("app.b == '' and text(limits.get(app.k)) == 'z'", {"b": "", "k": "zz"}, NO_TEXT),
        ("app.b == '' and text('') == 'z'", {"b": ""}, NO_TEXT),
        (
            "text(app.name) == 'x' or text(limits[app.k]) == 'y'",
            {"name": "Ann", "k": "zz"},
            f"{NO_TEXT} ('dict object' has no attribute 'zz')",
        ),
    ],
    ids=[
        "a-lookup-that-found-nothing",
        "a-filter-over-a-value-there",
        "a-lookup-among-fallbacks",
        "days-between",
        "an-or-with-a-side-there",
        "an-if-with-a-branch-there",
        "a-method-of-a-value-there",
        "a-default-that-is-a-lookup",
        "a-helper-over-a-lookup-among-fallbacks",
        "days-between-over-a-lookup-among-fallbacks",
        "first-present-over-a-lookup-among-fallbacks",
        "a-get-by-a-key-read-at-run-time",
        "a-literal-alone",
        "the-second-of-two-helpers",
    ],
)
def test_a_blank_no_path_accounts_for_is_never_skipped_nor_named_after_another_blank(
    check, app, hint, policy, status, suffix
):
    """A blank a helper made where any value it could have come from is not blank in the record, a lookup that found
    nothing or a value that is there, or that comes from a literal alone, is no data the record lacks: the reason is
    the helper's own, naming no value, with what an undefined value it was handed says of itself, and a rule set to
    skip missing data is held. Naming the blank `app.b` or `app.c` read before it would skip a gap in the table as if
    the record lacked a value."""
    node = screening(check, policy, inputs=("app", "limits"))

    output = run(node, {"app": app, "limits": {"x": 2}})

    assert outcome(output) == (status, f"{hint}{suffix}")


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("check", "record", "error"),
    [
        (
            "app.a == 1 or app.b > 1",
            {"app": {"b": "x"}},
            "'>' not supported between instances of 'str' and 'int'",
        ),
        (
            "limits[app.program] > 5 or app.b == 1",
            {"app": {"program": "jumbo", "b": 1}, "limits": {"standard": 1}},
            "'dict object' has no attribute 'jumbo'",
        ),
        (
            "app.b == 1 or limits[app.program] > 5",
            {"app": {"program": "jumbo"}, "limits": {"standard": 1}},
            "'dict object' has no attribute 'jumbo'",
        ),
        ("number(app.a) > 1 or app.b == 1", {"app": {"a": "TBD", "b": 1}}, "not a number: 'TBD'"),
        ("app.b == 1 or number(app.a) > 1", {"app": {"a": "TBD"}}, "not a number: 'TBD'"),
        ("app.b == 1 or firstpresent(app.a) == 1", {"app": {"a": 1}}, "'firstpresent' is undefined"),
        (
            "app.b == 1 or ticket.tags.append == 1",
            {"app": {}, "ticket": {"tags": ["a"]}},
            "access to attribute 'append' of 'list' object is unsafe.",
        ),
        (
            "limits[app.p] > 5 and app.b == 1",
            {"app": {"p": "jumbo", "b": 0}, "limits": {"standard": 1}},
            "'dict object' has no attribute 'jumbo'",
        ),
        ("number(app.a) > 5 and app.b == 1", {"app": {"a": "TBD", "b": 0}}, "not a number: 'TBD'"),
        ("firstpresent(app.a) == 1 and app.b == 1", {"app": {"a": 1, "b": 0}}, "'firstpresent' is undefined"),
    ],
    ids=[
        "a-type-error-beside-a-missing-value",
        "a-lookup-that-found-nothing-before-a-side-that-decides",
        "a-lookup-that-found-nothing-beside-a-missing-value",
        "an-unreadable-number-before-a-side-that-decides",
        "an-unreadable-number-beside-a-missing-value",
        "a-mistyped-helper-beside-a-missing-value",
        "an-unsafe-attribute-beside-a-missing-value",
        "a-lookup-that-found-nothing-before-a-false-side-of-an-and",
        "an-unreadable-number-before-a-false-side-of-an-and",
        "a-mistyped-helper-before-a-false-side-of-an-and",
    ],
)
def test_only_a_missing_value_gives_way_to_the_other_side_and_an_error_on_either_side_is_an_error(
    check, record, error, policy
):
    """A side that fails, rather than stops at a missing value, fails the check with the status an error always had,
    whatever the other side holds, in an `and` as in an `or`: a lookup that found nothing, a value nobody could read, a
    name nothing defines and an attribute the sandbox refuses are defects of the record or the check, not data the
    record lacks, so a false side of an `and` does not stand in for them."""
    node = screening(check, policy, inputs=("app", "limits", "ticket"))

    assert outcome(run(node, record)) == (
        "fail" if policy == "fail" else "not_evaluated",
        f"check could not be evaluated: {error}",
    )


MISSPELLED_METHOD = ("not_evaluated", "check could not be evaluated: 'str object' has no attribute 'startwith'")


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"name": "Ann"}, MISSPELLED_METHOD),
        ({"name": "Ann", "age": 30}, MISSPELLED_METHOD),
        ({"name": "Ann", "age": 10}, ("fail", None)),
    ],
    ids=["without-the-age", "with-an-age-of-18-or-more", "with-an-age-that-decides-first"],
)
def test_an_error_on_the_other_side_of_an_and_a_missing_value_gives_way_to_surfaces_with_or_without_the_value(
    app, expected
):
    """A missing age gives way to the other side of the `and`, whose misspelled method fails: the error is reported
    without the age as with an age of 18 or more, and a rule set to skip missing data is not skipped. An age under 18
    decides the `and` before the method is reached."""
    node = screening("app.age >= 18 and text(app.name).startwith('A')", "not_applicable")

    assert outcome(run(node, {"app": app})) == expected


def test_an_error_after_a_missing_value_outside_an_and_or_an_or_surfaces_only_where_the_value_is_there():
    """A comparison reads its left side first: a missing limit stops it before the division by zero on its right, so a
    rule set to skip missing data is skipped on the record without the limit, and the error surfaces where it is."""
    node = screening(
        "appraisal.max_ltv >= loan.amount / appraisal.value", "not_applicable", inputs=("loan", "appraisal")
    )

    assert outcome(run(node, {"loan": {"amount": 300000}, "appraisal": {"value": 0}})) == (
        "not_applicable",
        "does not apply: missing value for appraisal.max_ltv",
    )
    assert outcome(run(node, {"loan": {"amount": 300000}, "appraisal": {"value": 0, "max_ltv": 0.8}})) == (
        "not_evaluated",
        "check could not be evaluated: division by zero",
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_an_error_behind_a_missing_value_the_other_side_decides_past_is_never_reached(policy):
    """The missing `app.b` gives way to `app.a == 1`, which decides: the division by the zero `app.q` behind it is never
    reached, so the rule passes, as a rule skipped for `app.b` would not see the error either. Where `app.b` is there,
    the division fails."""
    node = screening("app.b / app.q > 1 or app.a == 1", policy)

    assert outcome(run(node, {"app": {"q": 0, "a": 1}})) == ("pass", None)
    assert outcome(run(node, {"app": {"b": 5, "q": 0, "a": 1}})) == (
        "fail" if policy == "fail" else "not_evaluated",
        "check could not be evaluated: division by zero",
    )


@pytest.mark.parametrize(
    ("policy", "held", "held_elsewhere"),
    [
        (None, ("not_evaluated", "missing value for dco.a"), ("not_evaluated", "missing value for doc.x")),
        ("fail", ("fail", "missing value for dco.a"), ("fail", "missing value for doc.x")),
        (
            "not_applicable",
            ("not_evaluated", "missing value for dco.a (not skipped: dco is not an input or a derived value)"),
            ("not_evaluated", "missing value for doc.x (not skipped: dco is not an input or a derived value)"),
        ),
    ],
)
def test_a_value_missing_under_a_name_the_node_does_not_declare_gives_way_where_the_other_side_decides(
    policy, held, held_elsewhere
):
    """A typo is no data the record lacks, so a rule set to skip is not skipped for it wherever the check has it: where
    the rule stops at the typo, and where it stops at another missing value, the typo in a branch the evaluation never
    took. It gives way to the other side of an `or` as any missing value does, though, and where that side decides,
    the rule has its verdict."""
    node = screening("dco.a == 1 or doc.b == 1", policy, inputs=("doc",))
    elsewhere = screening("(doc.x if doc.k else dco.a) == 1", policy, inputs=("doc",))

    assert outcome(run(node, {"doc": {"b": 1}})) == ("pass", None)
    assert outcome(run(node, {"doc": {"b": 0}})) == held
    assert outcome(run(elsewhere, {"doc": {"k": True}})) == held_elsewhere


@pytest.mark.parametrize("policy", POLICIES)
def test_a_derived_value_nobody_could_compute_speaks_over_the_missing_value_from_a_branch_not_taken(policy):
    """The check stops at the missing `app.x`, and the branch that reads `ltv` is never taken; yet the reads are scanned
    for a value nobody could read wherever the check has them, so the rule reports the division by zero behind `ltv`,
    under every policy: stricter than the evaluation, never looser."""
    node = screening("(app.x if app.k else ltv) == 1", policy, derived=(("ltv", "app.p / app.q"),))

    assert outcome(run(node, {"app": {"k": True, "p": 1, "q": 0}})) == (
        "fail" if policy == "fail" else "not_evaluated",
        "check could not be evaluated: division by zero",
    )
    assert outcome(run(node, {"app": {"k": True, "p": 1, "q": 1}}))[1].endswith("missing value for app.x")


@pytest.mark.parametrize("check", ["not app.a", "(1 if app.a else 2) == 2"])
def test_a_not_or_an_if_still_stops_at_the_missing_value_it_reads(check):
    node = screening(check)

    assert outcome(run(node, {"app": {}})) == ("not_evaluated", "missing value for app.a")


@pytest.mark.parametrize(
    ("check", "app", "expected"),
    [
        ("(app.a or 'fallback') == 'fallback'", {"a": "x"}, ("fail", None)),
        ("(app.a or 'fallback') == 'fallback'", {"a": ""}, ("pass", None)),
        ("(app.a or 'fallback') == 'fallback'", {"a": None}, ("not_evaluated", "missing value for app.a")),
        ("(app.a or 'fallback') == 'fallback'", {}, ("not_evaluated", "missing value for app.a")),
        ("(app.nickname or app.name) == 'Ann'", {"nickname": "Annie", "name": "Ann"}, ("fail", None)),
        ("(app.nickname or app.name) == 'Ann'", {"nickname": "", "name": "Ann"}, ("pass", None)),
        ("(app.nickname or app.name) == 'Ann'", {"name": "Ann"}, ("not_evaluated", "missing value for app.nickname")),
        ("(app.nickname or app.name) == 'Ann'", {"name": ""}, ("not_evaluated", "missing value for app.nickname")),
        ("(app.a and app.b) == ''", {"b": ""}, ("not_evaluated", "missing value for app.a")),
        ("(app.a and app.b) == ''", {"b": "x"}, ("not_evaluated", "missing value for app.a")),
        ("first_present(app.nickname, app.name) == 'Ann'", {"name": "Ann"}, ("pass", None)),
    ],
    ids=[
        "or-first-side-true",
        "or-first-side-false",
        "or-first-side-null",
        "or-first-side-absent",
        "or-first-side-true-over-a-second-that-is-there",
        "or-first-side-false-over-a-second-that-is-there",
        "or-first-side-absent-the-second-true",
        "or-first-side-absent-the-second-false",
        "and-first-side-absent-the-second-false",
        "and-first-side-absent-the-second-true",
        "first-present-falls-back-past-a-missing-value",
    ],
)
def test_an_and_or_an_or_whose_value_the_check_compares_is_pythons_and_stops_at_a_missing_side(check, app, expected):
    """Compared, an `and` or an `or` is Python's: `a or 'fallback'` is `a` where `a` is true and the fallback where `a`
    is there and false. A missing side it reads holds the rule, whatever the other side holds, since a value in its
    place could change the verdict: a nickname would decide `(app.nickname or app.name) == 'Ann'`. `first_present`
    falls back past a missing value, as its author means."""
    node = screening(check)

    assert outcome(run(node, {"app": app})) == expected


@pytest.mark.parametrize(
    ("policy", "status", "prefix"),
    [
        (None, "not_evaluated", ""),
        ("fail", "fail", ""),
        ("not_applicable", "not_applicable", "does not apply: "),
    ],
)
@pytest.mark.parametrize(
    ("check", "app", "missing"),
    [
        ("(app.a or app.b) == 1", {"b": 1}, "app.a"),
        ("(app.a or app.b) == 1", {"b": "  "}, "app.a"),
        ("(app.a and app.b) == ''", {"b": ""}, "app.a"),
        ("(app.fee or 100) > 50", {}, "app.fee"),
        ("(app.fee or 0) < 50", {}, "app.fee"),
        ("(app.a or app.b) > 1", {"b": "x"}, "app.a"),
        ("(app.a or app.b) - 1 > 0", {"b": 5}, "app.a"),
        ("(app.a or app.b) | length > 1", {"b": "xy"}, "app.a"),
        ("max(app.a or 2, 1) > 1", {}, "app.a"),
        ("(app.a or app.b) is number", {"b": 3}, "app.a"),
        ("(app.x if app.k else (app.a or app.b)) == 1", {"k": False, "b": 1}, "app.a"),
    ],
    ids=[
        "compared-where-the-other-side-would-pass",
        "compared-where-the-other-side-would-fail",
        "an-and-compared",
        "a-fallback-that-would-pass",
        "a-fallback-that-would-not",
        "compared-where-the-other-side-would-raise",
        "computed-with",
        "filtered",
        "handed-to-a-helper",
        "tested",
        "the-branch-of-an-if-whose-value-is-compared",
    ],
)
def test_a_missing_side_of_an_and_or_an_or_whose_value_the_check_uses_holds_the_rule_under_its_policy(
    check, app, missing, policy, status, prefix
):
    """Wherever the check uses the value of an `and` or an `or`, comparing it, computing with it, filtering it, testing
    it, handing it to a helper or taking it as a branch of an `if` whose value it uses, the other side does not stand in
    for a missing one: `(app.fee or 100) > 50` is held without the fee as `(app.fee or 0) < 50` is, whatever the
    fallback would make of the verdict, and a type error the other side would raise is never reached. The missing value
    holds the rule under its policy, as it did before either side could decide."""
    node = screening(check, policy)

    assert outcome(run(node, {"app": app})) == (status, f"{prefix}missing value for {missing}")


@pytest.mark.parametrize(
    ("check", "app", "expected"),
    [
        ("not (app.a or app.b)", {"b": 1}, ("fail", None)),
        ("not (app.a or app.b)", {"b": 0}, ("not_evaluated", "missing value for app.a")),
        ("(1 if app.a or app.b else 2) == 1", {"b": 1}, ("pass", None)),
        ("(1 if app.a or app.b else 2) == 1", {"b": 0}, ("not_evaluated", "missing value for app.a")),
        ("(1 if (app.a or app.b) and app.c else 2) == 1", {"b": 1, "c": 1}, ("pass", None)),
        ("(not (app.a and app.b)) == true", {"b": 0}, ("pass", None)),
        ("app.a or app.b if app.k else app.c", {"k": True, "b": 1}, ("pass", None)),
        ("app.a or app.b if app.k else app.c", {"k": True, "b": 0}, ("not_evaluated", "missing value for app.a")),
        ("not (app.x if app.k else (app.a or app.b))", {"k": False, "b": 1}, ("fail", None)),
        ("(app.a or app.b) == 1 or app.c == 1", {"b": 1, "c": 1}, ("pass", None)),
        ("(app.a or app.b) == 1 or app.c == 1", {"b": 1, "c": 0}, ("not_evaluated", "missing value for app.a")),
    ],
    ids=[
        "under-not",
        "under-not-the-other-side-false",
        "the-test-of-an-if-whose-value-is-compared",
        "the-test-of-an-if-the-other-side-false",
        "inside-an-and-that-is-the-test-of-an-if",
        "under-a-not-whose-value-is-compared",
        "a-branch-of-an-if-that-is-the-check",
        "a-branch-of-an-if-that-is-the-check-the-other-side-false",
        "a-branch-of-an-if-under-not",
        "a-compared-or-as-a-side-the-other-side-decides",
        "a-compared-or-as-a-side-the-other-side-cannot-decide",
    ],
)
def test_an_and_or_an_or_whose_truth_alone_the_check_uses_is_decided_by_either_side(check, app, expected):
    """Under `not`, as the test of an `if`, as a branch of an `if` whose truth alone the check uses, and inside another
    `and` or `or` used so, only the truth of an `and` or an `or` counts, and either side decides it where the other is
    missing, wherever the check then uses what `not` or the `if` makes of it. A compared `or` that stops at a missing
    side is a side that stops at a missing value, and gives way to the other side of the `or` that is the check, which
    decides where it can."""
    node = screening(check)

    assert outcome(run(node, {"app": app})) == expected


def test_an_or_whose_value_applies_when_compares_holds_the_rule_at_a_missing_side():
    """`applies_when` uses an `and` or an `or` as a check does: compared, a missing side holds the rule."""
    node = screening("app.points <= 2", applies_when="(app.kind or app.type) == 'fixed'")

    assert outcome(run(node, {"app": {"type": "fixed", "points": 1}})) == (
        "not_evaluated",
        "missing value for app.kind",
    )
    assert outcome(run(node, {"app": {"kind": "", "type": "fixed", "points": 1}})) == ("pass", None)


@pytest.mark.parametrize(
    ("check", "items", "expected"),
    [
        ("app.exempt or app.items | select('odd')", [2, 4], ("not_evaluated", "missing value for app.exempt")),
        ("app.exempt or app.items | select('odd')", [1, 2], ("pass", None)),
        ("app.exempt and app.items | select('odd')", [2, 4], ("fail", None)),
        ("app.exempt and app.items | select('odd')", [1, 2], ("not_evaluated", "missing value for app.exempt")),
    ],
    ids=["or-yields-nothing", "or-yields-an-item", "and-yields-nothing", "and-yields-an-item"],
)
def test_a_lazy_side_that_stands_in_for_a_missing_one_decides_by_the_items_it_yields(check, items, expected):
    """A filter that selects yields its items lazily, and a generator is true whatever it would yield: standing in for
    the missing exemption, it is judged by its items, as the check's result is, so an `or` that selects nothing does
    not fail the rule on a value nobody gave, and an `and` that does is false."""
    node = screening(check)

    assert outcome(run(node, {"app": {"items": items}})) == expected


def test_a_lazy_side_that_stands_in_for_a_missing_one_in_applies_when_does_not_skip_the_rule_on_nothing():
    """`applies_when` is judged the same way: with the flag missing and no document of the kind, the rule is held for
    the flag, where a generator taken for true would have skipped it and let the record pass."""
    node = screening(
        "loan.addendum_signed",
        applies_when="loan.is_jumbo or loan.docs | selectattr('kind', 'equalto', 'jumbo')",
        inputs=("loan",),
    )

    held = run(node, {"loan": {"docs": [{"kind": "w2"}], "addendum_signed": False}})
    applies = run(node, {"loan": {"docs": [{"kind": "jumbo"}], "addendum_signed": False}})
    skipped = run(node, {"loan": {"docs": [{"kind": "w2"}], "addendum_signed": False, "is_jumbo": False}})

    assert (held["status"], outcome(held)) == ("not_evaluated", ("not_evaluated", "missing value for loan.is_jumbo"))
    assert (applies["status"], outcome(applies)) == ("fail", ("fail", None))
    assert (skipped["status"], outcome(skipped)[0]) == ("pass", "not_applicable")


def test_a_lazy_first_side_that_is_there_keeps_the_truth_python_gives_it_as_it_always_did():
    """Only a side that stands in for a missing one is judged by its items. A first side that is there keeps Python's
    truth, as it always did: the selection is true whatever it yields, so it is the `or`, and the check, judged by its
    items, fails though the exemption holds."""
    node = screening("app.items | select('odd') or app.exempt")

    assert outcome(run(node, {"app": {"items": [2, 4], "exempt": True}})) == ("fail", None)


@pytest.mark.parametrize(("check", "status"), [("false and app.x | lowr", "fail"), ("true or app.x | lowr", "pass")])
@pytest.mark.parametrize("app", [{"x": "A"}, {}], ids=["there", "missing"])
def test_a_constant_side_that_decides_folds_the_check_before_the_rest_of_it_is_looked_at(check, status, app):
    """Jinja folds an `and` or an `or` its constant first side decides into that constant when the check is built, and
    never looks at the rest: a filter no sandbox has there is never looked up, so the check builds, as it always did."""
    node = screening(check)

    assert outcome(run(node, {"app": app})) == (status, None)


def test_a_filter_no_sandbox_has_beside_an_and_or_an_or_is_still_refused_when_the_check_is_built():
    with pytest.raises(ValueError, match="the check is not a valid expression: No filter named 'lowr'"):
        screening("app.x | lowr and false")


@pytest.mark.parametrize(
    ("app", "expected"),
    [
        ({"a195": 195}, ("pass", None)),
        ({"a0": 0}, ("pass", None)),
        ({}, ("not_evaluated", "missing value for app.a0")),
    ],
    ids=["the-last-side-decides", "the-first-side-decides", "every-side-missing"],
)
def test_the_longest_or_chain_a_check_builds_still_builds_and_any_side_of_it_decides(app, expected):
    """Each `or` of a chain that is the check is a call with a function for each side, so the chain nests as deep as it
    is long, and `need()` nests its first read one level deeper: 196 terms of `app.aN == N` is the longest such chain
    Python compiles, a term shorter than Jinja alone compiles. It builds and runs, so a check that nested any deeper
    would fail here first."""
    node = screening(" or ".join(f"app.a{i} == {i}" for i in range(196)))

    assert outcome(run(node, {"app": app})) == expected


def test_a_check_compiles_with_the_vocabulary_the_rest_of_the_node_uses():
    """A check and an `applies_when` compile in a sandbox of their own, for `and` and `or`, with the same helpers,
    filters and tests as the derived values and the messages, which `workflow validate` mirrors."""
    checks, others = rules_module._CHECK_ENVIRONMENT, rules_module._ENVIRONMENT

    assert checks is not others
    assert (set(checks.globals), set(checks.filters), set(checks.tests)) == (
        set(others.globals),
        set(others.filters),
        set(others.tests),
    )


@contextmanager
def warnings_logged() -> Iterator[list[str]]:
    """The warnings the SDK logs inside the block, read from its logger whether or not pytest captures logs."""
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno == logging.WARNING:
                messages.append(record.getMessage())

    handler = Collect()
    logger.addHandler(handler)
    try:
        yield messages
    finally:
        logger.removeHandler(handler)


def compile_and_or_with_jinjas_own_visitors(monkeypatch) -> None:
    """Has Jinja's code generator compile `and` and `or` with its own visitors, as Python's, as a jinja2 release whose
    internals the Rules node no longer matches might."""
    dispatch = NodeVisitor.get_visitor

    def own_and_or(self, node):
        if isinstance(self, CodeGenerator) and isinstance(node, (nodes.And, nodes.Or)):
            return getattr(CodeGenerator, f"visit_{type(node).__name__}").__get__(self)
        return dispatch(self, node)

    monkeypatch.setattr(NodeVisitor, "get_visitor", own_and_or)


def test_the_module_warns_once_when_imported_under_a_jinja2_that_compiles_a_checks_and_and_or_with_its_own_visitors(
    monkeypatch,
):
    """When it is imported, the module compiles a check and makes sure either side of its `and` and its `or` decides
    where the other is missing. Under a jinja2 release whose code generator compiled them with its own visitors, as
    Python's `and` and `or`, a check reads them as main does: a missing value on either side stops the rule where the
    other side should decide it. The module imports all the same, and warns once, naming the release."""
    with warnings_logged() as warnings:
        assert rules_module._check_and_or_decide()
    assert warnings == []
    compile_and_or_with_jinjas_own_visitors(monkeypatch)
    spec = importlib.util.spec_from_file_location("rules_under_another_jinja2", rules_module.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    with warnings_logged() as warnings:
        spec.loader.exec_module(module)

    assert len(warnings) == 1
    assert warnings[0].startswith(f"jinja2 {jinja2.__version__} does not compile")
    node = module.Rules(name="n", input_fields=[NamedField(name="app")], rules=[Rule(id="r", check="app.a or app.b")])
    assert outcome(run(node, {"app": {"b": True}})) == ("not_evaluated", "missing value for app.a")


@pytest.mark.parametrize(
    ("release", "named"),
    [("3.1.99", "jinja2 3.1.99"), (None, "jinja2 (version unknown)")],
    ids=["the-release-the-package-names", "no-release-named-at-all"],
)
def test_the_warning_names_the_jinja2_release_where_the_package_has_no_metadata(monkeypatch, release, named):
    """A bundled application may ship jinja2 without its metadata, where looking the release up there raises. The
    warning reads the release from the package itself, or says it is unknown, so the check that must never stop the
    import still only warns."""

    def no_metadata(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", no_metadata)
    if release is None:
        monkeypatch.delattr(jinja2, "__version__")
    else:
        monkeypatch.setattr(jinja2, "__version__", release)
    compile_and_or_with_jinjas_own_visitors(monkeypatch)

    with warnings_logged() as warnings:
        assert not rules_module._check_and_or_decide()

    assert len(warnings) == 1
    assert warnings[0].startswith(f"{named} does not compile")


def test_the_warning_says_the_release_is_unknown_where_reading_it_raises(monkeypatch):
    """jinja2 3.2 looks `__version__` up in the package's metadata, and the lookup raises where a bundled application
    ships none, which a default for a missing attribute does not catch: the warning says the release is unknown, so
    the import still only warns."""

    def no_metadata(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    stand_in = ModuleType("jinja2")
    stand_in.__getattr__ = no_metadata
    monkeypatch.setattr(rules_module, "jinja2", stand_in)
    compile_and_or_with_jinjas_own_visitors(monkeypatch)

    with warnings_logged() as warnings:
        assert not rules_module._check_and_or_decide()

    assert len(warnings) == 1
    assert warnings[0].startswith("jinja2 (version unknown) does not compile")


def test_a_check_still_calls_what_the_record_holds_through_the_sandbox():
    """A check calls its own guards directly, and anything the record holds through the sandbox, which refuses a
    callable marked as one that alters data, on either side of an `or`."""

    def purge() -> int:
        return 1

    purge.alters_data = True
    node = screening("ops.purge() == 1 or app.a == 1", inputs=("app", "ops"))

    status, message = outcome(run(node, {"ops": {"purge": purge}, "app": {"a": 1}}))

    assert status == "not_evaluated"
    assert message.startswith("check could not be evaluated: ") and message.endswith(" is not safely callable")


# --- a path reads what Jinja reads ------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize(
    ("check", "record", "evaluated"),
    [
        ("code[0] == 'A'", {"code": "ABC"}, {"code[0]": "A"}),
        ("app.zip[0] == '9'", {"app": {"zip": "90210"}}, {"app.zip[0]": "9"}),
        ("m[0] == 'x'", {"m": {0: "x"}}, {"m[0]": "x"}),
        ("m.k == 1 and m[0] == 'x'", {"m": MappingProxyType({"k": 1, 0: "x"})}, {"m.k": 1, "m[0]": "x"}),
    ],
    ids=["a-string-index", "a-string-index-under-a-member", "an-int-key", "a-mapping-that-is-no-dict"],
)
def test_a_value_read_by_a_string_index_or_a_mapping_key_is_there_in_the_verdict_and_the_finding(
    check, record, evaluated, policy
):
    """The check reads what Jinja reads, and the finding shows the same value: a character of a text by its index,
    a key of any mapping, whatever the key's type."""
    node = screening(check, policy, inputs=tuple(record))

    (finding,) = run(node, record)["findings"]

    assert (finding["status"], finding["message"], finding["evaluated"]) == ("pass", None, evaluated)


@pytest.mark.parametrize(
    ("policy", "status", "message", "evaluated"),
    [
        (None, "not_evaluated", "missing value for code[3]", {"code[3]": None}),
        ("fail", "fail", "missing value for code[3]", {"code[3]": None}),
        ("not_applicable", "not_applicable", "does not apply: missing value for code[3]", {}),
    ],
)
def test_an_index_past_the_end_of_a_text_is_a_missing_value(policy, status, message, evaluated):
    node = screening("code[3] == 'D'", policy, inputs=("code",))

    (finding,) = run(node, {"code": "ABC"})["findings"]

    assert (finding["status"], finding["message"], finding["evaluated"]) == (status, message, evaluated)


@pytest.mark.parametrize(
    ("path", "context", "value"),
    [
        ("code[0]", {"code": "ABC"}, "A"),
        ("code[-1]", {"code": "ABC"}, "C"),
        ("m[0]", {"m": {0: "x"}}, "x"),
        ("m.k", {"m": MappingProxyType({"k": 1})}, 1),
        ("m[0]", {"m": MappingProxyType({0: "x"})}, "x"),
        ("t._id", {"t": {"_id": 7}}, 7),
    ],
    ids=[
        "a-text-index",
        "a-text-index-from-the-end",
        "an-int-key",
        "a-key-of-a-mapping",
        "an-int-key-of-a-mapping",
        "a-key-with-an-underscore",
    ],
)
def test_a_path_steps_into_a_text_by_index_and_into_any_mapping_by_key(path, context, value):
    assert resolve_path(context, path) == value


@pytest.mark.parametrize(
    ("path", "context"),
    [
        ("code[3]", {"code": "ABC"}),
        ("m[1]", {"m": {0: "x"}}),
        ("m['0']", {"m": {0: "x"}}),
        ("t.items", {"t": {}}),
        ("o._value", {"o": SimpleNamespace(_value=1)}),
    ],
    ids=["past-the-end", "a-key-it-lacks", "text-for-an-int-key", "a-method-of-a-dict", "a-private-attribute"],
)
def test_a_path_still_finds_nothing_where_the_record_holds_nothing(path, context):
    """A dict's method is no key the record holds, and an attribute behind an underscore stays out of reach."""
    assert not has(resolve_path(context, path))


@pytest.mark.parametrize(
    ("expression", "record", "errors", "expected"),
    [
        (
            "code[0] / 2",
            {"code": "ABC"},
            {"d": "unsupported operand type(s) for /: 'str' and 'int'"},
            ("not_evaluated", "check could not be evaluated: unsupported operand type(s) for /: 'str' and 'int'"),
        ),
        (
            "limits[code[0]]",
            {"code": "ABC", "limits": {"Z": 1}},
            {},
            ("not_evaluated", "missing value for d (not skipped: the lookup for d found nothing)"),
        ),
    ],
    ids=["a-failure-beside-a-character-that-is-there", "a-lookup-by-a-character-that-found-nothing"],
)
def test_a_derived_value_does_not_pass_off_a_character_that_is_there_as_missing(expression, record, errors, expected):
    """A failure beside a value read by a text index is the derived value's error, and a lookup by one that found
    nothing is a gap no rule may skip: the value read is there, so neither counts as data the record lacks."""
    node = screening("d == 1", "not_applicable", inputs=("code", "limits"), derived=(("d", expression),))

    output = run(node, record)

    assert (output["derived"], output["derived_errors"]) == ({"d": None}, errors)
    assert outcome(output) == expected


# --- a derived value is computed as it always was ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "record", "value"),
    [
        ("(app.margin or 0) < 3", {"app": {"margin": None}}, True),
        (
            "{'first': items[0].discount, 'second': items[1].discount}",
            {"items": [{"discount": 5}, {"sku": "x"}]},
            {"first": 5, "second": None},
        ),
        ("[app.a, app.b]", {"app": {"a": 1}}, [1, None]),
        ("app.rate if app.kind == 'fixed' else app.margin", {"app": {"kind": "fixed", "rate": 5}}, 5),
    ],
    ids=["a-null-it-falls-back-past", "a-dict-without-a-member", "a-list-without-a-member", "the-branch-it-takes"],
)
def test_a_derived_value_is_computed_as_it_always_was(expression, record, value):
    """A derived value does not stop where it reads a missing value: a null it falls back past, or a member missing from
    a list or a dict it builds, still gives it a value."""
    node = screening("value is defined", inputs=("app", "items"), derived=(("value", expression),))

    output = run(node, record)

    assert (output["derived"], output["derived_errors"]) == ({"value": value}, {})
    assert outcome(output) == ("pass", None)


@pytest.mark.parametrize(
    ("check", "record", "policy", "inline", "named"),
    [
        (
            "(app.margin or 0) < 3",
            {"app": {"margin": None}},
            None,
            ("not_evaluated", "missing value for app.margin"),
            ("pass", None),
        ),
        (
            "app.amount / app.term > 1000 or app.margin < 3",
            {"app": {"amount": 5, "term": 0}},
            "not_applicable",
            ("not_evaluated", "check could not be evaluated: division by zero"),
            ("not_applicable", "does not apply: missing value for named"),
        ),
    ],
    ids=["a-null-the-text-falls-back-past", "an-error-before-a-missing-value"],
)
def test_naming_part_of_a_check_as_a_derived_value_can_change_what_the_rule_reports(
    check, record, policy, inline, named
):
    """The check stops at a needed value that is missing, the null `x` in `(x or 0) < 3` as well, and reports an error
    it reaches before one; a derived value computes over the null, and is missing where it fails beside a missing
    value it needs."""
    assert outcome(run(screening(check, policy), record)) == inline
    assert outcome(run(screening("named", policy, derived=(("named", check),)), record)) == named


@pytest.mark.parametrize("policy", POLICIES)
def test_a_derived_value_whose_lookup_found_nothing_gives_way_where_the_same_lookup_in_the_check_is_an_error(policy):
    """Naming part of a check can make the rule looser as well: the derived `lim` is missing where the table lacks the
    program, and gives way to the other side of the `or`, which decides, where the same lookup written in the check is
    an error no side gives way to."""
    record = {"app": {"program": "jumbo", "a": 1}, "limits": {"standard": 1}}
    lookup = ("lim", "limits[app.program]")

    named = screening("lim > 5 or app.a == 1", policy, inputs=("app", "limits"), derived=(lookup,))
    inline = screening("limits[app.program] > 5 or app.a == 1", policy, inputs=("app", "limits"))

    assert outcome(run(named, record)) == ("pass", None)
    assert outcome(run(inline, record)) == (
        "fail" if policy == "fail" else "not_evaluated",
        "check could not be evaluated: 'dict object' has no attribute 'jumbo'",
    )


@pytest.mark.parametrize(
    ("policy", "named"),
    [
        (None, ("not_evaluated", "missing value for d")),
        ("fail", ("fail", "missing value for d")),
        ("not_applicable", ("not_applicable", "does not apply: missing value for d")),
    ],
)
@pytest.mark.parametrize(
    ("expression", "inline"),
    [
        ("app.x or app.y", ("pass", None)),
        ("text(app.blank) or app.y", ("pass", None)),
        ("app.x and app.flag", ("fail", None)),
        ("text(app.blank) and app.flag", ("fail", None)),
    ],
    ids=[
        "or-over-an-absent-side",
        "or-over-a-blank-a-helper-made",
        "and-over-an-absent-side",
        "and-over-a-blank-a-helper-made",
    ],
)
def test_a_derived_value_reads_an_and_or_an_or_as_it_always_did(expression, inline, policy, named):
    """A derived value reads `and` and `or` as Python does: a side that is not there, or a blank a helper made, raises
    when its truth is tested, so the value is missing though the other side would decide, where a check of the same
    text is decided by it."""
    record = {"app": {"y": "y", "flag": False, "blank": "  "}}

    output = run(screening("d", policy, derived=(("d", expression),)), record)

    assert (output["derived"], output["derived_errors"]) == ({"d": None}, {})
    assert outcome(output) == named
    assert outcome(run(screening(expression, policy), record)) == inline


# --- what decides nothing reads as it always did ----------------------------------------------------------------


def test_a_message_reads_a_missing_value_as_it_always_did():
    node = screening("app.amount > 10", message="amount {{ app.amount }} under the floor {{ app.floor }}")

    assert outcome(run(node, {"app": {"amount": 5}})) == ("fail", "amount 5 under the floor")


@pytest.mark.parametrize(
    "message",
    ["note {{ text(app.note) or 'none' }}", "note {{ app.missing or 'none' }}"],
    ids=["or-over-a-blank-a-helper-made", "or-over-an-absent-side"],
)
def test_a_message_reads_an_and_or_an_or_as_it_always_did(message):
    """A message decides nothing: an `or` over a side that is not there, or a blank a helper made, fails to render, and
    the message comes back as written, as it always did, where a check of the same text would take the other side."""
    node = screening("app.amount > 10", message=message)

    assert outcome(run(node, {"app": {"amount": 5, "note": "  "}})) == ("fail", message)


def test_an_expression_node_reads_a_null_as_it_always_did():
    node = Expression(
        name="calc",
        input_fields=[NamedField(name="app")],
        expressions=[ExpressionItem(key="margin", expression="app.margin or 0")],
    )

    result = node.run(input_data={"app": {"margin": None}}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"margin": 0}


@pytest.mark.parametrize(
    ("expression", "app", "value"),
    [
        ("app.a or app.b", {"b": "b"}, "b"),
        ("app.a or 'fallback'", {}, "fallback"),
        ("app.a and app.b", {"b": "b"}, None),
        ("text(app.a) or 'fallback'", {"a": "  "}, "fallback"),
        ("text(app.a) and 'x'", {"a": "  "}, None),
    ],
    ids=["or-absent", "or-absent-fallback", "and-absent", "or-blank", "and-blank"],
)
def test_an_expression_node_reads_an_and_or_an_or_as_it_always_did(expression, app, value):
    """An expression reads a missing input as None: a side that is not there, or a blank, is false, as it always was."""
    node = Expression(
        name="calc", input_fields=[NamedField(name="app")], expressions=[ExpressionItem(key="v", expression=expression)]
    )

    result = node.run(input_data={"app": app}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"v": value}


@pytest.mark.parametrize("expression", ["(number(app.a) + 1) or 5", "(number(app.a) + 1) and 0"], ids=["or", "and"])
def test_an_expression_node_fails_its_run_on_a_blank_it_computes_with_whatever_the_other_side_holds(expression):
    """A blank `number()` used in arithmetic fails an Expression node's run, as it always did: the other side of an
    `and` or an `or` does not stand in for it."""
    node = Expression(
        name="calc", input_fields=[NamedField(name="app")], expressions=[ExpressionItem(key="v", expression=expression)]
    )

    result = node.run(input_data={"app": {"a": "  "}}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message == "missing value: number() found no number"
