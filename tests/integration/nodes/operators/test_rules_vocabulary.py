"""The words a rule is written with: `is present` and `is blank`, `text()`, `number()`, `date()` and `first_present()`.

A record from a form or an extraction says "no answer" in more ways than a missing key: a null, an empty string, a
string of spaces, an empty list. `is present` and `is blank` answer the question outright; `text()` and
`first_present()` read every one of these as missing, a key the record lacks included, so a comparison over them
never passes or fails on a value nobody gave: `text(app.purpose) == 'purchase'` is not evaluated for a blank purpose,
where `app.purpose | trim == 'purchase'` would fail it.

A record also says things a rule cannot read: `TBD` where an amount goes, `March` where a date goes. `number()` and
`date()` read what documents write, `$586,764.00` or `Oct 1, 2026`, and hold anything else as unreadable: the value is
there, so it is present, but a rule that uses it is not evaluated, naming the value, where `| float` would read 0.
"""

import json
from datetime import date
from decimal import Decimal
from typing import Any

import pytest
from jinja2.exceptions import TemplateRuntimeError, UndefinedError

from dynamiq.nodes.operators import Expression, Rules
from dynamiq.nodes.operators.rules import (
    MissingValue,
    RuleUndefined,
    Unreadable,
    UnreadableValue,
    is_blank,
    is_present,
    read_paths,
    to_date,
)
from dynamiq.nodes.types import DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder

# A key the record does not carry at all.
ABSENT = object()


def run(node: Rules | Expression, data: dict) -> dict:
    result = node.run(input_data=data, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output


def by_id(output: dict) -> dict[str, dict]:
    return {finding["rule_id"]: finding for finding in output["findings"]}


def statuses(output: dict) -> dict[str, str]:
    return {rule_id: finding["status"] for rule_id, finding in by_id(output).items()}


def rules(*checks: Rule, member: str = "app") -> Rules:
    return Rules(name="vocabulary", input_fields=[NamedField(name=member)], rules=list(checks))


def record(member: str = "app", **values) -> dict:
    """One member holding `values`; a value of ABSENT leaves its key out."""
    return {member: {key: value for key, value in values.items() if value is not ABSENT}}


# --- `is blank` and `is present` --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "blank"),
    [
        (None, True),
        (ABSENT, True),
        ("", True),
        ("  ", True),
        ([], True),
        ({}, True),
        (0, False),
        (False, False),
        ("x", False),
    ],
    ids=["null", "absent", "empty", "spaces", "empty-list", "empty-dict", "zero", "false", "text"],
)
def test_blank_is_a_value_with_nothing_in_it_and_present_is_every_other(value, blank):
    node = rules(Rule(id="blank", check="app.x is blank"), Rule(id="present", check="app.x is present"))

    output = run(node, record(x=value))

    expected = {"blank": "pass", "present": "fail"} if blank else {"blank": "fail", "present": "pass"}
    assert statuses(output) == expected


def test_is_present_fails_a_rule_whose_value_is_absent_rather_than_holding_it():
    """The test asks about the value, so its absence is the answer, not a value the check could not read."""
    output = run(rules(Rule(id="R1", check="app.x is present")), record())

    assert statuses(output) == {"R1": "fail"}
    assert by_id(output)["R1"]["evaluated"] == {"app.x": None}
    assert output["status"] == "fail"


def test_a_key_named_like_a_dict_method_is_blank_when_the_record_lacks_it():
    """Without the key, `invoice.items` reaches the dict's method, which is no value: it is blank, as an absent key is,
    and a fallback stands in for it."""
    node = rules(
        Rule(id="present", check="invoice.items is present"),
        Rule(id="blank", check="invoice.items is blank"),
        Rule(id="fallback", check="first_present(invoice.items, invoice.lines) == ['a']"),
        Rule(id="counted", check="first_present(invoice.items, invoice.lines) | length > 0"),
        member="invoice",
    )

    absent = run(node, record("invoice", lines=["a"]))
    given = run(node, record("invoice", items=["a"], lines=[]))

    assert statuses(absent) == {"present": "fail", "blank": "pass", "fallback": "pass", "counted": "pass"}
    # With the key, the member is the data, as everywhere else.
    assert statuses(given) == {"present": "pass", "blank": "fail", "fallback": "pass", "counted": "pass"}


# --- text() -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "status", "message"),
    [
        (" purchase ", "pass", None),
        ("lease", "fail", None),
        ("  ", "not_evaluated", "missing value for app.purpose"),
    ],
    ids=["padded", "other", "blank"],
)
def test_text_trims_the_value_and_reads_blank_text_as_missing(purpose, status, message):
    output = run(rules(Rule(id="PUR-01", check="text(app.purpose) == 'purchase'")), record(purpose=purpose))

    finding = by_id(output)["PUR-01"]
    assert (finding["status"], finding["message"]) == (status, message)
    assert finding["evaluated"] == {"app.purpose": purpose}


def test_a_test_answers_about_blank_text_where_a_comparison_cannot():
    node = rules(
        Rule(id="given", check="text(app.purpose) is present"),
        Rule(id="compared", check="text(app.purpose) == 'purchase'"),
    )

    assert statuses(run(node, record(purpose="  "))) == {"given": "fail", "compared": "not_evaluated"}


@pytest.mark.parametrize(
    ("check", "purpose"),
    [
        ("text(app.purpose) | lower == 'purchase'", " Purchase "),
        ("text(app.purpose) | upper == 'PURCHASE'", "purchase"),
        ("text(app.purpose) | trim == 'purchase'", "purchase"),
        ("text(app.purpose) | title == 'Hire Purchase'", "hire purchase"),
        ("text(app.purpose) | capitalize == 'Hire purchase'", "HIRE PURCHASE"),
        ("text(app.purpose) | replace('-', ' ') == 'hire purchase'", "hire-purchase"),
    ],
    ids=["lower", "upper", "trim", "title", "capitalize", "replace"],
)
def test_a_text_filter_keeps_a_blank_value_missing_rather_than_failed(check, purpose):
    """Jinja's text filters read an undefined value as '', which a comparison would take for an answer."""
    node = rules(Rule(id="PUR-01", check=check))

    assert statuses(run(node, record(purpose=purpose))) == {"PUR-01": "pass"}
    blank = by_id(run(node, record(purpose="  ")))["PUR-01"]
    assert (blank["status"], blank["message"]) == ("not_evaluated", "missing value for app.purpose")


@pytest.mark.parametrize("tags", [[], {}], ids=["empty-list", "empty-dict"])
def test_the_reason_names_an_empty_list_or_mapping_as_the_missing_value(tags):
    output = run(rules(Rule(id="TAG-01", check="text(app.tags) == 'priority'")), record(tags=tags))

    finding = by_id(output)["TAG-01"]
    assert (finding["status"], finding["message"]) == ("not_evaluated", "missing value for app.tags")


@pytest.mark.parametrize(
    ("address", "status", "message"),
    [
        ({"city": "Springfield", "state": "ca"}, "pass", None),
        ({"city": "Springfield"}, "not_evaluated", "missing value for app.address.state"),
        (ABSENT, "fail", None),
    ],
    ids=["given", "no-state", "no-address"],
)
def test_text_keeps_an_absent_value_missing_through_a_filter(address, status, message):
    """A guard lets a value under it be missing; `| upper` alone would read the absent state as '' and fail the rule,
    where `text()` keeps it missing. Without an address the guard decides, as it always has."""
    node = rules(Rule(id="ST-01", check="has(app.address) and text(app.address.state) | upper == 'CA'"))

    finding = by_id(run(node, record(address=address)))["ST-01"]

    assert (finding["status"], finding["message"]) == (status, message)


# --- first_present() --------------------------------------------------------------------------------------------


def test_first_present_reads_each_value_leniently_and_guards_no_other_read():
    """`has(a.b)` lets a read under `a.b` be missing as well; a fallback is no question about `a.b`."""
    reads = read_paths("first_present(a.b, c) == 1 and a.b.c > 0")
    assert (reads.required, reads.optional) == (["a.b.c"], ["a.b", "c"])

    # A value read as a fallback and on its own as well is required.
    reads = read_paths("first_present(a.b, c) == 1 and a.b > 0")
    assert (reads.required, reads.optional) == (["a.b"], ["c"])

    # A helper inside the call reads leniently too, so an absent value reaches `text()` and is skipped as blank.
    reads = read_paths("first_present(text(a.b), 'none') == 'x'")
    assert (reads.required, reads.optional) == ([], ["a.b"])


@pytest.mark.parametrize(
    ("delivery", "billing", "status", "message"),
    [
        ("DE", "FR", "pass", None),
        (ABSENT, "DE", "pass", None),
        ("  ", "DE", "pass", None),
        ("FR", "DE", "fail", None),
        (ABSENT, None, "not_evaluated", "missing value for shipment.delivery_country"),
    ],
    ids=["first", "absent-first", "blank-first", "first-decides", "none-present"],
)
def test_first_present_takes_the_first_value_that_is_present(delivery, billing, status, message):
    node = rules(
        Rule(id="VAT-DE", check="first_present(shipment.delivery_country, shipment.billing_country) == 'DE'"),
        member="shipment",
    )

    output = run(node, record("shipment", delivery_country=delivery, billing_country=billing))

    finding = by_id(output)["VAT-DE"]
    assert (finding["status"], finding["message"]) == (status, message)


def test_a_fallback_chain_that_finds_nothing_is_missing_wherever_it_stands():
    node = rules(
        Rule(id="bare", check="first_present(ticket.owner, ticket.team)"),
        Rule(id="applies", applies_when="first_present(ticket.owner, ticket.team) == 'billing'", check="true"),
        Rule(id="unequal", check="first_present(ticket.owner, ticket.team) != 'billing'"),
        Rule(id="texts", check="first_present(text(ticket.owner), text(ticket.team), 'triage') == 'triage'"),
        member="ticket",
    )

    output = run(node, record("ticket", owner="  ", team=None))

    assert statuses(output) == {
        "bare": "not_evaluated",
        "applies": "not_evaluated",
        "unequal": "not_evaluated",
        "texts": "pass",
    }
    assert {by_id(output)[rule_id]["message"] for rule_id in ("bare", "applies", "unequal")} == {
        "missing value for ticket.owner"
    }


# --- names ------------------------------------------------------------------------------------------------------


def test_a_derived_value_may_take_a_name_the_vocabulary_added():
    """A workflow built with a derived value called `text` or `number` keeps building: the value is what a rule
    reads under that name, and the helper what a rule calls, as with an input of that name."""
    node = Rules(
        name="vocabulary",
        input_fields=[NamedField(name="app")],
        derived_values=[
            DerivedValue(name="text", expression="app.purpose | upper"),
            DerivedValue(name="number", expression="app.units * 2"),
        ],
        rules=[
            Rule(id="read", check="text == ' PURCHASE ' and number == 6"),
            Rule(id="called", check="text(app.purpose) == 'purchase'"),
        ],
    )

    output = run(node, record(purpose=" purchase ", units=3))

    assert output["derived"] == {"text": " PURCHASE ", "number": 6}
    assert statuses(output) == {"read": "pass", "called": "pass"}


@pytest.mark.parametrize("name", ["has", "date", "round"])
def test_a_derived_value_named_like_an_original_helper_is_still_refused(name):
    with pytest.raises(ValueError, match=f"derived value '{name}' is already the name of an input or a helper"):
        Rules(name="vocabulary", derived_values=[DerivedValue(name=name, expression="1")], rules=[])


def test_a_rule_reading_and_calling_a_name_the_vocabulary_added_is_held_when_it_runs_not_refused_at_build():
    """A workflow with an input called `text` and a check `text(text) == 'x'`, written before `text` was a helper,
    keeps building. The clash holds that one rule as an error, whether or not the record carries `text`, and every
    other rule runs."""
    node = Rules(
        name="vocabulary",
        input_fields=[NamedField(name="text"), NamedField(name="app")],
        derived_values=[DerivedValue(name="label", expression="text(text) | upper")],
        rules=[
            Rule(id="clash", name="clash", check="text(text) == 'x'"),
            Rule(id="sibling", check="app.units > 0"),
        ],
    )

    for data in ({"text": "x", "app": {"units": 3}}, {"app": {"units": 3}}):
        output = run(node, data)

        assert statuses(output) == {"clash": "not_evaluated", "sibling": "pass"}
        assert by_id(output)["clash"]["message"] == (
            "check could not be evaluated: Rules 'vocabulary', rule 1 (clash): the check reads 'text' as a value and"
            " calls it as a helper"
        )
        assert output["derived"] == {"label": None}


def test_an_expression_reading_and_calling_a_name_the_vocabulary_added_fails_its_run_not_its_build():
    node = Expression(
        name="labels",
        input_fields=[NamedField(name="text")],
        expressions=[ExpressionItem(key="label", expression="text(text)")],
    )

    result = node.run(input_data={"text": " a "}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert "Expression 'labels': 'label' reads 'text' as a value and calls it as a helper" in result.error.message


def test_a_clash_on_an_original_helper_name_is_still_refused_at_build():
    with pytest.raises(ValueError, match="rule 1: the check reads 'date' as a value and calls it as a helper"):
        Rules(name="vocabulary", input_fields=[NamedField(name="date")], rules=[Rule(id="c", check="date(date)")])
    with pytest.raises(ValueError, match="'due' reads 'date' as a value and calls it as a helper"):
        Expression(name="due", expressions=[ExpressionItem(key="due", expression="date(date)")])


def test_a_missing_value_error_names_its_path_when_the_raiser_knows_it():
    missing = MissingValue.for_path("app.purpose")

    assert isinstance(missing, UndefinedError)
    assert (str(missing), missing.path) == ("missing value for app.purpose", "app.purpose")
    # Jinja raises an undefined value's error with the message alone.
    assert MissingValue("'x' is undefined").path is None


# --- the Expression node ----------------------------------------------------------------------------------------


def test_an_expression_reads_a_blank_as_none_and_every_other_undefined_as_before():
    node = Expression(
        name="profile",
        input_fields=[NamedField(name="name"), NamedField(name="nickname"), NamedField(name="count")],
        expressions=[
            ExpressionItem(key="spaces", expression="text('  ')"),
            ExpressionItem(key="trimmed", expression="text(name)"),
            ExpressionItem(key="zero", expression="text(count)"),
            ExpressionItem(key="blank_lowered", expression="text(nickname) | lower"),
            ExpressionItem(key="absent_lowered", expression="text(missing) | lower"),
            ExpressionItem(key="missing_lowered", expression="missing | lower"),
            ExpressionItem(key="replaced", expression="name | replace('A', 'E')"),
            ExpressionItem(key="shown", expression="first_present(nickname, missing, name)"),
            ExpressionItem(key="nothing", expression="first_present(nickname, missing)"),
            ExpressionItem(key="asked", expression="nickname is blank"),
        ],
    )

    output = run(node, {"name": " Ada ", "nickname": "  ", "count": 0})

    assert output == {
        "spaces": None,
        "trimmed": "Ada",
        "zero": "0",
        "blank_lowered": None,
        "absent_lowered": None,
        "missing_lowered": "",
        "replaced": " Eda ",
        "shown": " Ada ",
        "nothing": None,
        "asked": True,
    }


# --- number() ---------------------------------------------------------------------------------------------------


def reader(expression: str, inputs: dict) -> Expression:
    return Expression(
        name="reader",
        input_fields=[NamedField(name=name) for name in inputs],
        expressions=[ExpressionItem(key="value", expression=expression)],
    )


def evaluate(expression: str, **inputs) -> Any:
    """What one expression of an Expression node computes over `inputs`."""
    return run(reader(expression, inputs), inputs)["value"]


def failure(expression: str, **inputs) -> str:
    """The error one expression of an Expression node fails its run with."""
    result = reader(expression, inputs).run(input_data=inputs, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.FAILURE
    return result.error.message


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$586,764.00", 586764.0),
        ("1,234", 1234),
        ("6.25%", 6.25),
        ("(1,200.50)", -1200.5),
        (" -£1,234,567.89 ", -1234567.89),
        ("€ 12", 12),
        ("+0.5", 0.5),
        ("1 234", 1234),
        ("1 234.56", 1234.56),
        ("6.25 %", 6.25),
        (42, 42),
        (2.5, 2.5),
        (Decimal("586764.00"), 586764.0),
        (Decimal("1E+3"), 1000),
    ],
    ids=[
        "currency",
        "grouped",
        "percent",
        "parenthesised",
        "negative-pounds",
        "euro-spaced",
        "signed",
        "space-grouped",
        "space-grouped-decimal",
        "spaced-percent",
        "int",
        "float",
        "decimal",
        "decimal-exponent",
    ],
)
def test_number_reads_an_amount_the_way_a_document_writes_it(raw, expected):
    value = evaluate("number(raw)", raw=raw)

    # With a decimal point it is a float, without one an int, and never a Decimal, which a workflow's JSON cannot carry.
    assert (value, type(value)) == (expected, type(expected))


@pytest.mark.parametrize(
    "raw",
    [
        "TBD",
        "12,5",
        "1.234,56",
        "1,23",
        "0,125",
        "nan",
        "inf",
        "1_000",
        "1e5",
        "١٢",
        "(-5)",
        "5.",
        "$",
        "100, 200",
        "5 10",
        "12 5",
        "1 23",
        "1 234,56",
        "12$5",
        True,
        float("nan"),
        float("inf"),
    ],
    ids=[
        "words",
        "decimal-comma",
        "european",
        "short-group",
        "zero-group",
        "nan",
        "inf",
        "underscore",
        "exponent",
        "arabic-indic",
        "double-negative",
        "trailing-point",
        "symbol-only",
        "comma-then-space",
        "spaced-short-group",
        "spaced-decimal",
        "spaced-group-of-two",
        "spaced-with-decimal-comma",
        "currency-between-digits",
        "true",
        "nan-float",
        "inf-float",
    ],
)
def test_number_holds_a_value_it_cannot_read_rather_than_guessing(raw):
    """Each of these is there, so it is not missing, yet any reading of it is a guess: `12,5` is 12.5 or 125
    depending on who wrote it, and so is `12 5`; `100, 200` may be two amounts; `| float` reads `TBD` as 0; and
    Python itself reads `1e5`, `1_000` and `١٢`."""
    node = rules(
        Rule(id="AMT-01", check="number(doc.amount) > 1000"),
        Rule(id="bare", check="number(doc.amount)"),
        member="doc",
    )

    output = run(node, record("doc", amount=raw))

    reason = f"check could not be evaluated: not a number: {raw!r}"
    assert {(finding["status"], finding["message"]) for finding in output["findings"]} == {("not_evaluated", reason)}


@pytest.mark.parametrize(
    ("amount", "status", "message"),
    [
        ("$1,500.00", "pass", None),
        ("900", "fail", None),
        ("TBD", "not_evaluated", "check could not be evaluated: not a number: 'TBD'"),
        ("  ", "not_evaluated", "missing value for doc.amount"),
        (ABSENT, "not_evaluated", "missing value for doc.amount"),
    ],
    ids=["over", "under", "unreadable", "blank", "absent"],
)
def test_a_rule_over_number_decides_on_an_amount_and_holds_one_it_cannot_read(amount, status, message):
    node = rules(Rule(id="AMT-01", check="number(doc.amount) > 1000"), member="doc")

    finding = by_id(run(node, record("doc", amount=amount)))["AMT-01"]

    assert (finding["status"], finding["message"]) == (status, message)
    # The finding shows what the record says, not what the reader made of it.
    assert finding["evaluated"] == {"doc.amount": None if amount is ABSENT else amount}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234,56", 1234.56),
        ("12,5", 12.5),
        ("1.234", 1234),
        ("€ 1.234.567,89", 1234567.89),
        ("12 345,67", 12345.67),
        ("1 234,5", 1234.5),
    ],
    ids=["grouped", "decimal", "thousands", "euro", "space-grouped", "narrow-space-grouped"],
)
def test_number_reads_a_decimal_comma_when_the_rule_says_so(raw, expected):
    value = evaluate("number(raw, decimal=',')", raw=raw)

    assert (value, type(value)) == (expected, type(expected))


def test_a_decimal_comma_reader_holds_a_decimal_point_as_unreadable():
    assert "not a number: '1,234.56'" in failure("number(raw, decimal=',')", raw="1,234.56")
    assert "not a number: '1.5'" in failure("number(raw, decimal=',')", raw="1.5")
    assert "not a number: '1 234.567,89'" in failure("number(raw, decimal=',')", raw="1 234.567,89")


def test_a_float_or_int_filter_after_number_holds_the_rule_where_the_filter_alone_reads_zero():
    node = rules(
        Rule(id="filter", check="doc.amount | float == 0"),
        Rule(id="float", check="number(doc.amount) | float == 0"),
        Rule(id="int", check="number(doc.amount) | int == 0"),
        member="doc",
    )

    output = run(node, record("doc", amount="TBD"))

    # The filter alone passes a rule on an amount nobody gave.
    assert statuses(output) == {"filter": "pass", "float": "not_evaluated", "int": "not_evaluated"}
    assert by_id(output)["float"]["message"] == "check could not be evaluated: not a number: 'TBD'"


def test_first_present_stops_at_an_unreadable_value_rather_than_skipping_it():
    """A value nobody could read is there: skipping it would let a fallback decide over what the record says."""
    assert evaluate("first_present(number('5'), number('TBD'))") == 5
    assert evaluate("first_present(number(''), number('7'))") == 7

    node = rules(Rule(id="NET-01", check="first_present(number(doc.net), number(doc.gross)) > 0"), member="doc")
    finding = by_id(run(node, record("doc", net="TBD", gross="5")))["NET-01"]

    assert (finding["status"], finding["message"]) == (
        "not_evaluated",
        "check could not be evaluated: not a number: 'TBD'",
    )


@pytest.mark.parametrize(
    ("check", "value", "reason"),
    [
        ("has(date(doc.value))", "March", "not a date: 'March'"),
        ("date(doc.value) is defined", "March", "not a date: 'March'"),
        ("date(doc.value) is not none", "March", "not a date: 'March'"),
        ("date(doc.value) is present", "March", "not a date: 'March'"),
        ("date(doc.value) is string", "March", "not a date: 'March'"),
        ("date(doc.value) is sameas none", "March", "not a date: 'March'"),
        ("[date(doc.value)] | select('defined') | list | length == 1", "March", "not a date: 'March'"),
        ("has(number(doc.value))", "TBD", "not a number: 'TBD'"),
        ("number(doc.value) is number", "TBD", "not a number: 'TBD'"),
        ("number(doc.value) is blank", "TBD", "not a number: 'TBD'"),
        ("first_present(number(doc.value), 0) > 1", "TBD", "not a number: 'TBD'"),
    ],
)
def test_asking_about_a_value_nobody_could_read_holds_the_rule_naming_the_text(check, value, reason):
    """`date('March')` raised before `date()` read what documents write, so `has(date(x))` was never answered over
    text that is no date; it still is not. A presence or type test is no verdict on the value, and a fallback does
    not hide it either."""
    node = rules(Rule(id="R1", check=check), member="doc")

    finding = by_id(run(node, record("doc", value=value)))["R1"]

    assert (finding["status"], finding["message"]) == ("not_evaluated", f"check could not be evaluated: {reason}")


@pytest.mark.parametrize(
    "expression", ["has(date(value))", "date(value) is defined", "date(value) is none", "date(value) is present"]
)
def test_an_expression_asking_about_a_date_nobody_could_read_fails_the_run_as_using_it_does(expression):
    assert "not a date: 'March'" in failure(expression, value="March")


@pytest.mark.parametrize(
    "check",
    [
        "number(doc.amount) | round(2) > 5",
        "round(number(doc.amount)) > 5",
        "abs(number(doc.amount)) > 5",
        "number(doc.amount) | abs > 5",
        "number(doc.amount) | int > 5",
        "date(doc.issued) | string == '2026-10-01'",
    ],
    ids=["round-filter", "round", "abs", "abs-filter", "int-filter", "string-filter"],
)
def test_a_blank_read_stays_missing_through_the_helpers_and_filters_that_round_or_print_it(check):
    """`round`, `abs` and `| string` would otherwise raise a type error or read a blank as '', a verdict on a value
    nobody gave: a blank amount rounded is as missing as a blank amount compared."""
    node = rules(Rule(id="R1", check=check), member="doc")

    assert statuses(run(node, record("doc", amount="12.345", issued="Oct 1, 2026"))) == {"R1": "pass"}
    finding = by_id(run(node, record("doc", amount="  ", issued="")))["R1"]
    assert (finding["status"], finding["message"]) == (
        "not_evaluated",
        f"missing value for {read_paths(check).required[0]}",
    )


def test_a_reader_reads_its_argument_and_never_its_own_name():
    reads = read_paths("number(doc.amount) > 1000 and date(doc.issued, format='%d.%m.%Y') < today()")
    assert (reads.required, reads.optional) == (["doc.amount", "doc.issued"], [])

    reads = read_paths("first_present(number(doc.net), number(doc.gross)) > 0")
    assert (reads.required, reads.optional) == ([], ["doc.net", "doc.gross"])


def test_a_rule_reading_and_calling_number_is_held_when_it_runs_not_refused_at_build():
    """`number` is a name the vocabulary added, so an input of that name keeps building, as one called `text` does."""
    node = Rules(
        name="vocabulary",
        input_fields=[NamedField(name="number")],
        rules=[Rule(id="clash", name="clash", check="number(number) > 1")],
    )

    finding = by_id(run(node, {"number": "5"}))["clash"]

    assert (finding["status"], finding["message"]) == (
        "not_evaluated",
        "check could not be evaluated: Rules 'vocabulary', rule 1 (clash): the check reads 'number' as a value and"
        " calls it as a helper",
    )
    assert "reads 'number' as a value and calls it as a helper" in failure("number(number)", number="5")


# --- date() -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "raw", "expected"),
    [
        ("date(raw)", "2026/08/07", date(2026, 8, 7)),
        ("date(raw)", "Oct 1, 2026", date(2026, 10, 1)),
        ("date(raw)", "October 1, 2026", date(2026, 10, 1)),
        ("date(raw)", "Sept. 1 2026", date(2026, 9, 1)),
        ("date(raw, format='%d.%m.%Y')", "17.07.2026", date(2026, 7, 17)),
        # The shapes it read before.
        ("date(raw)", "2026-08-07T10:15:00Z", date(2026, 8, 7)),
        ("date(raw)", "08/07/2026", date(2026, 8, 7)),
    ],
    ids=["year-first-slashes", "short-month", "month", "sept-without-comma", "format", "iso", "us"],
)
def test_date_reads_the_shapes_a_document_writes_a_date_in(expression, raw, expected):
    assert evaluate(expression, raw=raw) == expected


@pytest.mark.parametrize(
    ("opened", "status", "message"),
    [
        ("Oct 1, 2026", "pass", None),
        ("2027/01/05", "fail", None),
        ("March", "not_evaluated", "check could not be evaluated: not a date: 'March'"),
        ("", "not_evaluated", "missing value for app.opened"),
        ("  ", "not_evaluated", "missing value for app.opened"),
    ],
    ids=["before", "after", "unreadable", "empty", "spaces"],
)
def test_a_rule_over_date_holds_text_that_is_no_date_and_reads_blank_text_as_missing(opened, status, message):
    node = rules(Rule(id="DT-01", check="date(app.opened) < date('2026-12-31')"))

    finding = by_id(run(node, record(opened=opened)))["DT-01"]

    assert (finding["status"], finding["message"]) == (status, message)


def test_a_date_that_does_not_match_the_format_given_is_unreadable_naming_the_format():
    node = rules(Rule(id="DT-02", check="date(app.opened, format='%d.%m.%Y') < today()"))

    finding = by_id(run(node, record(opened="2026-07-17")))["DT-02"]

    assert (finding["status"], finding["message"]) == (
        "not_evaluated",
        "check could not be evaluated: not a date: '2026-07-17' (format '%d.%m.%Y')",
    )


@pytest.mark.parametrize(
    ("closed", "raw", "read"),
    [
        ("Oct 1, 2026", ("pass", None), ("pass", None)),
        (
            "",
            ("not_evaluated", "check could not be evaluated: not a date: ''"),
            ("not_evaluated", "missing value for app.closed"),
        ),
        (
            "March",
            ("not_evaluated", "check could not be evaluated: not a date: 'March'"),
            ("not_evaluated", "check could not be evaluated: not a date: 'March'"),
        ),
    ],
    ids=["dated", "blank", "unreadable"],
)
def test_days_between_reads_the_new_formats_and_a_blank_read_through_date_is_missing(closed, raw, read):
    """`days_between` reads raw values as it always has; read through `date()`, a blank one is missing instead."""
    node = rules(
        Rule(id="raw", check="days_between(app.opened, app.closed) == 55"),
        Rule(id="read", check="days_between(date(app.opened), date(app.closed)) == 55"),
    )

    findings = by_id(run(node, record(opened="2026/08/07", closed=closed)))

    assert (findings["raw"]["status"], findings["raw"]["message"]) == raw
    assert (findings["read"]["status"], findings["read"]["message"]) == read


@pytest.mark.parametrize(
    ("placed", "shipped", "message"),
    [
        ("sometime", "TBD", "check could not be evaluated: not a date: 'sometime'"),
        ("TBD", "sometime", "check could not be evaluated: not a date: 'TBD'"),
        (ABSENT, "TBD", "missing value for order.placed"),
        ("TBD", ABSENT, "missing value for order.shipped"),
    ],
    ids=["both-unreadable", "both-unreadable-reversed", "placed-missing", "shipped-missing"],
)
def test_days_between_names_the_first_argument_when_both_dates_are_bad(placed, shipped, message):
    """`days_between(date(a), date(b))` reads `a` before `b`, so when both are unreadable the reason is `a`'s,
    in reading order, whichever text it holds. A value the record lacks outright is caught before the check
    ever runs, so it is named instead, on whichever side it sits."""
    node = rules(Rule(id="R1", check="days_between(date(order.placed), date(order.shipped)) <= 30"), member="order")

    finding = by_id(run(node, record("order", placed=placed, shipped=shipped)))["R1"]

    assert (finding["status"], finding["message"]) == ("not_evaluated", message)


def test_to_date_reads_the_new_formats_and_still_raises_on_what_it_cannot_read():
    """`days_between`, an effective window and `as_of` read their dates through it."""
    assert to_date("2026/8/7") == date(2026, 8, 7)
    assert to_date(" oct 1, 2026 ") == date(2026, 10, 1)
    assert to_date("17.07.2026", format="%d.%m.%Y") == date(2026, 7, 17)
    for text in ("March", "Oct 12026", "Octember 1, 2026", "1 Oct 2026", ""):
        with pytest.raises(ValueError, match=f"^not a date: {text!r}$"):
            to_date(text)
    with pytest.raises(ValueError):
        to_date("2026/13/01")
    # A value it is handed that is already missing or unreadable raises its own error.
    with pytest.raises(UnreadableValue, match="^not a number: 'TBD'$"):
        to_date(Unreadable("TBD", "not a number: 'TBD'"))
    with pytest.raises(UndefinedError, match="^'opened' is undefined$"):
        to_date(RuleUndefined(name="opened"))


def test_an_effective_window_and_as_of_read_the_new_formats():
    node = Rules(name="vocabulary", rules=[Rule(id="NEW-01", check="true", effective_from="2026/10/01")])

    assert statuses(run(node, {"as_of": "Sep 30, 2026"})) == {"NEW-01": "not_applicable"}
    assert statuses(run(node, {"as_of": "October 1, 2026"})) == {"NEW-01": "pass"}


# --- unreadable values ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "use",
    [
        lambda value: value == 5,
        lambda value: value != 5,
        lambda value: value < 5,
        lambda value: 5 < value,
        lambda value: value + 1,
        lambda value: 1 - value,
        lambda value: value * 2,
        lambda value: 10 / value,
        lambda value: -value,
        lambda value: 5 in [value],
        lambda value: value["cents"],
        bool,
        len,
        hash,
        iter,
        float,
        int,
        round,
        abs,
        str,
        lambda value: f"{value}",
        lambda value: "%s" % value,
        lambda value: format(value, ".2f"),
    ],
    ids=[
        "equal",
        "unequal",
        "less",
        "reflected",
        "add",
        "subtract-from",
        "multiply",
        "divide-into",
        "negate",
        "in",
        "item",
        "bool",
        "len",
        "hash",
        "iter",
        "float",
        "int",
        "round",
        "abs",
        "str",
        "f-string",
        "percent-format",
        "format-spec",
    ],
)
def test_an_unreadable_value_refuses_every_use_naming_why(use):
    with pytest.raises(UnreadableValue, match="^not a number: 'TBD'$") as raised:
        use(Unreadable("TBD", "not a number: 'TBD'"))

    # The error carries the value, so whoever catches it can keep what the record said.
    assert raised.value.value == "TBD"


def test_an_unreadable_value_counts_as_present_and_raises_no_error_a_filter_would_read_as_zero():
    unreadable = Unreadable("TBD", "not a number: 'TBD'")

    assert is_present(unreadable) and not is_blank(unreadable)
    # A runtime error, never a ValueError or a TypeError: Jinja's `| float` and `| int` read either as "no number"
    # and return 0.
    assert issubclass(UnreadableValue, TemplateRuntimeError)
    assert not issubclass(UnreadableValue, (ValueError, TypeError))


@pytest.mark.parametrize(
    ("check", "reason"),
    [
        ("date(app.opened) | string == '2026-10-01'", "not a date: 'March'"),
        ("date(app.opened) | string != '2026-10-01'", "not a date: 'March'"),
        ("date(app.opened) | lower == 'march'", "not a date: 'March'"),
        ("(date(app.opened) ~ '') == 'March'", "not a date: 'March'"),
        ("text(date(app.opened)) == 'March'", "not a date: 'March'"),
        ("'2026' in date(app.opened) | string", "not a date: 'March'"),
        ("[date(app.opened)] | join == 'March'", "not a date: 'March'"),
        ("'%s' | format(date(app.opened)) == 'March'", "not a date: 'March'"),
        ("number(app.amount) | string == '1000'", "not a number: 'TBD'"),
        ("number(app.amount) | string != '1000'", "not a number: 'TBD'"),
    ],
    ids=[
        "string-equal",
        "string-unequal",
        "lower",
        "concatenated",
        "text",
        "in-string",
        "joined",
        "formatted",
        "number-string-equal",
        "number-string-unequal",
    ],
)
def test_an_unreadable_value_never_becomes_text_a_check_could_compare(check, reason):
    """`| string`, a text filter, `~` or `text()` would hand the check back the text the reader refused, and the
    check would pass or fail on a date or an amount nobody could read."""
    finding = by_id(run(rules(Rule(id="R1", check=check)), record(opened="March", amount="TBD")))["R1"]

    assert (finding["status"], finding["message"]) == ("not_evaluated", f"check could not be evaluated: {reason}")


def test_a_message_prints_an_unreadable_value_as_the_text_the_record_holds():
    """Only a message turns an unreadable value into text, so a reviewer reads what the record says."""
    node = rules(
        Rule(
            id="AMT-01",
            check="number(app.amount) > 1000",
            message="Amount {{ number(app.amount) }} for a file opened {{ date(app.opened) }}",
        )
    )

    finding = by_id(run(node, record(amount="900", opened="March")))["AMT-01"]

    assert (finding["status"], finding["message"]) == ("fail", "Amount 900 for a file opened March")


def test_an_unreadable_value_has_no_members_a_rule_can_read():
    node = rules(
        Rule(id="year", check="date(app.opened).year == 2026"),
        Rule(id="value", check="number(app.amount).value == 'TBD'"),
    )

    output = run(node, record(opened="March", amount="TBD"))

    assert {rule_id: finding["message"] for rule_id, finding in by_id(output).items()} == {
        "year": "check could not be evaluated: not a date: 'March'",
        "value": "check could not be evaluated: not a number: 'TBD'",
    }


def test_a_finding_shows_an_unreadable_value_with_its_reason():
    """A record may carry a value an earlier step could not read; the finding says so rather than show the text."""
    node = rules(Rule(id="AMT-01", check="doc.amount > 1000"), member="doc")

    finding = by_id(run(node, {"doc": {"amount": Unreadable("TBD", "not a number: 'TBD'")}}))["AMT-01"]

    assert (finding["status"], finding["message"]) == (
        "not_evaluated",
        "check could not be evaluated: not a number: 'TBD'",
    )
    assert finding["evaluated"] == {"doc.amount": "unreadable: not a number: 'TBD'"}


def test_an_expression_fails_its_run_on_a_value_it_cannot_read_and_reads_a_blank_as_none():
    """`date('March')` has always failed the run; `number('TBD')` does too, through a filter, as text or inside a
    list."""
    for expression in (
        "number(raw)",
        "number(raw) | float",
        "[1, number(raw)]",
        "number(raw) | string",
        "text(number(raw))",
        "number(raw) ~ ''",
    ):
        assert "not a number: 'TBD'" in failure(expression, raw="TBD")
    for expression in ("date(raw)", "date(raw) | string"):
        assert "not a date: 'March'" in failure(expression, raw="March")

    assert evaluate("number(raw)", raw="  ") is None
    assert evaluate("date(raw)", raw="") is None
    assert evaluate("date(missing)") is None


def test_the_readers_return_plain_values_a_workflow_can_serialise():
    doc = {"count": "1,234", "amount": "$586,764.00", "stored": Decimal("12.50"), "issued": "Oct 1, 2026"}
    expression = Expression(
        name="readers",
        input_fields=[NamedField(name="doc")],
        expressions=[
            ExpressionItem(key="count", expression="number(doc.count)"),
            ExpressionItem(key="amount", expression="number(doc.amount)"),
            ExpressionItem(key="stored", expression="number(doc.stored)"),
            ExpressionItem(key="issued", expression="date(doc.issued)"),
        ],
    )
    checks = Rules(
        name="vocabulary",
        input_fields=[NamedField(name="doc")],
        derived_values=[
            DerivedValue(name="amount", expression="number(doc.amount)"),
            DerivedValue(name="issued", expression="date(doc.issued)"),
        ],
        rules=[Rule(id="AMT-01", check="amount > 1000 and issued < date('2027-01-01')")],
    )

    computed = run(expression, {"doc": doc})
    checked = run(checks, {"doc": doc})

    assert computed == {"count": 1234, "amount": 586764.0, "stored": 12.5, "issued": date(2026, 10, 1)}
    assert [type(value) for value in computed.values()] == [int, float, float, date]
    assert json.loads(json.dumps(computed, cls=JsonWorkflowEncoder))["issued"] == "2026-10-01"
    assert statuses(checked) == {"AMT-01": "pass"}
    assert json.loads(json.dumps(checked, cls=JsonWorkflowEncoder))["derived"] == {
        "amount": 586764.0,
        "issued": "2026-10-01",
    }
