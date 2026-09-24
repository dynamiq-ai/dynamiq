"""The words a rule is written with: `is present` and `is blank`, `text()` and `first_present()`.

A record from a form or an extraction says "no answer" in more ways than a missing key: a null, an empty string, a
string of spaces, an empty list. These words read every one of them as missing, so a rule stays as short as the
policy it encodes and never passes or fails on a value nobody gave: `text(app.purpose) == 'purchase'` is not
evaluated for a blank purpose, where `app.purpose | trim == 'purchase'` would fail it.
"""

import pytest
from jinja2.exceptions import UndefinedError

from dynamiq.nodes.operators import Expression, Rules
from dynamiq.nodes.operators.rules import MissingValue, read_paths
from dynamiq.nodes.types import DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus

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


# --- first_present() --------------------------------------------------------------------------------------------


def test_first_present_reads_each_value_leniently_and_guards_no_other_read():
    """`has(a.b)` lets a read under `a.b` be missing as well; a fallback is no question about `a.b`."""
    reads = read_paths("first_present(a.b, c) == 1 and a.b.c > 0")
    assert (reads.required, reads.optional) == (["a.b.c"], ["a.b", "c"])

    # A value read as a fallback and on its own as well is required.
    reads = read_paths("first_present(a.b, c) == 1 and a.b > 0")
    assert (reads.required, reads.optional) == (["a.b"], ["c"])

    # A helper inside the call reads leniently too, so `text()` can hand an absent value on to be skipped.
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
        "missing_lowered": "",
        "replaced": " Eda ",
        "shown": " Ada ",
        "nothing": None,
        "asked": True,
    }
