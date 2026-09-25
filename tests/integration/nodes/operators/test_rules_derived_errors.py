"""A derived value that cannot be computed keeps its error; one computed from a missing value is missing.

A Rules node computes its derived values once per record, before the rules run. `ltv = loan.amount /
appraisal.value` over a null appraisal is missing, as the appraisal is: None under `derived`, and a rule that reads
`ltv` is not evaluated for a missing value. Over an appraisal of 0 nothing is missing, yet there is no ratio: the
value is None under `derived` as well, with the reason under `derived_errors`, and a rule that reads it is not
evaluated, naming the reason, a rule that only asks about it too: `has(ltv)`, `ltv is present` or `ltv is none`
would be a verdict on a figure nobody computed. The value is there, so `first_present` stops at it: a fallback never
speaks over a figure the record gives but nobody could compute or read. A value computed from one nobody could read
is unreadable too, even where another value it reads is missing.
"""

import json

import pytest

from dynamiq.nodes.operators import Rules
from dynamiq.nodes.types import DerivedValue, NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus

# A key the record does not carry at all.
ABSENT = object()
UNREADABLE_NET = "not a number: 'TBD'"


def run(node: Rules, data: dict) -> dict:
    result = node.run(input_data=data, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output


def by_id(output: dict) -> dict[str, dict]:
    return {finding["rule_id"]: finding for finding in output["findings"]}


def statuses(output: dict) -> dict[str, str]:
    return {rule_id: finding["status"] for rule_id, finding in by_id(output).items()}


def lending(*checks: Rule, ltv: str = "loan.amount / appraisal.value") -> Rules:
    return Rules(
        name="lending",
        input_fields=[NamedField(name="loan"), NamedField(name="appraisal")],
        derived_values=[DerivedValue(name="ltv", expression=ltv)],
        rules=list(checks),
    )


def appraised(**values) -> dict:
    """A loan of 300,000 and an appraisal holding `values`; a value of ABSENT leaves its key out."""
    return {
        "loan": {"amount": 300000, "program": "standard"},
        "appraisal": {key: value for key, value in values.items() if value is not ABSENT},
    }


def invoicing(*checks: Rule, derived: tuple[DerivedValue, ...] = ()) -> Rules:
    return Rules(
        name="invoicing",
        input_fields=[NamedField(name="doc")],
        derived_values=[DerivedValue(name="net", expression="number(doc.net)"), *derived],
        rules=list(checks),
    )


# --- a value that cannot be computed -----------------------------------------------------------------------------


def test_a_derived_value_that_cannot_be_computed_reports_why_and_holds_every_rule_that_reads_it():
    node = lending(
        Rule(id="limit", name="LTV within limit", check="ltv <= 0.8"),
        Rule(id="guarded", name="LTV within limit when there is one", check="has(ltv) and ltv <= 0.8"),
        Rule(id="high", name="high LTV is insured", applies_when="ltv > 0.8", check="loan.insured"),
        Rule(id="present", name="LTV computed", check="ltv is present"),
    )

    output = run(node, appraised(value=0))

    assert output["derived"] == {"ltv": None}
    assert output["derived_errors"] == {"ltv": "division by zero"}
    assert statuses(output) == {
        "limit": "not_evaluated",
        "guarded": "not_evaluated",
        "high": "not_evaluated",
        "present": "not_evaluated",
    }
    assert by_id(output)["limit"]["message"] == "check could not be evaluated: division by zero"
    assert by_id(output)["limit"]["evaluated"] == {"ltv": "unreadable: division by zero"}
    # A guard asks whether there is a value, and the only answer is the error: it holds the rule rather than pass
    # or fail it, as asking whether the LTV was computed does.
    assert by_id(output)["guarded"]["message"] == "check could not be evaluated: division by zero"
    assert by_id(output)["present"]["message"] == "check could not be evaluated: division by zero"
    assert by_id(output)["high"]["message"] == "applies_when could not be evaluated: division by zero"
    assert output["status"] == "not_evaluated"
    assert json.loads(json.dumps(output)) == output


@pytest.mark.parametrize(
    "check",
    [
        "has(ltv)",
        "not has(ltv) or ltv <= 0.8",
        "ltv is not none",
        "ltv is none",
        "ltv is present",
        "ltv is blank",
        "ltv is defined",
        "ltv is number",
    ],
)
@pytest.mark.parametrize(
    ("policy", "status"),
    [(None, "not_evaluated"), ("not_applicable", "not_evaluated"), ("fail", "fail")],
    ids=["unset", "not_applicable", "fail"],
)
def test_asking_about_a_derived_value_that_could_not_be_computed_holds_the_rule_naming_the_error(check, policy, status):
    """Before derived values kept their errors a failed LTV read as None, so `has(ltv)` failed and `ltv is none`
    passed on a figure nobody computed. Now the question is answered by the error: a rule that only asks about the
    value is held as one that uses it is, and never skipped."""
    node = lending(Rule(id="asked", name="LTV asked about", check=check, on_missing=policy))

    finding = by_id(run(node, appraised(value=0)))["asked"]

    assert (finding["status"], finding["message"]) == (status, "check could not be evaluated: division by zero")


@pytest.mark.parametrize(
    ("ltv", "values"),
    [
        ("loan.amount / appraisal.value", {"value": None}),
        ("loan.amount / appraisal.value", {"value": ABSENT}),
        ("loan.amount / number(appraisal.value)", {"value": "  "}),
        ("loan.amount / appraisal.by_program[loan.program]", {"by_program": {"premium": 400000}}),
    ],
    ids=["null", "absent", "blank", "lookup-finds-nothing"],
)
def test_a_derived_value_computed_from_a_missing_value_is_missing_not_an_error(ltv, values):
    node = lending(
        Rule(id="limit", name="LTV within limit", check="ltv <= 0.8"),
        Rule(id="present", name="LTV computed", check="ltv is present"),
        ltv=ltv,
    )

    output = run(node, appraised(**values))

    assert output["derived"] == {"ltv": None}
    assert output["derived_errors"] == {}
    assert statuses(output) == {"limit": "not_evaluated", "present": "fail"}
    assert by_id(output)["limit"]["message"] == "missing value for ltv"


def test_every_output_reports_derived_errors_and_a_value_that_computed_reads_as_before():
    computed = run(lending(Rule(id="limit", name="LTV within limit", check="ltv <= 0.8")), appraised(value=400000))
    plain = run(
        Rules(name="plain", input_fields=[NamedField(name="loan")], rules=[Rule(id="amount", check="loan.amount > 0")]),
        {"loan": {"amount": 1}},
    )

    assert computed["derived"] == {"ltv": 0.75}
    assert computed["derived_errors"] == {}
    assert statuses(computed) == {"limit": "pass"}
    assert plain["derived"] == {} and plain["derived_errors"] == {}


def test_a_member_the_expression_found_missing_stays_none_inside_what_it_built():
    """A list or a dict with a missing member is a value that computed, the missing member None inside it."""
    node = Rules(
        name="discounts",
        input_fields=[NamedField(name="items")],
        derived_values=[
            DerivedValue(name="discounts", expression="items | map(attribute='discount') | list"),
            DerivedValue(name="pair", expression="{'first': items[0].discount, 'second': items[1].discount}"),
        ],
        rules=[Rule(id="any", name="has discounts", check="discounts | length > 0")],
    )

    output = run(node, {"items": [{"discount": 5}, {"sku": "x"}]})

    assert output["derived"] == {"discounts": [5, None], "pair": {"first": 5, "second": None}}
    assert output["derived_errors"] == {}
    assert statuses(output) == {"any": "pass"}


# --- a value nobody could read ----------------------------------------------------------------------------------


def test_an_amount_nobody_could_read_stays_unreadable_for_the_rules_so_no_fallback_speaks_over_it():
    node = invoicing(
        Rule(id="net", name="net amount is positive", check="net > 0"),
        Rule(id="fallback", name="an amount is positive", check="first_present(net, number(doc.gross)) > 0"),
    )

    output = run(node, {"doc": {"net": "TBD", "gross": "5"}})

    assert output["derived"] == {"net": None}
    assert output["derived_errors"] == {"net": UNREADABLE_NET}
    assert statuses(output) == {"net": "not_evaluated", "fallback": "not_evaluated"}
    assert by_id(output)["net"]["message"] == f"check could not be evaluated: {UNREADABLE_NET}"
    # The gross amount reads, but the record gives a net amount: the fallback never passes over it.
    assert by_id(output)["fallback"]["message"] == f"check could not be evaluated: {UNREADABLE_NET}"
    assert by_id(output)["fallback"]["evaluated"] == {"net": f"unreadable: {UNREADABLE_NET}", "doc.gross": "5"}
    assert json.loads(json.dumps(output)) == output

    # A net amount the record does not give at all is missing, and the fallback stands in for it.
    without_net = run(node, {"doc": {"gross": "5"}})

    assert without_net["derived"] == {"net": None} and without_net["derived_errors"] == {}
    assert statuses(without_net) == {"net": "not_evaluated", "fallback": "pass"}
    assert by_id(without_net)["net"]["message"] == "missing value for net"


def test_a_message_prints_the_text_the_record_holds_for_a_derived_value_nobody_could_read():
    node = invoicing(
        Rule(
            id="gross",
            name="gross amount over the floor",
            check="number(doc.gross) > 100",
            message="Gross {{ doc.gross }} with net {{ net }}",
        )
    )

    output = run(node, {"doc": {"net": "TBD", "gross": "5"}})

    assert statuses(output) == {"gross": "fail"}
    assert by_id(output)["gross"]["message"] == "Gross 5 with net TBD"


@pytest.mark.parametrize(
    ("message", "shown"),
    [
        ("Amount {{ loan.amount }}{% if has(ltv) %}, LTV {{ ltv }}{% endif %}", "Amount 5000"),
        ("Amount {{ loan.amount }}{% if ltv is defined %}, LTV {{ ltv }}{% endif %}", "Amount 5000, LTV None"),
        ("Amount {{ loan.amount }}{% if ltv is not none %}, LTV {{ ltv }}{% endif %}", "Amount 5000"),
        ("Amount {{ loan.amount }}{% if ltv %}, LTV {{ ltv }}{% endif %}", "Amount 5000"),
        ("Amount {{ loan.amount }}, LTV {{ ltv | default('n/a', true) }}", "Amount 5000, LTV n/a"),
    ],
    ids=["has", "is-defined", "is-not-none", "truth", "default"],
)
def test_a_message_reads_a_derived_value_that_could_not_be_computed_as_it_always_did(message, shown):
    """Before derived values kept their errors, a ratio divided by zero was None, and a message guarded it; the
    message reads None there still, so its guard decides rather than the message coming back as written. Each
    message is the one origin/main renders."""
    node = Rules(
        name="lending",
        input_fields=[NamedField(name="loan")],
        derived_values=[DerivedValue(name="ltv", expression="loan.a / loan.v")],
        rules=[Rule(id="small", name="small loan", check="loan.amount < 1000", message=message)],
    )

    output = run(node, {"loan": {"amount": 5000, "a": 1, "v": 0}})

    assert output["derived_errors"] == {"ltv": "division by zero"}
    assert (by_id(output)["small"]["status"], by_id(output)["small"]["message"]) == ("fail", shown)


@pytest.mark.parametrize(
    "message",
    [
        "Net {{ d }}",
        "Net {% if has(d) %}{{ d }}{% else %}not given{% endif %}",
        # A value the message itself reads through `number()` prints the same way.
        "Net {{ number(doc.x) }}",
    ],
    ids=["printed", "guarded", "read-in-the-message"],
)
def test_a_message_reads_a_derived_value_nobody_could_read_as_the_text_the_record_holds(message):
    node = Rules(
        name="invoicing",
        input_fields=[NamedField(name="doc")],
        derived_values=[DerivedValue(name="d", expression="number(doc.x)")],
        rules=[Rule(id="gross", name="gross over the floor", check="doc.gross > 100", message=message)],
    )

    output = run(node, {"doc": {"x": "TBD", "gross": 5}})

    assert output["derived_errors"] == {"d": UNREADABLE_NET}
    assert (by_id(output)["gross"]["status"], by_id(output)["gross"]["message"]) == ("fail", "Net TBD")


def test_a_derived_value_computed_from_one_that_could_not_be_is_an_error_too_and_a_missing_one_stays_missing():
    node = invoicing(
        Rule(id="tax", name="tax is positive", check="tax > 0"),
        Rule(id="gross", name="gross is positive", check="gross > 0"),
        derived=(
            DerivedValue(name="tax", expression="net * 0.2"),
            DerivedValue(name="amounts", expression="[net, number(doc.gross)]"),
            DerivedValue(name="gross", expression="number(doc.gross)"),
        ),
    )

    unreadable = run(node, {"doc": {"net": "TBD", "gross": "5"}})
    missing = run(node, {"doc": {"gross": "5"}})

    assert unreadable["derived"] == {"net": None, "tax": None, "amounts": None, "gross": 5}
    assert unreadable["derived_errors"] == {"net": UNREADABLE_NET, "tax": UNREADABLE_NET, "amounts": UNREADABLE_NET}
    assert statuses(unreadable) == {"tax": "not_evaluated", "gross": "pass"}
    assert by_id(unreadable)["tax"]["message"] == f"check could not be evaluated: {UNREADABLE_NET}"
    assert json.loads(json.dumps(unreadable)) == unreadable

    assert missing["derived"] == {"net": None, "tax": None, "amounts": [None, 5], "gross": 5}
    assert missing["derived_errors"] == {}
    assert by_id(missing)["tax"]["message"] == "missing value for tax"


@pytest.mark.parametrize(
    ("tax", "rate"),
    [
        ("net * doc.rate", ABSENT),
        ("doc.rate * net", ABSENT),
        ("number(doc.rate) * net", "  "),
        ("[doc.rate, net]", ABSENT),
        ("number(doc.net) * doc.rate", ABSENT),
    ],
    ids=["unreadable-first", "missing-first", "blank-first", "in-a-list", "read-inline"],
)
def test_a_value_nobody_could_read_makes_a_derived_value_unreadable_even_beside_a_missing_one(tax, rate):
    """Whichever raises first, the missing rate or the unreadable net, the tax is unreadable: the missing rate
    does not hide a net amount the record gives but nobody could read."""
    node = invoicing(
        Rule(id="tax", name="tax is positive", check="tax > 0"),
        derived=(DerivedValue(name="tax", expression=tax),),
    )
    doc = {"net": "TBD"} if rate is ABSENT else {"net": "TBD", "rate": rate}

    output = run(node, {"doc": doc})

    assert output["derived"] == {"net": None, "tax": None}
    assert output["derived_errors"] == {"net": UNREADABLE_NET, "tax": UNREADABLE_NET}
    assert statuses(output) == {"tax": "not_evaluated"}
    assert by_id(output)["tax"]["message"] == f"check could not be evaluated: {UNREADABLE_NET}"
    assert json.loads(json.dumps(output)) == output


# --- paths and names ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("check", "path"),
    [
        ("ltv.value > 0", "ltv.value"),
        ("ltv.reason == 'division by zero'", "ltv.reason"),
        ("ltv.band == 'A'", "ltv.band"),
        ("ltv[0] > 0", "ltv[0]"),
    ],
    ids=["value", "reason", "member", "index"],
)
def test_a_path_through_a_derived_value_that_could_not_be_computed_names_the_error(check, path):
    """The marker that holds the error has members of its own; a path through it reads none of them, and reads
    nothing missing either."""
    output = run(lending(Rule(id="path", name="path", check=check)), appraised(value=0))

    finding = by_id(output)["path"]
    assert finding["status"] == "not_evaluated"
    assert finding["message"] == "check could not be evaluated: division by zero"
    assert finding["evaluated"] == {path: "unreadable: division by zero"}


def test_a_derived_value_that_reads_and_calls_the_same_name_reports_the_clash():
    """An expression over an input called `text` that also calls `text()` builds, and the clash is its error."""
    node = Rules(
        name="labels",
        input_fields=[NamedField(name="text")],
        derived_values=[DerivedValue(name="label", expression="text(text) | upper")],
        rules=[Rule(id="label", name="label reads X", check="label == 'X'")],
    )

    output = run(node, {"text": "x"})

    clash = "Rules 'labels': derived value 'label' reads 'text' as a value and calls it as a helper"
    assert output["derived"] == {"label": None}
    assert output["derived_errors"] == {"label": clash}
    assert by_id(output)["label"]["message"] == f"check could not be evaluated: {clash}"
