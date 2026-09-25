"""A rule says what a missing value means for it: `on_missing` on the rule overrides the node's.

`not_applicable` skips the rule for a record that does not carry a value it reads, as `applies_when` would, so the
rule needs no presence guard: `shipment.weight_kg <= 30` does not apply to a shipment nobody weighed, and a record
whose only unmet rules were skipped still passes. Only data the record lacks is skipped, and only where a read names
it: a blank no read accounts for, from a lookup inside `text()` that found nothing say, is not. Nor is a check that
reads a value nobody could read, calls a name no helper has, finds a value missing under a name a node with declared
inputs does not declare (a typo), or needs a derived value a lookup found nothing for, whatever else the record
lacks. Each is a problem to fix, not data to wait for: the rule is not evaluated, or reports its severity under
`fail`, and a rule set to skip says why it did not ("… (not skipped: lon is not an input or a derived value)"). A
derived value that came out missing counts as data the record lacks exactly when the same expression, written in the
check, would. What a skipped rule cannot see is an error its check would raise on the values that are there; that
surfaces on the records that carry the missing value. A rule's `on_missing` that is no policy at all leaves the
choice to the node, with a warning, rather than refusing the build.
"""

import json
import logging
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, NamedTuple

import pytest
import yaml
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.operators import Rules
from dynamiq.nodes.types import DerivedValue, NamedField, Rule, RuleMissingPolicy
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger

LENDING = ("loan", "appraisal", "limits", "rates")
LIMIT = ("limit", "limits[loan.program]")
LTV = ("ltv", "loan.amount / appraisal.value")
LIMITS = {"standard": 500000}


def run(node: Rules, data: dict) -> dict:
    result = node.run(input_data=data, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return result.output


def by_id(output: dict) -> dict[str, dict]:
    return {finding["rule_id"]: finding for finding in output["findings"]}


def outcome(output: dict, rule_id: str = "rule") -> tuple[str, str | None]:
    finding = by_id(output)[rule_id]
    return finding["status"], finding["message"]


def screening(
    check: str,
    policy: str | None = None,
    *,
    inputs: tuple[str, ...] = (),
    derived: tuple[tuple[str, str], ...] = (),
    applies_when: str | None = None,
    node_policy: str = "not_evaluated",
) -> Rules:
    """A node with one `warn` rule, `rule`, reading the named inputs and derived values."""
    return Rules(
        name="screening",
        input_fields=[NamedField(name=name) for name in inputs],
        derived_values=[DerivedValue(name=name, expression=expression) for name, expression in derived],
        on_missing=node_policy,
        rules=[
            Rule(
                id="rule",
                name="the rule",
                severity="warn",
                applies_when=applies_when,
                check=check,
                on_missing=policy,
            )
        ],
    )


# --- data the record lacks is skipped ----------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [{}, {"weight_kg": None}], ids=["absent", "null"])
def test_a_rule_set_to_skip_missing_data_does_not_apply_to_a_record_without_the_value(weight):
    node = Rules(
        name="shipping",
        input_fields=[NamedField(name="shipment")],
        rules=[
            Rule(
                id="weight",
                name="weight within the carrier limit",
                check="shipment.weight_kg <= 30",
                on_missing="not_applicable",
            ),
            Rule(id="destination", name="destination given", check="shipment.country is present"),
        ],
    )

    output = run(node, {"shipment": {"country": "DE", **weight}})

    assert outcome(output, "weight") == ("not_applicable", "does not apply: missing value for shipment.weight_kg")
    assert outcome(output, "destination") == ("pass", None)
    assert output["summary"] == {
        "pass": 1,
        "fail": 0,
        "warn": 0,
        "info": 0,
        "not_applicable": 1,
        "not_evaluated": 0,
    }
    # The skipped rule counts as having run: a record whose only unmet rule lacked its data still passes.
    assert output["status"] == "pass"
    # With the value the rule decides as any rule does.
    weighed = run(node, {"shipment": {"country": "DE", "weight_kg": 42}})
    assert outcome(weighed, "weight") == ("fail", None)
    assert weighed["status"] == "fail"


@pytest.mark.parametrize(
    ("policy", "expected", "status"),
    [
        (None, ("not_evaluated", "missing value for shipment.weight_kg"), "not_evaluated"),
        ("not_evaluated", ("not_evaluated", "missing value for shipment.weight_kg"), "not_evaluated"),
        ("fail", ("warn", "missing value for shipment.weight_kg"), "warn"),
        ("not_applicable", ("not_applicable", "does not apply: missing value for shipment.weight_kg"), "pass"),
    ],
)
def test_each_policy_reports_a_missing_value_its_own_way(policy, expected, status):
    output = run(screening("shipment.weight_kg <= 30", policy, inputs=("shipment",)), {"shipment": {}})

    assert outcome(output) == expected
    assert output["status"] == status


class Skipped(NamedTuple):
    node: Callable[[str | None], Rules]
    record: dict
    missing: str


SKIPPED = {
    "blank-text": Skipped(
        lambda policy: screening("text(app.purpose) == 'purchase'", policy, inputs=("app",)),
        {"app": {"purpose": "  "}},
        "app.purpose",
    ),
    "blank-number": Skipped(
        lambda policy: screening("number(doc.amount) > 1000", policy, inputs=("doc",)),
        {"doc": {"amount": ""}},
        "doc.amount",
    ),
    "nothing-present": Skipped(
        lambda policy: screening("first_present(order.coupon, order.promo) == 'SPRING'", policy, inputs=("order",)),
        {"order": {"promo": " "}},
        "order.coupon",
    ),
    "derived-from-absent-data": Skipped(
        lambda policy: screening("ltv <= 0.8", policy, inputs=LENDING, derived=(LTV,)),
        {"loan": {"amount": 300000}, "appraisal": {}},
        "ltv",
    ),
    "derived-lookup-by-an-absent-key": Skipped(
        lambda policy: screening("loan.amount <= limit", policy, inputs=LENDING, derived=(LIMIT,)),
        {"loan": {"amount": 300000}, "limits": LIMITS},
        "limit",
    ),
    "derived-blank-text-of-a-lookup-by-a-blank-key": Skipped(
        lambda policy: screening(
            "grade == 'A'", policy, inputs=LENDING, derived=(("grade", "text(limits[loan.program])"),)
        ),
        {"loan": {"program": " "}, "limits": {"standard": "A"}},
        "grade",
    ),
    "a-derived-value-name": Skipped(
        lambda policy: screening("net > 0", policy, inputs=("doc",), derived=(("net", "number(doc.net)"),)),
        {"doc": {}},
        "net",
    ),
    "nothing-declared": Skipped(
        lambda policy: screening("order.total > 100", policy),
        {"order": {}},
        "order.total",
    ),
    "in-applies-when": Skipped(
        lambda policy: screening(
            "docs.flood_cert is present",
            policy,
            inputs=("property", "docs"),
            applies_when="property.flood_zone in ['A', 'V']",
        ),
        {"property": {}, "docs": {}},
        "property.flood_zone",
    ),
    "beside-a-number-read-as-written": Skipped(
        lambda policy: screening("number(doc.amount) > doc.limit", policy, inputs=("doc",)),
        {"doc": {"amount": "$1,500.00"}},
        "doc.limit",
    ),
    # Without declared inputs no name can be told from a typo, whatever derived values the node computes.
    "derived-values-without-inputs": Skipped(
        lambda policy: screening("order.total > 100", policy, derived=(("vip", "order.tier == 'gold'"),)),
        {"order": {"tier": "gold"}},
        "order.total",
    ),
    "derived-from-absent-data-without-inputs": Skipped(
        lambda policy: screening("ltv <= 0.8", policy, derived=(LTV,)),
        {"loan": {"amount": 300000}, "appraisal": {}},
        "ltv",
    ),
    # The check only falls back on the lookup that found nothing, and it lacks the amount.
    "a-lookup-it-only-falls-back-on": Skipped(
        lambda policy: screening(
            "loan.amount <= first_present(limit, 500000)", policy, inputs=LENDING, derived=(LIMIT,)
        ),
        {"loan": {"program": "jumbo"}, "limits": LIMITS},
        "loan.amount",
    ),
    "as-of-not-supplied": Skipped(
        lambda policy: screening("days_between(loan.closed, as_of) <= 30", policy, inputs=LENDING),
        {"loan": {"closed": "2026-01-01"}},
        "as_of",
    ),
    # A derived value missing only because an earlier one's lookup found nothing is a lookup miss too.
    "derived-lookup-miss-beside-good-data-in-a-fallback": Skipped(
        lambda policy: screening(
            "first_present(z, 0) < loan.cap", policy, inputs=LENDING, derived=(LIMIT, ("z", "limit + loan.amount"))
        ),
        {"loan": {"program": "jumbo", "amount": 5}, "limits": LIMITS},
        "loan.cap",
    ),
    # A fallback stands in for a derived value whose lookup found nothing; the cap is what the record lacks.
    "derived-lookup-in-a-fallback": Skipped(
        lambda policy: screening(
            "first_present(doubled, 0) < loan.cap",
            policy,
            inputs=LENDING,
            derived=(("doubled", "limits[loan.program]"),),
        ),
        {"loan": {"program": "jumbo"}, "limits": LIMITS},
        "loan.cap",
    ),
}


@pytest.mark.parametrize("case", SKIPPED.values(), ids=SKIPPED.keys())
def test_data_the_record_lacks_is_skipped_by_a_rule_set_to_skip_it(case):
    output = run(case.node("not_applicable"), case.record)

    assert outcome(output) == ("not_applicable", f"does not apply: missing value for {case.missing}")
    assert output["status"] == "pass"
    assert outcome(run(case.node(None), case.record)) == ("not_evaluated", f"missing value for {case.missing}")
    assert outcome(run(case.node("fail"), case.record)) == ("warn", f"missing value for {case.missing}")


# --- a mistake or a value nobody could read is never skipped -----------------------------------------------------


# A loan whose program the limits table lacks, so `limit` is a lookup that found nothing; the cap is absent too.
UNLISTED = {"loan": {"program": "jumbo", "amount": 5}, "limits": LIMITS}


def falling_back_on_z(
    z: str, *, before: tuple[tuple[str, str], ...] = (), after: tuple[tuple[str, str], ...] = ()
) -> Callable[..., Rules]:
    """A node computing `limit` and `z`, with derived values before and after `z`, and one rule that only falls
    back on `z`, beside a cap the record lacks."""
    return lambda **policy: screening(
        "first_present(z, 0) < loan.cap", inputs=LENDING, derived=(LIMIT, *before, ("z", z), *after), **policy
    )


class Held(NamedTuple):
    node: Callable[..., Rules]
    record: dict
    reason: str
    # What a rule set to skip missing data adds to the reason: why this one was not skipped. None where the reason is
    # already an error that names its cause.
    why: str | None = None


HELD = {
    "unreadable-value": Held(
        lambda **policy: screening("number(doc.amount) > 1000", inputs=("doc",), **policy),
        {"doc": {"amount": "TBD"}},
        "check could not be evaluated: not a number: 'TBD'",
    ),
    "unreadable-in-applies-when": Held(
        lambda **policy: screening("doc.approved", inputs=("doc",), applies_when="number(doc.amount) > 1000", **policy),
        {"doc": {"amount": "TBD"}},
        "applies_when could not be evaluated: not a number: 'TBD'",
    ),
    "lookup-in-the-check": Held(
        lambda **policy: screening("loan.amount <= limits[loan.program]", inputs=LENDING, **policy),
        {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS},
        "check could not be evaluated: 'dict object' has no attribute 'jumbo'",
    ),
    # A blank with no path: the lookup inside `text()` found nothing, and no value the check reads is blank.
    "lookup-inside-text": Held(
        lambda **policy: screening(
            "text(categories[claim.code]) == 'dental'", inputs=("claim", "categories"), **policy
        ),
        {"claim": {"code": "D9"}, "categories": {"D1": "dental"}},
        "missing value: text() found no text",
        "no field of the record is named",
    ),
    "derived-lookup": Held(
        lambda **policy: screening("loan.amount <= limit", inputs=LENDING, derived=(LIMIT,), **policy),
        {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS},
        "missing value for limit",
        "the lookup for limit found nothing",
    ),
    "derived-lookup-in-applies-when": Held(
        lambda **policy: screening(
            "loan.insured", inputs=LENDING, derived=(LIMIT,), applies_when="loan.amount > limit", **policy
        ),
        {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS},
        "missing value for limit",
        "the lookup for limit found nothing",
    ),
    "derived-lookup-by-a-blank-key": Held(
        lambda **policy: screening("loan.amount <= limit", inputs=LENDING, derived=(LIMIT,), **policy),
        {"loan": {"amount": 300000, "program": ""}, "limits": LIMITS},
        "missing value for limit",
        "the lookup for limit found nothing",
    ),
    "derived-lookup-through-number": Held(
        lambda **policy: screening(
            "loan.rate <= rate", inputs=LENDING, derived=(("rate", "number(rates[loan.program])"),), **policy
        ),
        {"loan": {"rate": 7, "program": "jumbo"}, "rates": {"standard": "6.5"}},
        "missing value for rate",
        "the lookup for rate found nothing",
    ),
    "derived-from-a-derived-lookup": Held(
        lambda **policy: screening(
            "headroom >= 0",
            inputs=LENDING,
            derived=(LIMIT, ("headroom", "limit - loan.amount")),
            **policy,
        ),
        {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS},
        "missing value for headroom",
        "the lookup for limit found nothing",
    ),
    "derived-lookup-beside-missing-data": Held(
        lambda **policy: screening("loan.exempt or loan.amount <= limit", inputs=LENDING, derived=(LIMIT,), **policy),
        {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS},
        "missing value for loan.exempt",
        "the lookup for limit found nothing",
    ),
    # No input is declared, so only the call itself tells the mistyped helper from data the record lacks.
    "mistyped-helper": Held(
        lambda **policy: screening("firstpresent(order.coupon, order.promo) == 'SPRING'", **policy),
        {"order": {"coupon": "SPRING"}},
        "missing value for firstpresent",
        "firstpresent is not a helper",
    ),
    "mistyped-helper-beside-missing-data": Held(
        lambda **policy: screening(
            "order.total > 100 and firstpresent(order.coupon) == 'SPRING'", inputs=("order",), **policy
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for order.total",
        "firstpresent is not a helper",
    ),
    "derived-mistyped-helper": Held(
        lambda **policy: screening(
            "code == 'SPRING'", inputs=("order",), derived=(("code", "firstpresent(order.coupon)"),), **policy
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for code",
        "firstpresent is not a helper",
    ),
    "derived-error": Held(
        lambda **policy: screening("ltv <= 0.8", inputs=LENDING, derived=(LTV,), **policy),
        {"loan": {"amount": 300000}, "appraisal": {"value": 0}},
        "check could not be evaluated: division by zero",
    ),
    # The missing limit must not hide the ratio nobody could compute.
    "unreadable-beside-missing-data": Held(
        lambda **policy: screening("ltv <= appraisal.max_ltv", inputs=LENDING, derived=(LTV,), **policy),
        {"loan": {"amount": 300000}, "appraisal": {"value": 0}},
        "check could not be evaluated: division by zero",
    ),
    "unreadable-beside-missing-data-in-applies-when": Held(
        lambda **policy: screening(
            "doc.approved",
            inputs=("doc",),
            derived=(("net", "number(doc.net)"),),
            applies_when="net > doc.threshold",
            **policy,
        ),
        {"doc": {"net": "TBD"}},
        "applies_when could not be evaluated: not a number: 'TBD'",
    ),
    # The check reads the text itself through `number()` or `date()`: the missing value stops it before the reader
    # runs, and must not hide text nobody could read. The finding still names the missing value.
    "unreadable-number-beside-missing-data": Held(
        lambda **policy: screening("number(doc.amount) > doc.limit", inputs=("doc",), **policy),
        {"doc": {"amount": "TBD"}},
        "missing value for doc.limit",
        "doc.amount is not a number: 'TBD'",
    ),
    "unreadable-date-beside-missing-data": Held(
        lambda **policy: screening("date(doc.issued, format='%d.%m.%Y') <= date(doc.due)", inputs=("doc",), **policy),
        {"doc": {"issued": "March"}},
        "missing value for doc.due",
        "doc.issued is not a date: 'March' (format '%d.%m.%Y')",
    ),
    "impossible-date-beside-missing-data": Held(
        lambda **policy: screening("date(doc.issued) <= date(doc.due)", inputs=("doc",), **policy),
        {"doc": {"issued": "2026-02-30"}},
        "missing value for doc.due",
        "doc.issued is unreadable: day is out of range for month",
    ),
    "unreadable-date-in-days-between": Held(
        lambda **policy: screening("days_between(doc.opened, doc.closed) <= 30", inputs=("doc",), **policy),
        {"doc": {"opened": "March"}},
        "missing value for doc.closed",
        "doc.opened is not a date: 'March'",
    ),
    "unreadable-number-as-a-fallback": Held(
        lambda **policy: screening("first_present(number(doc.net), 0) > doc.limit", inputs=("doc",), **policy),
        {"doc": {"net": "TBD"}},
        "missing value for doc.limit",
        "doc.net is not a number: 'TBD'",
    ),
    "number-told-a-decimal-it-cannot-read": Held(
        lambda **policy: screening("number(doc.amount, decimal=';') > doc.limit", inputs=("doc",), **policy),
        {"doc": {"amount": "5"}},
        "missing value for doc.limit",
        "doc.amount could not be read: number() reads a decimal point '.' or a decimal comma ',', not ';'",
    ),
    "derived-unreadable-inline": Held(
        lambda **policy: screening(
            "tax > 0", inputs=("doc",), derived=(("tax", "doc.rate * number(doc.net)"),), **policy
        ),
        {"doc": {"net": "TBD"}},
        "missing value for tax",
        "doc.net is not a number: 'TBD'",
    ),
    "derived-unreadable-fallback": Held(
        lambda **policy: screening(
            "tax > 0",
            inputs=("doc",),
            derived=(("net", "number(doc.net)"), ("tax", "doc.rate * first_present(net, 0)")),
            **policy,
        ),
        {"doc": {"net": "TBD"}},
        "missing value for tax",
        "net is not a number: 'TBD'",
    ),
    "derived-unreadable-ratio-fallback": Held(
        lambda **policy: screening(
            "share > 0", inputs=LENDING, derived=(LTV, ("share", "loan.fee * first_present(ltv, 0)")), **policy
        ),
        {"loan": {"amount": 300000}, "appraisal": {"value": 0}},
        "missing value for share",
        "ltv is unreadable: division by zero",
    ),
    "typo": Held(
        lambda **policy: screening("lon.amount > 0", inputs=("loan",), **policy),
        {"loan": {"amount": 5}},
        "missing value for lon.amount",
        "lon is not an input or a derived value",
    ),
    "typo-beside-missing-data": Held(
        lambda **policy: screening("loan.amount <= 500000 or lon.exempt", inputs=("loan",), **policy),
        {"loan": {}},
        "missing value for loan.amount",
        "lon is not an input or a derived value",
    ),
    "derived-typo": Held(
        lambda **policy: screening("doubled > 0", inputs=("loan",), derived=(("doubled", "lon.amount * 2"),), **policy),
        {"loan": {"amount": 5}},
        "missing value for doubled",
        "lon is not an input or a derived value",
    ),
    # A derived value's own defect holds a rule wherever the rule reads it, a fallback included, beside a value the
    # record does lack; only a lookup that found nothing gives way to a fallback.
    "derived-typo-in-a-fallback": Held(
        lambda **policy: screening(
            "first_present(doubled, 0) < loan.cap", inputs=("loan",), derived=(("doubled", "lon.amount * 2"),), **policy
        ),
        {"loan": {"amount": 5}},
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-mistyped-helper-in-a-fallback": Held(
        lambda **policy: screening(
            "first_present(code, 'NONE') == order.expected",
            inputs=("order",),
            derived=(("code", "firstpresent(order.coupon)"),),
            **policy,
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for order.expected",
        "firstpresent is not a helper",
    ),
    "derived-unreadable-in-a-fallback": Held(
        lambda **policy: screening(
            "first_present(tax, 0) > doc.limit",
            inputs=("doc",),
            derived=(("tax", "doc.rate * number(doc.net)"),),
            **policy,
        ),
        {"doc": {"net": "TBD"}},
        "missing value for doc.limit",
        "doc.net is not a number: 'TBD'",
    ),
    # A defect of the derived value's own expression speaks over a lookup that found nothing in it, whichever the
    # expression reads first, so a rule that only falls back on the value is still held.
    "derived-typo-after-a-lookup-miss": Held(
        falling_back_on_z("limit + lon.amount"),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-typo-before-a-lookup-miss": Held(
        falling_back_on_z("lon.amount + limit"),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-fallback-typo-after-a-lookup-miss": Held(
        falling_back_on_z("limit + first_present(lon.amount, 0)"),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-fallback-typo-before-a-lookup-miss": Held(
        falling_back_on_z("first_present(lon.amount, 0) + limit"),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-defect-after-a-lookup-miss": Held(
        falling_back_on_z("limit + doubled", before=(("doubled", "lon.amount * 2"),)),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-defect-before-a-lookup-miss": Held(
        falling_back_on_z("doubled + limit", before=(("doubled", "lon.amount * 2"),)),
        UNLISTED,
        "missing value for loan.cap",
        "lon is not an input or a derived value",
    ),
    "derived-later-value-after-a-lookup-miss": Held(
        falling_back_on_z("limit + later", after=(("later", "loan.amount"),)),
        UNLISTED,
        "missing value for loan.cap",
        "later is computed after z",
    ),
    "derived-later-value-before-a-lookup-miss": Held(
        falling_back_on_z("later + limit", after=(("later", "loan.amount"),)),
        UNLISTED,
        "missing value for loan.cap",
        "later is computed after z",
    ),
    # Nothing the derived value reads is missing, yet its expression calls a name no helper has: a defect, not a
    # lookup that found nothing, wherever the call sits.
    "derived-mistyped-helper-inside-its-own-fallback": Held(
        lambda **policy: screening(
            "first_present(code, 'X') == order.expected",
            inputs=("order",),
            derived=(("code", "first_present(firstpresent(order.coupon), 'NONE')"),),
            **policy,
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for order.expected",
        "firstpresent is not a helper",
    ),
    "derived-mistyped-helper-under-default": Held(
        lambda **policy: screening(
            "first_present(code, 'X') == order.expected",
            inputs=("order",),
            derived=(("code", "firstpresent(order.coupon) | default('NONE')"),),
            **policy,
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for order.expected",
        "firstpresent is not a helper",
    ),
    "derived-mistyped-helper-inside-its-own-fallback-needed": Held(
        lambda **policy: screening(
            "code == order.expected",
            inputs=("order",),
            derived=(("code", "first_present(firstpresent(order.coupon), 'NONE')"),),
            **policy,
        ),
        {"order": {"coupon": "SPRING"}},
        "missing value for code",
        "firstpresent is not a helper",
    ),
    "derived-read-of-a-later-derived-value-without-inputs": Held(
        lambda **policy: screening(
            "doubled > 0", derived=(("doubled", "amount * 2"), ("amount", "loan.amount")), **policy
        ),
        {"loan": {"amount": 5}},
        "missing value for doubled",
        "amount is computed after doubled",
    ),
    "derived-read-of-a-later-derived-value": Held(
        lambda **policy: screening(
            "doubled > 0",
            inputs=("loan",),
            derived=(("doubled", "amount * 2"), ("amount", "loan.amount")),
            **policy,
        ),
        {"loan": {"amount": 5}},
        "missing value for doubled",
        "amount is computed after doubled",
    ),
}
# Every way a rule or its node can say "hold" or leave it to the default, and every way it can say "skip": none of
# them skips any of these.
HOLDING = [{"policy": None}, {"policy": "not_evaluated"}]
SKIPPING = [
    {"policy": "not_applicable"},
    {"policy": None, "node_policy": "not_applicable"},
    {"policy": "not_applicable", "node_policy": "fail"},
]


def policy_id(policy: dict) -> str:
    return "-".join(str(value) for value in policy.values())


@pytest.mark.parametrize("policy", HOLDING, ids=policy_id)
@pytest.mark.parametrize("case", HELD.values(), ids=HELD.keys())
def test_a_mistake_or_a_value_nobody_could_read_is_not_evaluated_as_it_was(case, policy):
    output = run(case.node(**policy), case.record)

    assert outcome(output) == ("not_evaluated", case.reason)
    assert output["status"] == "not_evaluated"


@pytest.mark.parametrize("policy", SKIPPING, ids=policy_id)
@pytest.mark.parametrize("case", HELD.values(), ids=HELD.keys())
def test_a_mistake_or_a_value_nobody_could_read_is_never_skipped_and_the_finding_says_why(case, policy):
    output = run(case.node(**policy), case.record)

    reason = case.reason if case.why is None else f"{case.reason} (not skipped: {case.why})"
    assert outcome(output) == ("not_evaluated", reason)
    assert output["status"] == "not_evaluated"


@pytest.mark.parametrize("policy", [{"policy": "fail"}, {"policy": "fail", "node_policy": "not_applicable"}])
@pytest.mark.parametrize("case", HELD.values(), ids=HELD.keys())
def test_a_mistake_or_a_value_nobody_could_read_reports_the_severity_under_fail(case, policy):
    output = run(case.node(**policy), case.record)

    assert outcome(output) == ("warn", case.reason)
    assert output["status"] == "warn"


def test_a_lookup_that_found_nothing_is_judged_record_by_record():
    node = screening("loan.amount <= limit", "not_applicable", inputs=LENDING, derived=(LIMIT,))

    unlisted = run(node, {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS})
    unknown = run(node, {"loan": {"amount": 300000}, "limits": LIMITS})
    listed = run(node, {"loan": {"amount": 300000, "program": "standard"}, "limits": LIMITS})
    again = run(node, {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS})

    assert outcome(unlisted) == (
        "not_evaluated",
        "missing value for limit (not skipped: the lookup for limit found nothing)",
    )
    assert outcome(unknown) == ("not_applicable", "does not apply: missing value for limit")
    assert outcome(listed) == ("pass", None)
    assert outcome(again) == outcome(unlisted)
    # The output reports the value as missing either way: only whether a rule may skip it differs.
    assert unlisted["derived"] == unknown["derived"] == {"limit": None}
    assert unlisted["derived_errors"] == unknown["derived_errors"] == {}


@pytest.mark.parametrize(
    ("check", "inputs", "lacking", "carrying", "error"),
    [
        (
            "loan.amount / appraisal.value <= appraisal.max_ltv",
            LENDING,
            {"loan": {"amount": 300000}, "appraisal": {"value": 0}},
            {"loan": {"amount": 300000}, "appraisal": {"value": 0, "max_ltv": 0.8}},
            "division by zero",
        ),
        (
            "app.age >= 18 and text(app.name).startwith('A')",
            ("app",),
            {"app": {"name": "Ann"}},
            {"app": {"name": "Ann", "age": 30}},
            "'str object' has no attribute 'startwith'",
        ),
        (
            "doc.total + doc.label > doc.limit",
            ("doc",),
            {"doc": {"total": 5, "label": "x"}},
            {"doc": {"total": 5, "label": "x", "limit": 1}},
            "unsupported operand type(s) for +: 'int' and 'str'",
        ),
    ],
    ids=["zero-divisor", "misspelled-method", "wrong-type"],
)
def test_an_error_the_check_would_raise_on_the_values_there_shows_where_the_record_carries_the_missing_one(
    check, inputs, lacking, carrying, error
):
    """A missing value stops the check before the values that are there are used, so the rule is skipped on a record
    that lacks it; the error surfaces on the records that carry it."""
    node = screening(check, "not_applicable", inputs=inputs)

    skipped, missing = outcome(run(node, lacking))
    assert skipped == "not_applicable" and missing.startswith("does not apply: missing value for ")
    assert outcome(run(node, carrying)) == ("not_evaluated", f"check could not be evaluated: {error}")


@pytest.fixture
def judging(monkeypatch) -> list[str]:
    """The derived values judged for a skip, in the order `Rules._why_missing` is asked about them."""
    names: list[str] = []
    why_missing = Rules._why_missing

    def spy(self, name, *args):
        names.append(name)
        return why_missing(self, name, *args)

    monkeypatch.setattr(Rules, "_why_missing", spy)
    return names


@pytest.mark.parametrize(
    ("node_policy", "rule_policies", "judged"),
    [
        ("not_evaluated", (None, "fail"), False),
        ("fail", (None, "not_evaluated"), False),
        # The node would skip, but every rule sets a policy of its own that does not.
        ("not_applicable", ("fail", "not_evaluated"), False),
        ("not_applicable", (None, "fail"), True),
        ("not_evaluated", ("not_applicable", None), True),
    ],
)
def test_derived_values_are_judged_for_a_skip_only_where_some_rule_can_skip(
    judging, node_policy, rule_policies, judged
):
    node = Rules(
        name="lending",
        input_fields=[NamedField(name=name) for name in LENDING],
        derived_values=[DerivedValue(name="limit", expression="limits[loan.program]")],
        on_missing=node_policy,
        rules=[
            Rule(id=f"r{index}", check="loan.amount <= limit", on_missing=policy)
            for index, policy in enumerate(rule_policies)
        ],
    )

    run(node, {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS})

    assert judging == (["limit"] if judged else [])


def test_a_node_copied_with_a_policy_that_skips_judges_its_derived_values(judging):
    holding = screening("loan.amount <= limit", inputs=LENDING, derived=(LIMIT,))
    record = {"loan": {"amount": 300000, "program": "jumbo"}, "limits": LIMITS}

    assert outcome(run(holding, record)) == ("not_evaluated", "missing value for limit")
    assert judging == []

    skipping = holding.model_copy(update={"on_missing": "not_applicable"})
    assert outcome(run(skipping, record)) == (
        "not_evaluated",
        "missing value for limit (not skipped: the lookup for limit found nothing)",
    )
    assert judging == ["limit"]


# --- the rule's policy and the node's ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node_policy", "rule_policy", "expected"),
    [
        ("fail", "not_applicable", ("not_applicable", "does not apply: missing value for claim.amount")),
        ("not_evaluated", "not_applicable", ("not_applicable", "does not apply: missing value for claim.amount")),
        ("not_applicable", "not_evaluated", ("not_evaluated", "missing value for claim.amount")),
        ("not_applicable", "fail", ("warn", "missing value for claim.amount")),
        ("not_applicable", None, ("not_applicable", "does not apply: missing value for claim.amount")),
        ("fail", None, ("warn", "missing value for claim.amount")),
    ],
)
def test_a_rule_policy_overrides_the_node_policy(node_policy, rule_policy, expected):
    node = screening("claim.amount <= 10000", rule_policy, inputs=("claim",), node_policy=node_policy)

    assert outcome(run(node, {"claim": {}})) == expected


def test_a_node_set_to_skip_missing_data_skips_it_for_every_rule_that_sets_nothing():
    node = Rules(
        name="claims",
        input_fields=[NamedField(name="claim")],
        on_missing="not_applicable",
        rules=[
            Rule(id="amount", name="amount within the limit", check="claim.amount <= 10000"),
            Rule(id="coded", name="diagnosis coded", check="text(claim.diagnosis) is present"),
            Rule(id="dated", name="date of service", check="date(claim.service_date) <= today()", on_missing="fail"),
        ],
    )

    output = run(node, {"claim": {"diagnosis": "J45"}})

    assert outcome(output, "amount") == ("not_applicable", "does not apply: missing value for claim.amount")
    assert outcome(output, "coded") == ("pass", None)
    assert outcome(output, "dated") == ("fail", "missing value for claim.service_date")
    assert output["status"] == "fail"


# --- the setting ------------------------------------------------------------------------------------------------


def test_the_setting_takes_not_applicable_and_an_empty_value_leaves_it_to_the_node():
    assert RuleMissingPolicy.NOT_APPLICABLE.value == "not_applicable"
    assert Rule(id="r", check="true").on_missing is None
    assert Rule(id="r", check="true", on_missing="not_applicable").on_missing is RuleMissingPolicy.NOT_APPLICABLE
    assert Rule.model_validate({"id": "r", "check": "true", "on_missing": ""}).on_missing is None
    assert Rules(name="n", on_missing="not_applicable").on_missing is RuleMissingPolicy.NOT_APPLICABLE

    node = screening("shipment.weight_kg <= 30", "", inputs=("shipment",), node_policy="not_applicable")
    assert node.rules[0].on_missing is None
    assert outcome(run(node, {"shipment": {}}))[0] == "not_applicable"


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


@pytest.mark.parametrize("unknown", ["skip", "not_aplicable", "Not_Applicable", 1], ids=repr)
def test_a_rule_policy_that_is_no_policy_builds_warns_and_leaves_the_choice_to_the_node(unknown):
    with warnings_logged() as warnings:
        node = screening("shipment.weight_kg <= 30", unknown, inputs=("shipment",), node_policy="not_applicable")

    assert node.rules[0].on_missing is None
    assert warnings == [
        f"Rule 'rule': on_missing {unknown!r} is not one of not_evaluated, fail, not_applicable, "
        "so the node's policy applies."
    ]
    assert outcome(run(node, {"shipment": {}})) == (
        "not_applicable",
        "does not apply: missing value for shipment.weight_kg",
    )


@pytest.mark.parametrize("policy", [None, "", "  ", "not_evaluated", "fail", "not_applicable"], ids=repr)
def test_a_rule_policy_that_is_a_policy_or_empty_logs_nothing(policy):
    with warnings_logged() as warnings:
        Rule(id="r", check="true", on_missing=policy)

    assert warnings == []


def test_a_node_policy_that_is_no_policy_is_still_refused():
    with pytest.raises(ValidationError, match="on_missing"):
        Rules(name="n", on_missing="skip")


def shipping_workflow() -> Workflow:
    start = Input(id="start", name="start")
    review = Rules(
        id="shipping",
        name="shipping",
        input_fields=[NamedField(name="shipment")],
        on_missing="not_applicable",
        rules=[
            Rule(id="weight", name="weight within the limit", check="shipment.weight_kg <= 30"),
            Rule(id="value", name="declared value", check="shipment.value <= 1000", severity="warn", on_missing="fail"),
            Rule(
                id="country", name="destination", check="shipment.country in ['DE', 'FR']", on_missing="not_evaluated"
            ),
            Rule(id="hazmat", name="hazmat declared", check="shipment.hazmat is present", on_missing="not_applicable"),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"shipment": "$.start.output.shipment"}),
    )
    end = Output(id="end", name="end", depends=[NodeDependency(node=review)])
    return Workflow(id="workflow", flow=Flow(id="flow", nodes=[start, review, end]))


def test_the_policies_survive_a_yaml_round_trip(tmp_path):
    path = tmp_path / "shipping.yaml"
    shipping_workflow().to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    node = next(node for node in loaded.flow.nodes if isinstance(node, Rules))

    assert node.on_missing == RuleMissingPolicy.NOT_APPLICABLE
    assert [rule.on_missing for rule in node.rules] == [
        None,
        RuleMissingPolicy.FAIL,
        RuleMissingPolicy.NOT_EVALUATED,
        RuleMissingPolicy.NOT_APPLICABLE,
    ]
    record = {"shipment": {"hazmat": False}}
    output = loaded.run(input_data=record, config=RunnableConfig(callbacks=[])).output["shipping"]["output"]
    original = shipping_workflow().run(input_data=record, config=RunnableConfig(callbacks=[]))
    assert output == original.output["shipping"]["output"]
    assert {finding["rule_id"]: finding["status"] for finding in output["findings"]} == {
        "weight": "not_applicable",
        "value": "warn",
        "country": "not_evaluated",
        "hazmat": "pass",
    }


def test_a_rule_that_sets_no_policy_serializes_as_it_did_before_rules_had_one(tmp_path):
    """A flow saved before rules had a policy of their own must serialize as it did, or every such flow would read as
    changed: the key is left out where the rule sets nothing, and kept where it sets a policy."""
    unset = Rule(id="weight", check="shipment.weight_kg <= 30")
    kept = Rule(id="value", check="shipment.value <= 1000", on_missing="fail")

    assert "on_missing" not in unset.model_dump()
    assert "on_missing" not in json.loads(unset.model_dump_json())
    assert kept.model_dump()["on_missing"] == RuleMissingPolicy.FAIL
    assert json.loads(kept.model_dump_json())["on_missing"] == "fail"
    node = Rules(name="shipping", input_fields=[NamedField(name="shipment")], rules=[unset, kept])
    assert [rule.get("on_missing", "unset") for rule in node.to_dict()["rules"]] == ["unset", RuleMissingPolicy.FAIL]

    path = tmp_path / "shipping.yaml"
    shipping_workflow().to_yaml_file(path)

    saved = yaml.safe_load(path.read_text())["nodes"]["shipping"]
    assert saved["on_missing"] == "not_applicable"
    assert [rule.get("on_missing", "unset") for rule in saved["rules"]] == [
        "unset",
        "fail",
        "not_evaluated",
        "not_applicable",
    ]
    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    node = next(node for node in loaded.flow.nodes if isinstance(node, Rules))
    assert [rule.on_missing for rule in node.rules] == [
        None,
        RuleMissingPolicy.FAIL,
        RuleMissingPolicy.NOT_EVALUATED,
        RuleMissingPolicy.NOT_APPLICABLE,
    ]


EDITOR_YAML = textwrap.dedent(
    """
    nodes:
      start:
        type: dynamiq.nodes.utils.Input
        name: start

      review:
        type: dynamiq.nodes.operators.Rules
        name: review
        input_fields:
          - { id: f1, name: shipment }
        rules:
          - id: weight
            name: Weight within the carrier limit
            severity: fail
            applies_when: ""
            check: shipment.weight_kg <= 30
            message: ""
            on_missing: ""
          - id: country
            name: Destination served
            severity: warn
            applies_when: ""
            check: shipment.country in ['DE', 'FR']
            message: ""
            on_missing: not_applicable
          - id: hazmat
            name: Hazardous goods declared
            severity: warn
            applies_when: ""
            check: shipment.hazmat == false
            message: ""
            on_missing: skip
        on_missing: not_evaluated
        depends:
          - node: start
        input_transformer:
          selector:
            shipment: $.start.output.shipment

    flows:
      review-flow:
        name: Review
        nodes: [start, review]

    workflows:
      review:
        flow: review-flow
    """
)


def test_a_rule_saved_with_an_empty_policy_or_one_that_is_no_policy_takes_the_node_policy(tmp_path):
    path = tmp_path / "review.yaml"
    path.write_text(EDITOR_YAML)

    with warnings_logged() as warnings:
        workflow = Workflow.from_yaml_file(str(path), init_components=True)
    review = next(node for node in workflow.flow.nodes if isinstance(node, Rules))
    result = workflow.run(input_data={"shipment": {}}, config=RunnableConfig(callbacks=[]))

    assert [rule.on_missing for rule in review.rules] == [None, RuleMissingPolicy.NOT_APPLICABLE, None]
    assert warnings == [
        "Rule 'hazmat': on_missing 'skip' is not one of not_evaluated, fail, not_applicable, "
        "so the node's policy applies."
    ]
    assert result.status == RunnableStatus.SUCCESS
    output = result.output["review"]["output"]
    assert [(finding["status"], finding["message"]) for finding in output["findings"]] == [
        ("not_evaluated", "missing value for shipment.weight_kg"),
        ("not_applicable", "does not apply: missing value for shipment.country"),
        # Not skipped: the value that is no policy left the rule to the node's `not_evaluated`.
        ("not_evaluated", "missing value for shipment.hazmat"),
    ]
