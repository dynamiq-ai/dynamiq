"""A missing value counts only where a check reads it.

Jinja evaluates an expression left to right and stops where the result is decided: the branch of an `if` it does not
take, the side of an `and` or an `or` the other side already decided and the rest of a comparison chain already false
are never read, so a value missing there cannot change the result. A check and an `applies_when` each stop at a
missing value where they read it, and name the value they stopped at; a call of a name nothing defines, neither a
helper nor the record, is an error before its arguments are read. A path reads what Jinja reads, a character of a
text by its index and a key of any mapping. A value an expression only asks about (`has`, `is present`, `| default`,
`first_present`) was never needed, and reads as it always did; so does everything in a message, which decides nothing,
in an Expression node, which reads a missing input as None, and in a derived value, which is computed as it always was.
"""

from types import MappingProxyType, SimpleNamespace

import pytest

from dynamiq.nodes.operators import Expression, Rules
from dynamiq.nodes.operators.rules import has, read_paths, resolve_path
from dynamiq.nodes.types import DerivedValue, ExpressionItem, NamedField, Rule
from dynamiq.runnables import RunnableConfig, RunnableStatus

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


# --- a check reads only what decides it ------------------------------------------------------------------------


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


# --- what decides nothing reads as it always did ----------------------------------------------------------------


def test_a_message_reads_a_missing_value_as_it_always_did():
    node = screening("app.amount > 10", message="amount {{ app.amount }} under the floor {{ app.floor }}")

    assert outcome(run(node, {"app": {"amount": 5}})) == ("fail", "amount 5 under the floor")


def test_an_expression_node_reads_a_null_as_it_always_did():
    node = Expression(
        name="calc",
        input_fields=[NamedField(name="app")],
        expressions=[ExpressionItem(key="margin", expression="app.margin or 0")],
    )

    result = node.run(input_data={"app": {"margin": None}}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output == {"margin": 0}
