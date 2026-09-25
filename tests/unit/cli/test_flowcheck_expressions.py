"""`workflow validate` parses Rules and Expression node text, catching a typo before it runs.

flowcheck.check_expressions() walks the parsed Jinja2 AST rather than relying on compiling it:
Jinja only checks a filter or test used inside a conditional expression at run time, so a plain
compile would miss `x | lowr if a else b`. It never imports the SDK engine (see flowcheck's
module docstring), so RULE_HELPERS/RULE_TESTS/RULE_RESERVED_NAMES are a hand-kept copy of
dynamiq.nodes.operators.rules.HELPERS/.TESTS/.RESERVED_NAMES - test_the_mirrored_vocabulary_matches_the_engine
below is the drift guard, for the filter, test and global names too, and it is the one test here
allowed to import the engine.
"""

import json
import uuid

import pytest
from click.testing import CliRunner

from dynamiq.cli import flowcheck
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.workflow import workflow
from dynamiq.cli.config import Settings

RULES = flowcheck.RULES_TYPE
EXPRESSION = flowcheck.EXPRESSION_TYPE


def rules_node(**overrides) -> dict:
    node = {
        "id": "screen",
        "name": "screen",
        "type": RULES,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"loan": "$.start.output.loan"}},
        "input_fields": [{"id": "f1", "name": "loan"}],
        "derived_values": [],
        "rules": [],
    }
    node.update(overrides)
    return node


def expression_node(**overrides) -> dict:
    node = {
        "id": "calc",
        "name": "calc",
        "type": EXPRESSION,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"loan": "$.start.output.loan"}},
        "input_fields": [{"id": "f1", "name": "loan"}],
        "expressions": [],
    }
    node.update(overrides)
    return node


def rule(**overrides) -> dict:
    r = {"id": "R1", "check": "true"}
    r.update(overrides)
    return r


def test_a_syntax_error_names_the_node_and_the_rule():
    node = rules_node(rules=[rule(check="loan.amount >")])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert warnings == []
    assert any("screen" in e and "R1" in e and "not a valid expression" in e for e in errors), errors


def test_an_unclosed_bracket_reads_as_the_end_of_the_expression_not_the_wrappers_brace():
    # check_expressions() itself wraps the author's text as "{{ " + text + " }}" to parse it. An
    # unclosed bracket runs Jinja's lexer off the end of the author's OWN text and into that
    # wrapping, so Jinja's raw message names the wrapper's own "}" - something the author never
    # typed - as the unexpected token: "unexpected '}', expected ')'".
    node = rules_node(rules=[rule(check="loan.amount > 100 and (((")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert errors == [
        "rules 'screen': rule 'R1' check is not a valid expression: unexpected end of expression, expected ')'"
    ]
    assert "'}'" not in errors[0]


def test_an_unterminated_quote_points_at_the_authors_own_text_not_the_wrapped_one():
    # Jinja's fallback error for a string with no closing quote names an absolute character
    # position - counted from the start of the WRAPPED text, so it is off by len("{{ "), unless
    # translated back to where the quote actually sits in the check the author wrote (15, here).
    node = rules_node(rules=[rule(check="loan.status == 'unterminated")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert errors == ["rules 'screen': rule 'R1' check is not a valid expression: unexpected char \"'\" at 15"]
    assert "at 18" not in errors[0]  # the wrapped-text position, which the author never wrote


def test_an_ordinary_syntax_error_keeps_jinjas_own_wording():
    # A mistake that is really and only in the author's own text - not one that runs off the end
    # into check_expressions()'s own wrapping - is not touched at all.
    node = rules_node(rules=[rule(check="loan..amount")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert errors == ["rules 'screen': rule 'R1' check is not a valid expression: expected name or number"]


def test_an_unknown_filter_inside_a_conditional_is_still_caught():
    # Compiling `loan.amount | lowr if loan.flag else loan.amount` would not catch this: Jinja
    # only resolves a filter used inside a conditional expression at run time.
    node = rules_node(rules=[rule(check="loan.amount | lowr if loan.flag else loan.amount")])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert warnings == []
    assert any("R1" in e and "unknown filter 'lowr'" in e and "'lower'" in e for e in errors), errors


def test_an_unknown_test_is_an_error_with_a_suggestion():
    node = rules_node(rules=[rule(check="loan.amount is presnt")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert any("R1" in e and "unknown test 'presnt'" in e and "'present'" in e for e in errors), errors


def test_calling_an_unknown_name_as_a_helper_is_an_error_with_a_suggestion():
    node = rules_node(rules=[rule(check="firstpresent(loan.amount)")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert any("R1" in e and "unknown helper 'firstpresent'" in e and "'first_present'" in e for e in errors), errors


def test_an_unknown_root_name_is_a_warning_never_an_error():
    node = rules_node(rules=[rule(check="lon.amount > 100")])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == []
    assert any("R1" in w and "'lon'" in w and "'loan'" in w for w in warnings), warnings


def test_an_unknown_root_is_not_warned_about_when_the_node_declares_no_shape():
    # Neither input_fields nor a selector: the record's real shape is a total unknown, so nothing
    # here could tell a typo from a legitimate field validate has never heard of.
    node = rules_node(input_fields=[], input_transformer={}, rules=[rule(check="anything.at_all > 1")])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [] and warnings == []


def test_an_unknown_root_is_not_warned_about_when_input_transformer_sets_a_path():
    # A `path` makes the record a sub-tree selected out of a larger payload, so declared inputs
    # no longer describe the whole shape the check reads.
    node = rules_node(
        input_transformer={"path": "$.start.output", "selector": {"loan": "amount"}},
        rules=[rule(check="lon.amount > 1")],
    )

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [] and warnings == []


def test_a_disabled_rules_check_is_not_parsed_at_all():
    node = rules_node(rules=[rule(check="loan.amount | lowr", enabled=False)])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [] and warnings == []


def test_a_disabled_rules_on_missing_is_still_validated():
    # `on_missing` is a plain configuration value, checked in check_rules() (not check_expressions())
    # the same unconditional way `severity` already is - unlike the check/applies_when text, which
    # a disabled rule never compiles and so is never parsed either.
    node = rules_node(rules=[rule(check="loan.amount | lowr", on_missing="skip", enabled=False)])

    errors = flowcheck.check_rules(node, "screen")

    assert any("R1" in e and "on_missing 'skip'" in e for e in errors), errors


def test_an_applies_when_typo_is_caught_the_same_way_as_a_check():
    node = rules_node(rules=[rule(check="true", applies_when="loan.amount is presnt")])

    errors, _ = flowcheck.check_expressions(node, "screen")

    assert any("applies_when" in e and "unknown test 'presnt'" in e for e in errors), errors


def test_a_derived_value_that_reads_one_defined_later_gets_its_own_wording():
    node = rules_node(
        derived_values=[
            {"id": "d1", "name": "first", "expression": "second + 1"},
            {"id": "d2", "name": "second", "expression": "loan.amount"},
        ]
    )

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == []
    assert any("'first'" in w and "'second'" in w and "computed after" in w for w in warnings), warnings


def test_a_well_formed_rules_node_is_silent():
    node = rules_node(
        derived_values=[{"id": "d1", "name": "payable", "expression": "number(loan.amount) - loan.deductible"}],
        rules=[
            rule(id="R1", check="has(loan.amount) and payable > 0", applies_when="loan.amount is present"),
            rule(id="R2", check="text(loan.status) == 'approved'"),
        ],
    )

    assert flowcheck.check_expressions(node, "screen") == ([], [])


def test_an_expression_node_checks_filters_and_roots_too():
    node = expression_node(expressions=[{"id": "x1", "key": "rate", "expression": "loan.amount | lowr"}])
    errors, _ = flowcheck.check_expressions(node, "calc")
    assert any("unknown filter 'lowr'" in e for e in errors), errors

    node = expression_node(expressions=[{"id": "x1", "key": "rate", "expression": "lon.amount"}])
    _, warnings = flowcheck.check_expressions(node, "calc")
    assert any("'lon'" in w and "'loan'" in w for w in warnings), warnings


def test_a_well_formed_expression_node_is_silent():
    node = expression_node(expressions=[{"id": "x1", "key": "rate", "expression": "loan.amount | round(2)"}])

    assert flowcheck.check_expressions(node, "calc") == ([], [])


def test_an_expression_node_accepts_the_same_vocabulary_as_rules():
    # expression.py's Expression node shares rules.py's RecordSandbox (same HELPERS, same TESTS),
    # so a name the docstring lists only for Rules - text/number/first_present, is present/is
    # blank - must not falsely read as unknown here.
    node = expression_node(
        expressions=[
            {
                "id": "x1",
                "key": "rate",
                "expression": (
                    "text(loan.status) == 'approved' and number(loan.amount) > 0 "
                    "and first_present(loan.a, loan.b) is present and loan.c is blank"
                ),
            }
        ]
    )

    assert flowcheck.check_expressions(node, "calc") == ([], [])


def test_as_of_is_not_a_known_root_for_an_expression_node():
    # `as_of` is the Rules node's own input (it fixes the effective-window date); an Expression
    # node has no such thing, so unlike on a Rules node, reading it here is an unknown root.
    node = expression_node(expressions=[{"id": "x1", "key": "k", "expression": "as_of"}])

    errors, warnings = flowcheck.check_expressions(node, "calc")

    assert errors == []
    assert any("'as_of'" in w and "does not declare" in w for w in warnings), warnings

    # The same name IS known on a Rules node.
    control = rules_node(rules=[rule(check="as_of is present")])
    assert flowcheck.check_expressions(control, "screen") == ([], [])


# --- what the engine refuses to build ----------------------------------------------------------------------------

SELF_REFUSAL = (
    "Jinja reserves the name 'self' inside an expression, so a top-level key of that name cannot be read; "
    "nest it inside a record or rename the input"
)


@pytest.mark.parametrize("check", ["loan.a }} loan.b", "loan.a }}{{ loan.b", "loan.a }}"])
def test_text_the_engine_reads_as_more_than_one_expression_is_an_error(check):
    # Wrapped in "{{ }}", `loan.a }} loan.b` parses as a template: an expression, then text. The engine compiles one
    # expression and refuses whatever follows the "}}" that ends it.
    node = rules_node(rules=[rule(check=check)])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [
        "rules 'screen': rule 'R1' check is not a valid expression: chunk after expression; "
        "write the expression alone, without '{{' or '}}'"
    ]
    assert warnings == []


def test_an_expression_node_item_that_is_more_than_one_expression_is_an_error():
    node = expression_node(expressions=[{"id": "x1", "key": "rate", "expression": "loan.a }}{{ loan.b"}])

    errors, _ = flowcheck.check_expressions(node, "calc")

    assert errors == [
        "expression 'calc': key 'rate' is not a valid expression: chunk after expression; "
        "write the expression alone, without '{{' or '}}'"
    ]


DEEP = {
    "nested-brackets": "(" * 400 + "loan.a" + ")" * 400 + " > 1",
    "long-chain": " or ".join(f"loan.f{index} == 'x'" for index in range(3000)),
}


@pytest.mark.parametrize("check", DEEP.values(), ids=DEEP.keys())
def test_an_expression_nested_too_deeply_to_read_is_an_error_naming_the_node_and_rule(check):
    node = rules_node(rules=[rule(check=check)])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == ["rules 'screen': rule 'R1' check is nested too deeply to read; split it into smaller expressions"]
    assert warnings == []


def test_validate_reports_an_expression_nested_too_deeply_rather_than_crashing():
    flow = _flow_with_a_root_typo_warning()
    flow["nodes"][1]["rules"] = [rule(check=DEEP["nested-brackets"])]

    errors, _ = flowcheck.validate(flow)

    assert any("rule 'R1' check is nested too deeply to read" in error for error in errors), errors


@pytest.mark.parametrize(
    ("check", "refusal"),
    [
        ("self.limit > 1", f"reads 'self.limit': {SELF_REFUSAL}"),
        ("has(self)", f"reads 'self': {SELF_REFUSAL}"),
        ("loan.__class__ is defined", "reads a private attribute (loan.__class__)"),
        ("loan['__dict__'] is defined", "reads a private attribute (loan.__dict__)"),
        ('loan["__b.c"] is defined', "reads a private attribute (loan['__b.c'])"),
        ("date(loan.closed) < date", "reads 'date' as a value and calls it as a helper"),
        ("has(len) and len(loan.items) > 1", "reads 'len' as a value and calls it as a helper"),
        ("range(3) | list == range", "reads 'range' as a value and calls it as a helper"),
    ],
    ids=[
        "self",
        "self-asked-about",
        "private-attribute",
        "private-key",
        "private-quoted-key",
        "helper-read",
        "helper-asked-about",
        "jinja-global-read",
    ],
)
def test_what_the_engine_refuses_to_build_is_an_error_with_its_reason(check, refusal):
    node = rules_node(rules=[rule(check=check)])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [f"rules 'screen': rule 'R1' check {refusal}"]
    assert warnings == []


@pytest.mark.parametrize(
    "check",
    [
        "loan.self > 1",
        "loan._id > 1",
        "loan['a.__b'] is defined",
        # A method's name is no read of the record: the sandbox refuses this one when it runs, not the build.
        "loan.__len__() > 1",
        # `text`, `number` and `first_present` came after flows that read inputs so named: the engine holds such a
        # rule when it runs instead.
        "text(text) == 'x'",
        # The engine reads neither what `sameas` compares with nor a keyword argument of `default`.
        "loan.x is sameas date and date(loan.a) > today()",
        "(loan.a | default(boolean=date)) and date(loan.b) > today()",
    ],
)
def test_what_the_engine_builds_is_not_refused(check):
    errors, _ = flowcheck.check_expressions(rules_node(rules=[rule(check=check)]), "screen")

    assert errors == []


def test_an_expression_node_mirrors_the_refusals_its_engine_makes():
    # The Expression node refuses a read of `self` and a helper's name read as a value, as the Rules node does; a
    # private attribute it leaves to the sandbox, which refuses the read when the expression runs.
    def errors_of(expression: str) -> list:
        node = expression_node(expressions=[{"id": "x1", "key": "k", "expression": expression}])
        return flowcheck.check_expressions(node, "calc")[0]

    assert errors_of("self.a") == [f"expression 'calc': key 'k' reads 'self.a': {SELF_REFUSAL}"]
    assert errors_of("len(loan.items) > len") == [
        "expression 'calc': key 'k' reads 'len' as a value and calls it as a helper"
    ]
    assert errors_of("loan.__class__") == []


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("date", "rules 'screen': derived value 'date' is already the name of a helper."),
        ("len", "rules 'screen': derived value 'len' is already the name of a helper."),
        (
            "self",
            "rules 'screen': derived value 'self' could not be read by a rule: Jinja reserves the name inside an "
            "expression.",
        ),
    ],
)
def test_a_derived_value_named_as_the_engine_refuses_is_an_error(name, error):
    node = rules_node(derived_values=[{"id": "d1", "name": name, "expression": "loan.amount"}])

    assert flowcheck.check_rules(node, "screen") == [error]


@pytest.mark.parametrize("name", ["text", "number", "first_present", "range"])
def test_a_derived_value_may_take_a_name_the_engine_lets_it_shadow(name):
    # The helpers added since derived values could first be named keep flows that already use those names building;
    # Jinja's own globals were never refused.
    node = rules_node(derived_values=[{"id": "d1", "name": name, "expression": "loan.amount"}])

    assert flowcheck.check_rules(node, "screen") == []


# --- a rule's message --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("{{ order.total | lowr }}", "rules 'screen': rule 'R1' message: unknown filter 'lowr'. Did you mean 'lower'?"),
        (
            "{{ order.total",
            "rules 'screen': rule 'R1' message is not a valid template: unexpected end of template, expected 'end of "
            "print statement'.",
        ),
        (
            "{{ order.total is presnt }}",
            "rules 'screen': rule 'R1' message: unknown test 'presnt'. Did you mean 'present'?",
        ),
        # Jinja looks a filter up inside `{% if %}` only when that branch renders, and the message then comes back as it
        # was written.
        (
            "{% if order.late %}{{ order.total | lowr }}{% endif %}",
            "rules 'screen': rule 'R1' message: unknown filter 'lowr'. Did you mean 'lower'?",
        ),
        ("Total {{ self.total }}", f"rules 'screen': rule 'R1' message reads 'self.total': {SELF_REFUSAL}"),
        (
            "{{ " + DEEP["nested-brackets"] + " }}",
            "rules 'screen': rule 'R1' message is nested too deeply to read; split it into smaller expressions",
        ),
    ],
    ids=["unknown-filter", "unfinished", "unknown-test", "unknown-filter-in-a-branch", "self", "nested-too-deeply"],
)
def test_a_rule_message_is_checked_as_the_template_the_engine_compiles(message, error):
    node = rules_node(rules=[rule(message=message)])

    errors, warnings = flowcheck.check_expressions(node, "screen")

    assert errors == [error]
    assert warnings == []


@pytest.mark.parametrize(
    "message",
    [
        # `order` is no input of the node: a message prints a value the record lacks as empty text.
        "Total {{ order.total }}",
        # A message may call what it defines itself.
        "{% set comma = joiner(', ') %}{% for item in loan.items %}{{ comma() }}{{ item.name }}{% endfor %}",
        "{% macro money(x) %}{{ x | round(2) }}{% endmacro %}Total {{ money(loan.amount) }}",
        # The engine refuses a private attribute in a check, but builds a message that reads one: it prints the key.
        "Kind {{ loan.__typename }}",
    ],
    ids=["undeclared-name", "joiner", "macro", "private-key"],
)
def test_a_message_the_engine_builds_is_not_refused(message):
    node = rules_node(rules=[rule(message=message)])

    assert flowcheck.check_expressions(node, "screen") == ([], [])


def test_a_disabled_rules_message_is_not_parsed_at_all():
    node = rules_node(rules=[rule(message="{{ order.total | lowr", enabled=False)])

    assert flowcheck.check_expressions(node, "screen") == ([], [])


# --- a node inside another ---------------------------------------------------------------------------------------

MAP = "dynamiq.nodes.operators.Map"


def _flow_mapping(inner: dict) -> dict:
    """A flow whose Map runs `inner` over each order the input holds."""
    start = {"id": "start", "name": "start", "type": flowcheck.INPUT_TYPE}
    mapper = {
        "id": "map-1",
        "name": "map-1",
        "type": MAP,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"input": "$.start.output.orders"}},
        "node": inner,
    }
    end = {
        "id": "end",
        "name": "end",
        "type": flowcheck.OUTPUT_TYPE,
        "depends": [{"node": "map-1"}],
        "input_transformer": {"selector": {"result": "$.map-1.output"}},
    }
    return {"id": str(uuid.uuid4()), "nodes": [start, mapper, end]}


def test_a_rules_node_inside_a_map_is_checked_as_a_top_level_one_is():
    """The loader builds a Map's node with the flow, so its rules compile, or fail to, as a top-level node's do."""
    inner = rules_node(
        id="rules-1",
        name="rules-1",
        depends=[],
        rules=[rule(check="loan.amount | lowr == 'x'", message="{{ loan.amount"), rule(id="R2", on_missing="skip")],
    )

    errors, _ = flowcheck.validate(_flow_mapping(inner))

    assert [error for error in errors if "rules-1" in error] == [
        "rules 'map-1 > rules-1': rule 'R2' on_missing 'skip' is not one of not_evaluated, fail, not_applicable.",
        "rules 'map-1 > rules-1': rule 'R1' check: unknown filter 'lowr'. Did you mean 'lower'?",
        "rules 'map-1 > rules-1': rule 'R1' message is not a valid template: unexpected end of template, expected "
        "'end of print statement'.",
    ]


def test_an_expression_node_inside_a_map_is_checked_as_a_top_level_one_is():
    inner = expression_node(
        id="calc-1",
        name="calc-1",
        depends=[],
        expressions=[{"id": "x1", "key": "rate", "expression": "lon.amount | lowr"}],
    )

    errors, warnings = flowcheck.validate(_flow_mapping(inner))

    assert [error for error in errors if "calc-1" in error] == [
        "expression 'map-1 > calc-1': key 'rate': unknown filter 'lowr'. Did you mean 'lower'?"
    ]
    assert [warning for warning in warnings if "calc-1" in warning] == [
        "expression 'map-1 > calc-1': key 'rate' reads 'lon', which this node does not declare. Did you mean 'loan'?"
    ]


def test_a_node_nested_deeper_is_labelled_with_the_path_to_it():
    inner = rules_node(depends=[], rules=[rule(check="loan.amount | lowr == 'x'")])
    del inner["id"]
    outer = _flow_mapping({"id": "map-2", "name": "map-2", "type": MAP, "node": inner})

    errors, _ = flowcheck.validate(outer)

    assert "rules 'map-1 > map-2 > screen': rule 'R1' check: unknown filter 'lowr'. Did you mean 'lower'?" in errors


def test_an_unrecognized_rule_level_on_missing_is_an_error():
    node = rules_node(rules=[rule(check="true", on_missing="skip")])

    errors = flowcheck.check_rules(node, "screen")

    assert any(
        "R1" in e and "on_missing 'skip'" in e and "not_evaluated" in e and "not_applicable" in e for e in errors
    ), errors


def test_a_blank_rule_level_on_missing_is_left_to_the_node():
    node = rules_node(rules=[rule(check="true", on_missing="  ")])

    errors = flowcheck.check_rules(node, "screen")

    assert not any("on_missing" in e for e in errors), errors


def test_node_level_on_missing_accepts_not_applicable():
    node = rules_node(on_missing="not_applicable", rules=[rule(check="true")])

    errors = flowcheck.check_rules(node, "screen")

    assert not any("on_missing" in e for e in errors), errors


def test_the_mirrored_vocabulary_matches_the_engine():
    """The one test here allowed to import the engine (see flowcheck's module docstring) - it is
    the guard against flowcheck's hand-kept names drifting from the engine's: the helpers, the
    tests, the original helper names, the name Jinja reserves, the tests that only ask about their
    value, and the filters, tests and globals each engine's sandbox carries."""
    from dynamiq.nodes.operators import expression, rules

    assert flowcheck.RULE_HELPERS == set(rules.HELPERS)
    assert flowcheck.RULE_TESTS == set(rules.TESTS)
    assert flowcheck.RULE_RESERVED_NAMES == rules.RESERVED_NAMES
    assert flowcheck.RESERVED_ROOT == rules.RESERVED_ROOT
    assert flowcheck._EXEMPT_TESTS == rules._EXEMPT_TESTS
    assert flowcheck._GLOBAL_NAMES == rules.GLOBAL_NAMES
    for sandbox in (rules._ENVIRONMENT, expression._ENVIRONMENT):
        assert flowcheck._FILTER_NAMES == set(sandbox.filters)
        assert flowcheck._TEST_NAMES == set(sandbox.tests)
        assert flowcheck._GLOBAL_NAMES == set(sandbox.globals)


def _flow_with_a_root_typo_warning() -> dict:
    start = {"id": "start", "name": "start", "type": flowcheck.INPUT_TYPE}
    screen = rules_node(rules=[rule(check="lon.amount > 100")])
    end = {
        "id": "end",
        "name": "end",
        "type": flowcheck.OUTPUT_TYPE,
        "depends": [{"node": "screen"}],
        "input_transformer": {"selector": {"result": "$.screen.output"}},
    }
    return {"id": str(uuid.uuid4()), "nodes": [start, screen, end]}


def test_cli_validate_offline_exits_zero_and_prints_the_warning():
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id=str(uuid.uuid4()))

    result = CliRunner().invoke(
        workflow, ["validate", json.dumps(_flow_with_a_root_typo_warning()), "--offline"], obj=dctx
    )

    assert result.exit_code == 0, result.output
    assert "warning:" in result.output
    assert "lon" in result.output
