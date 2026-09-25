"""`workflow validate` parses Rules and Expression node text, catching a typo before it runs.

flowcheck.check_expressions() walks the parsed Jinja2 AST rather than relying on compiling it:
Jinja only checks a filter or test used inside a conditional expression at run time, so a plain
compile would miss `x | lowr if a else b`. It never imports the SDK engine (see flowcheck's
module docstring), so RULE_HELPERS/RULE_TESTS are a hand-kept copy of
dynamiq.nodes.operators.rules.HELPERS/.TESTS - test_the_mirrored_vocabulary_matches_the_engine
below is the drift guard, and it is the one test here allowed to import the engine.
"""

import json
import uuid

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
    the guard against RULE_HELPERS/RULE_TESTS drifting from the real HELPERS/TESTS dicts."""
    from dynamiq.nodes.operators.rules import HELPERS, TESTS

    assert flowcheck.RULE_HELPERS == set(HELPERS)
    assert flowcheck.RULE_TESTS == set(TESTS)


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
