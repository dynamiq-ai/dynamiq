import uuid

from dynamiq.cli import flowcheck
from dynamiq.cli.commands.workflow import flow_ui_for

CHOICE = "dynamiq.nodes.operators.Choice"
TABLE = "dynamiq.nodes.operators.DecisionTable"
RULES = "dynamiq.nodes.operators.Rules"
EXPRESSION = "dynamiq.nodes.operators.Expression"
SUB_WORKFLOW = "dynamiq.nodes.operators.SubWorkflow"


def flow_with(*nodes: dict) -> dict:
    """A flow whose Input feeds the given nodes and whose Output reads the last of them."""
    start = {"id": "start", "name": "start", "type": flowcheck.INPUT_TYPE}
    end = {
        "id": "end",
        "name": "end",
        "type": flowcheck.OUTPUT_TYPE,
        "depends": [{"node": nodes[-1]["id"]}],
        "input_transformer": {"selector": {"result": f"$.{nodes[-1]['id']}.output"}},
    }
    return {"id": str(uuid.uuid4()), "nodes": [start, *nodes, end]}


def table(**overrides) -> dict:
    node = {
        "id": "eligibility",
        "name": "eligibility",
        "type": TABLE,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"fico": "$.start.output.fico", "program": "$.start.output.program"}},
        "hit_policy": "first",
        "input_columns": [
            {"id": "c1", "name": "fico", "type": "int"},
            {"id": "c2", "name": "program", "type": "string"},
        ],
        "output_columns": [{"id": "c3", "name": "decision", "type": "string"}],
        "rules": [
            {"id": "r1", "when": ["< 580", None], "then": ["decline"]},
            {"id": "r2", "when": [None, "FHA, VA"], "then": ["review"]},
        ],
    }
    node.update(overrides)
    return node


def rules(**overrides) -> dict:
    node = {
        "id": "review",
        "name": "review",
        "type": RULES,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"claim": "$.start.output.claim", "policy": "$.start.output.policy"}},
        "input_fields": [{"id": "f1", "name": "claim"}, {"id": "f2", "name": "policy"}],
        "derived_values": [{"id": "d1", "name": "payable", "expression": "claim.estimate - policy.deductible"}],
        "rules": [
            {"id": "POL-01", "check": "claim.loss_date >= policy.effective_from", "severity": "fail"},
            {"id": "DOC-01", "check": "", "enabled": False},
        ],
        "on_missing": "not_evaluated",
    }
    node.update(overrides)
    return node


def errors_of(flow: dict) -> list[str]:
    errors, _ = flowcheck.validate(flow)
    return errors


def test_a_well_formed_decision_flow_passes():
    choice = {
        "id": "route",
        "name": "route",
        "type": CHOICE,
        "depends": [{"node": "eligibility"}],
        "hit_policy": "all",
        "options": [
            {
                "id": "approve",
                "name": "approve",
                "condition": {
                    "variable": "$.eligibility.output.decision",
                    "operator": "string-equals",
                    "value": "approve",
                },
            },
            {"id": "other", "name": "other"},
        ],
    }
    expression = {
        "id": "rate",
        "name": "rate",
        "type": EXPRESSION,
        "depends": [{"node": "route", "option": "approve"}],
        "input_transformer": {"selector": {"base_rate": "$.start.output.base_rate"}},
        "input_fields": [{"id": "f1", "name": "base_rate"}],
        "expressions": [{"id": "x1", "key": "rate", "expression": "base_rate + 0.25"}],
    }
    sub = {
        "id": "price-one",
        "name": "price-one",
        "type": SUB_WORKFLOW,
        "depends": [{"node": "rate"}],
        "input_transformer": {"selector": {"rate": "$.rate.output.rate"}},
        "workflow_id": str(uuid.uuid4()),
        "input_fields": [{"name": "rate", "type": "float", "required": True}],
    }

    assert errors_of(flow_with(table(), rules(depends=[{"node": "eligibility"}]), choice, expression, sub)) == []


def test_a_decision_table_is_checked_the_way_the_platform_checks_it():
    short = table(rules=[{"id": "r1", "when": ["< 580"], "then": ["decline"]}])
    reserved = table(output_columns=[{"id": "c3", "name": "matched_rules"}])
    policy = table(hit_policy="last", aggregation="average")
    duplicated = table(output_columns=[{"id": "c3", "name": "rate"}, {"id": "c4", "name": "rate"}])

    assert [e for e in errors_of(flow_with(short)) if "1 `when` cells for 2 input columns" in e]
    assert [e for e in errors_of(flow_with(reserved)) if "reserved for the rules that fired" in e]
    assert [e for e in errors_of(flow_with(duplicated)) if "output column names used more than once: rate" in e]
    found = errors_of(flow_with(policy))
    assert [e for e in found if "hit_policy 'last'" in e] and [e for e in found if "aggregation 'average'" in e]


def test_a_disabled_table_row_short_of_cells_is_left_alone_as_the_node_leaves_it():
    # The node never compiles a disabled row, so a draft switched off while unfinished loads and runs.
    drafted = table(
        rules=[
            {"id": "r1", "when": ["< 580", ""], "then": ["decline"]},
            {"id": "draft", "when": ["< 580"], "then": [], "enabled": False},
            {"id": "short", "when": ["< 580"], "then": ["decline"]},
        ]
    )

    found = [e for e in errors_of(flow_with(drafted)) if "`when` cells" in e or "`then` cells" in e]

    assert len(found) == 1 and "rule 'short' has 1 `when` cells for 2 input columns" in found[0]


def test_a_rules_node_is_checked_the_way_the_platform_checks_it():
    found = errors_of(
        flow_with(
            rules(
                input_fields=[{"id": "f1", "name": "the claim"}, {"id": "f2", "name": "policy"}],
                derived_values=[{"id": "d1", "name": "policy", "expression": ""}],
                rules=[
                    {"id": "POL-01", "check": "", "severity": "block"},
                    {"id": "POL-01", "check": "true"},
                ],
                on_missing="skip",
            )
        )
    )

    for fragment in (
        "input name 'the claim' is not an identifier",
        "derived value 'policy' takes a name",
        "derived value 'policy' has no `expression`",
        "rule 'POL-01' is on but has no `check`",
        "severity 'block'",
        "rule ids used more than once: POL-01",
        "on_missing 'skip'",
    ):
        assert [e for e in found if fragment in e], fragment


def test_an_expression_and_a_sub_workflow_are_checked():
    expression = {
        "id": "rate",
        "name": "rate",
        "type": EXPRESSION,
        "depends": [{"node": "start"}],
        "expressions": [
            {"key": "rate-1", "expression": ""},
            {"key": "tier", "expression": "1"},
            {"key": "tier", "expression": "2"},
        ],
    }
    sub = {
        "id": "price-one",
        "name": "price-one",
        "type": SUB_WORKFLOW,
        "depends": [{"node": "rate"}],
        "workflow_id": "pricing",
        "workflow_version_id": "latest",
        "flow": "pricing",
    }

    found = errors_of(flow_with(expression, sub))
    for fragment in (
        "key 'rate-1' is not an identifier",
        "key 'rate-1' has no `expression`",
        "keys used more than once: tier",
        "workflow_id 'pricing' is not a workflow UUID",
        "workflow_version_id 'latest' is not a version UUID",
        "carries `flow`",
    ):
        assert [e for e in found if fragment in e], fragment


def test_a_branch_must_name_an_option_of_a_choice():
    choice = {
        "id": "route",
        "name": "route",
        "type": CHOICE,
        "depends": [{"node": "start"}],
        "options": [{"id": "approve", "name": "approve"}],
    }
    on_missing_option = {
        "id": "a",
        "name": "a",
        "type": EXPRESSION,
        "depends": [{"node": "route", "option": "decline"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }
    on_plain_node = {
        "id": "b",
        "name": "b",
        "type": EXPRESSION,
        "depends": [{"node": "start", "option": "approve"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }

    named = {
        "id": "gate",
        "name": "gate",
        "type": CHOICE,
        "depends": [{"node": "start"}],
        "options": [{"id": "opt-hi-id", "name": "high"}],
    }
    # The runtime matches an option's id alone, so a branch gated on the name would never be gated.
    on_name = {
        "id": "c",
        "name": "c",
        "type": EXPRESSION,
        "depends": [{"node": "gate", "option": "high"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }

    found = errors_of(flow_with(choice, on_missing_option, on_plain_node, named, on_name))
    assert [e for e in found if "has no such option (it has: approve)" in e]
    assert [e for e in found if "which is not a Choice node" in e]
    assert [e for e in found if "by its name" in e and "Use 'opt-hi-id'" in e]
    assert not [e for e in found if "hit_policy" in e]


def test_the_canvas_draws_a_branch_through_the_options_handle():
    choice = {
        "id": "route",
        "name": "route",
        "type": CHOICE,
        "depends": [{"node": "start"}],
        "options": [{"id": "approve", "name": "Approve it"}, {"id": "other", "name": "other"}],
    }
    approved = {
        "id": "a",
        "name": "a",
        "type": EXPRESSION,
        "depends": [{"node": "route", "option": "approve"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }
    by_name = {
        "id": "b",
        "name": "b",
        "type": EXPRESSION,
        "depends": [{"node": "route", "option": "Approve it"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }
    flow = flow_with(choice, approved, by_name)

    ui = flow_ui_for(flow)
    canvas = {node["node_name"]: node["id"] for node in ui["nodes"]}
    edges = {edge["target"]: edge for edge in ui["edges"]}

    edge = edges[canvas["a"]]
    assert edge["source"] == canvas["route"]
    assert edge["source_handle"] == "approve"
    assert edge["label"] == "Approve it"
    assert edge["is_choice_option"] is True
    assert edge["id"] == f"reactflow__edge-{canvas['route']}approve-{canvas['a']}target"

    # A branch naming the option's name gates nothing at run time, so the canvas draws it ungated.
    ungated = edges[canvas["b"]]
    assert ungated["source_handle"] == "source"
    assert ungated["label"] is None
    assert ungated["is_choice_option"] is False

    plain = edges[canvas["route"]]
    assert plain["source_handle"] == "source"
    assert plain["label"] is None
    assert plain["is_choice_option"] is False
