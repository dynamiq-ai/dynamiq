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


def warnings_of(flow: dict) -> list[str]:
    _, warnings = flowcheck.validate(flow)
    return warnings


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


def test_an_error_edge_names_a_known_trigger_and_no_branch():
    route = {
        "id": "route",
        "name": "route",
        "type": CHOICE,
        "depends": [{"node": "start"}],
        "options": [{"id": "approve", "name": "approve"}],
    }
    on_failure = {
        "id": "refund",
        "name": "refund",
        "type": EXPRESSION,
        "depends": [{"node": "route", "trigger": "failure"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }
    on_unknown_trigger = {
        "id": "a",
        "name": "a",
        "type": EXPRESSION,
        "depends": [{"node": "route", "trigger": "error"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }
    on_failed_branch = {
        "id": "b",
        "name": "b",
        "type": EXPRESSION,
        "depends": [{"node": "route", "option": "approve", "trigger": "failure"}],
        "expressions": [{"key": "x", "expression": "1"}],
    }

    found = errors_of(flow_with(route, on_failure, on_unknown_trigger, on_failed_branch))

    assert not [e for e in found if "'refund'" in e]
    assert [e for e in found if "with trigger 'error'" in e]
    assert [e for e in found if "a Choice that fails takes no branch" in e]


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


JUDGEMENT = "dynamiq.nodes.tools.Judgement"


def judgement(**overrides) -> dict:
    node = {
        "id": "triage",
        "name": "triage",
        "type": JUDGEMENT,
        "depends": [{"node": "start"}],
        "input_transformer": {"selector": {"ticket": "$.start.output.ticket"}},
        "judge": system_one_judge(),
        "input_fields": [{"id": "f1", "name": "ticket"}],
        "questions": [
            {"id": "q1", "name": "is_urgent", "type": "noul", "instructions": "The customer needs an answer today"},
            {
                "id": "q2",
                "name": "team",
                "type": "choice",
                "instructions": "Which team handles this?",
                "options": [{"id": "o1", "name": "billing"}, {"id": "o2", "name": "technical"}],
            },
        ],
        "min_confidence": 0.7,
    }
    node.update(overrides)
    return node


def system_one_judge(**overrides) -> dict:
    judge = {"type": "dynamiq.nodes.detectors.SystemOne", "connection": str(uuid.uuid4())}
    judge.update(overrides)
    return judge


def llm_judge(**overrides) -> dict:
    judge = {"type": "dynamiq.nodes.llms.OpenAI", "model": "gpt-4o", "connection": str(uuid.uuid4())}
    judge.update(overrides)
    return judge


def test_a_judgement_node_is_checked_the_way_the_platform_checks_it():
    assert errors_of(flow_with(judgement())) == []
    assert errors_of(flow_with(judgement(judge=llm_judge()))) == []

    found = errors_of(
        flow_with(
            judgement(
                judge={"type": "dynamiq.nodes.tools.Python"},
                questions=[
                    {"id": "q1", "name": "is urgent", "type": "maybe", "instructions": " "},
                    {
                        "id": "q2",
                        "name": "anger",
                        "type": "score",
                        "instructions": "How angry?",
                        "options": [{"id": "o1", "name": "calm"}, {"id": "o2", "name": "calm"}],
                    },
                    {
                        "id": "q3",
                        "name": "anger",
                        "type": "choice",
                        "instructions": "x",
                        "options": [{"id": "o1", "name": "a"}],
                    },
                ],
                noul_threshold=1.5,
                confidence_mode="sampling",
                samples=1,
            )
        )
    )

    assert [e for e in errors_of(flow_with(judgement(judge=None))) if "has no `judge`" in e]
    sampled = errors_of(flow_with(judgement(confidence_mode="sampling", samples=3)))
    assert [e for e in sampled if "sampling needs an LLM or agent judge" in e], sampled

    for fragment in (
        "`judge` must be a System One, an LLM or an agent node object, got 'dynamiq.nodes.tools.Python'",
        "question 'is urgent' has a name that is not an identifier",
        "type 'maybe' is not one of noul, choice, score",
        "question 'is urgent' has no `instructions`",
        "question 'anger' names a level twice",
        "question 'anger' needs between 2 and 255 options, got 1",
        "question names used more than once: anger",
        "noul_threshold 1.5 is not a number between 0 and 1",
    ):
        assert [e for e in found if fragment in e], fragment


def test_a_judge_missing_what_its_node_class_requires_is_caught_before_the_load_fails():
    """The judge is loaded as a node of its own: `model` and `connection` have no defaults, so a
    judge written with a bare `type` passes every other check and then fails to build."""
    bare = errors_of(flow_with(judgement(judge={"type": "dynamiq.nodes.llms.OpenAI"})))
    assert [e for e in bare if "judge: an LLM node needs `model`" in e], bare
    assert [e for e in bare if "judge: connection None is not a connection UUID" in e], bare

    # An agent judge carries the same requirements one level down.
    agent = errors_of(flow_with(judgement(judge={"type": "dynamiq.nodes.agents.Agent"})))
    assert [e for e in agent if "the agent judge has no `llm` object" in e], agent

    named = errors_of(flow_with(judgement(judge={"type": "dynamiq.nodes.agents.Agent", "llm": llm_judge(model="")})))
    assert [e for e in named if "judge.llm: an LLM node needs `model`" in e], named

    assert errors_of(flow_with(judgement(judge={"type": "dynamiq.nodes.agents.Agent", "llm": llm_judge()}))) == []


def test_sampling_without_samples_is_caught_the_way_the_node_defaults_it():
    """`samples` left out is the node's default of 1, which the node refuses on load."""
    sampling = judgement(judge=llm_judge(), confidence_mode="sampling")
    found = errors_of(flow_with(sampling))
    assert [e for e in found if "sampling needs at least 2 samples" in e], found

    assert errors_of(flow_with({**sampling, "samples": 3})) == []


def test_an_option_that_is_not_an_object_is_named_rather_than_filtered_away():
    """Counting only what survived the isinstance filter let a mixed list report clean and then
    fail to load, since `options` is `list[JudgementOption]` with no string coercion."""
    mixed = judgement(
        questions=[
            {
                "id": "q1",
                "name": "team",
                "type": "choice",
                "instructions": "Which team?",
                "options": [{"id": "o1", "name": "billing"}, {"id": "o2", "name": "sales"}, "refunds"],
            }
        ]
    )
    found = errors_of(flow_with(mixed))
    assert [e for e in found if "has an option that is not an object (1 of 3)" in e], found
    # The count now describes what was written, not what survived.
    assert not [e for e in found if "got 2" in e]


def test_a_judgement_used_as_an_agent_tool_is_checked_the_same_way():
    """The node is built to be an agent's tool, so it reaches the loader from `tools[]` too - where
    nothing but Pipedream used to be inspected."""
    agent = {
        "id": "writer",
        "name": "writer",
        "type": "dynamiq.nodes.agents.Agent",
        "depends": [{"node": "start"}],
        "llm": llm_judge(),
        "tools": [
            {
                "type": JUDGEMENT,
                "name": "triage",
                "questions": [
                    {"id": "q1", "name": "urgent", "type": "noul", "instructions": "x"},
                    {
                        "id": "q2",
                        "name": "urgent",
                        "type": "choice",
                        "instructions": "y",
                        "options": [{"id": "o1", "name": "a"}, {"id": "o2", "name": "b"}],
                    },
                ],
                "noul_threshold": 1.5,
            }
        ],
    }
    found = errors_of(flow_with(agent))

    for fragment in (
        "has no `judge`",
        "question names used more than once: urgent",
        "noul_threshold 1.5 is not a number between 0 and 1",
    ):
        assert [e for e in found if "triage on node writer" in e and fragment in e], (fragment, found)

    # A well-formed one passes, so the check does not just reject every tool placement.
    agent["tools"][0] = {
        "type": JUDGEMENT,
        "name": "triage",
        "judge": system_one_judge(),
        "questions": [{"id": "q1", "name": "urgent", "type": "noul", "instructions": "x"}],
    }
    assert errors_of(flow_with(agent)) == []


def test_a_judgement_that_judges_an_agent_answer_is_not_told_to_become_a_tool():
    agent = {"id": "writer", "name": "writer", "type": "dynamiq.nodes.agents.Agent", "depends": [{"node": "start"}]}
    judge = judgement(depends=[{"node": "writer"}], input_transformer={"selector": {"ticket": "$.writer.output"}})
    assert [w for w in warnings_of(flow_with(agent, judge)) if "standalone step" in w] == []

    # A real tool after an agent still gets the advice.
    python = {
        "id": "after",
        "name": "after",
        "type": "dynamiq.nodes.tools.Python",
        "depends": [{"node": "writer"}],
    }
    assert [w for w in warnings_of(flow_with(agent, python)) if "standalone step" in w]
