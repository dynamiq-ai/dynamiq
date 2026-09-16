import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.node import ErrorHandling, NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, DecisionTable, Expression
from dynamiq.nodes.types import Behavior, ChoiceCondition, ConditionOperator, DecisionRule, ExpressionItem, NamedField
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

COLUMNS = [
    NamedField(id="fico", name="fico", type="int"),
    NamedField(id="ltv", name="ltv", type="int"),
    NamedField(id="program", name="program", type="string"),
]


def adjustments_table(**overrides) -> DecisionTable:
    """Loan-level price adjustments: every matching row adds to the rate, the way a scorecard adds points."""
    fields = {
        "id": "adjustments",
        "name": "adjustments",
        "hit_policy": "collect",
        "aggregation": "sum",
        "input_columns": COLUMNS,
        "output_columns": [NamedField(id="llpa", name="llpa", type="float"), NamedField(id="note", name="note")],
        "rules": [
            DecisionRule(id="r1", name="mid fico", when=["[680..739]", "", ""], then=["0.75", "fico"]),
            DecisionRule(id="r2", name="high ltv", when=["", "> 80", ""], then=["0.25", "ltv"]),
            DecisionRule(id="r3", name="government", when=["", "", "FHA, VA"], then=["0.125", ""]),
            DecisionRule(id="r4", name="legacy", when=["< 620", "", ""], then=["1.5", "legacy"], enabled=False),
        ],
    }
    return DecisionTable(**(fields | overrides))


def run_node(node: DecisionTable, input_data: dict):
    return node.run(input_data=input_data, config=RunnableConfig(callbacks=[]))


def test_collect_folds_numeric_columns_and_keeps_the_rest_as_lists():
    result = run_node(adjustments_table(), {"fico": 700, "ltv": 85, "program": "FHA"})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {
        "llpa": 1.125,
        "note": ["fico", "ltv", None],
        "matched_rules": [
            {"id": "r1", "name": "mid fico"},
            {"id": "r2", "name": "high ltv"},
            {"id": "r3", "name": "government"},
        ],
    }


@pytest.mark.parametrize(
    ("aggregation", "llpa"),
    [("list", [0.75, 0.25]), ("sum", 1.0), ("min", 0.25), ("max", 0.75), ("count", 2)],
)
def test_collect_aggregations(aggregation, llpa):
    result = run_node(adjustments_table(aggregation=aggregation), {"fico": 700, "ltv": 90, "program": "Conventional"})

    assert result.output["llpa"] == llpa


def test_first_returns_the_first_matching_rule_in_table_order():
    result = run_node(adjustments_table(hit_policy="first"), {"fico": 700, "ltv": 90, "program": "FHA"})

    assert result.output == {"llpa": 0.75, "note": "fico", "matched_rules": [{"id": "r1", "name": "mid fico"}]}


def test_unique_fails_the_run_when_rules_overlap():
    result = run_node(adjustments_table(hit_policy="unique"), {"fico": 700, "ltv": 90, "program": "FHA"})

    assert result.status == RunnableStatus.FAILURE
    assert "rules mid fico, high ltv, government all match" in result.error.message


def test_no_match_yields_null_outputs_and_a_disabled_rule_never_fires():
    result = run_node(adjustments_table(), {"fico": 600, "ltv": 50, "program": "Conventional"})

    assert result.output == {"llpa": None, "note": [], "matched_rules": []}


def test_inputs_are_read_as_their_column_type():
    result = run_node(adjustments_table(hit_policy="first"), {"fico": "700", "ltv": "90", "program": 12})

    assert result.output["matched_rules"] == [{"id": "r1", "name": "mid fico"}]


def test_a_missing_input_matches_only_empty_cells():
    result = run_node(adjustments_table(), {"program": "VA"})

    assert result.output["matched_rules"] == [{"id": "r3", "name": "government"}]


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ([DecisionRule(name="bad", when=[">=", "", ""], then=["", ""])], "rule 1 \\(bad\\), input 'fico': >= needs"),
        ([DecisionRule(when=["abc", "", ""], then=["", ""])], "input 'fico': expected a number, got 'abc'"),
        ([DecisionRule(when=["", "", ""], then=["high", ""])], "output 'llpa': expected a number"),
        ([DecisionRule(when=["", ""], then=["", ""])], "one cell per input and per output column"),
    ],
)
def test_malformed_rules_fail_when_the_node_is_built(rules, message):
    with pytest.raises(ValueError, match=message):
        adjustments_table(rules=rules)


def test_reserved_output_name_is_refused():
    with pytest.raises(ValueError, match="'matched_rules' is reserved"):
        adjustments_table(output_columns=[NamedField(name="matched_rules")], rules=[])


def pricing_workflow() -> Workflow:
    """Input → eligibility table → adjustments table → rate expression → route on the rate → Output."""
    start = Input(id="start", name="start")
    eligibility = DecisionTable(
        id="eligibility",
        name="eligibility",
        input_columns=COLUMNS,
        output_columns=[NamedField(id="decision", name="decision", type="string")],
        rules=[
            DecisionRule(id="e1", name="decline", when=["< 580", "", ""], then=["decline"]),
            DecisionRule(id="e2", name="fha floor", when=["[580..619]", "", "FHA"], then=["review"]),
            DecisionRule(id="e3", name="approve", when=["", "<= 97", ""], then=["approve"]),
            DecisionRule(id="e4", name="fallback", when=["", "", ""], then=["review"]),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={"fico": "$.start.output.fico", "ltv": "$.start.output.ltv", "program": "$.start.output.program"}
        ),
    )
    adjustments = adjustments_table(
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={"fico": "$.start.output.fico", "ltv": "$.start.output.ltv", "program": "$.start.output.program"}
        ),
    )
    rate = Expression(
        id="rate",
        name="rate",
        expressions=[
            ExpressionItem(key="rate", expression="(base_rate + (llpa or 0)) | round(3)"),
            ExpressionItem(key="tier", expression="'A' if (llpa or 0) < 0.5 else 'B'"),
        ],
        pass_through=True,
        depends=[NodeDependency(node=start), NodeDependency(node=adjustments)],
        input_transformer=InputTransformer(
            selector={"base_rate": "$.start.output.base_rate", "llpa": "$.adjustments.output.llpa"}
        ),
    )
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="approved",
                condition=ChoiceCondition(
                    operator=ConditionOperator.STRING_EQUALS, variable="$.decision", value="approve"
                ),
            ),
            ChoiceOption(id="manual"),
        ],
        depends=[NodeDependency(node=eligibility)],
        input_transformer=InputTransformer(selector={"decision": "$.eligibility.output.decision"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=route, option="approved"), NodeDependency(node=rate)],
        input_transformer=InputTransformer(selector={"rate": "$.rate.output.rate", "tier": "$.rate.output.tier"}),
    )
    return Workflow(
        id="pricing", flow=Flow(id="pricing-flow", nodes=[start, eligibility, adjustments, rate, route, end])
    )


def test_tables_expression_and_choice_price_a_loan_end_to_end(mock_tracing_client):
    tracing = TracingCallbackHandler(client=mock_tracing_client())
    workflow = pricing_workflow()

    result = workflow.run(
        input_data={"fico": 700, "ltv": 85, "program": "FHA", "base_rate": 6.5},
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["eligibility"]["output"]["decision"] == "approve"
    assert result.output["rate"]["output"] == {"base_rate": 6.5, "llpa": 1.125, "rate": 7.625, "tier": "B"}
    assert result.output["end"]["output"] == {"rate": 7.625, "tier": "B"}
    table_run = next(run for run in tracing.runs.values() if run.name == "adjustments")
    assert table_run.status == RunStatus.SUCCEEDED
    assert table_run.output["matched_rules"] == [
        {"id": "r1", "name": "mid fico"},
        {"id": "r2", "name": "high ltv"},
        {"id": "r3", "name": "government"},
    ]
    assert len(table_run.executions) == 1


def test_a_declined_loan_skips_the_approved_branch():
    result = pricing_workflow().run(input_data={"fico": 550, "ltv": 80, "program": "VA", "base_rate": 6.5})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["eligibility"]["output"]["decision"] == "decline"
    assert result.output["route"]["output"]["approved"]["status"] == RunnableStatus.FAILURE
    assert result.output["end"]["status"] == RunnableStatus.SKIP


def test_yaml_round_trip_keeps_the_rules(tmp_path):
    workflow = pricing_workflow()
    input_data = {"fico": 700, "ltv": 85, "program": "FHA", "base_rate": 6.5}
    expected = workflow.run(input_data=input_data).output["end"]["output"]

    first = tmp_path / "pricing.yaml"
    workflow.to_yaml_file(first)
    loaded = Workflow.from_yaml_file(str(first), init_components=True)
    second = tmp_path / "pricing_again.yaml"
    loaded.to_yaml_file(second)
    reloaded = Workflow.from_yaml_file(str(second), init_components=True)

    table = next(node for node in reloaded.flow.nodes if node.id == "adjustments")
    assert table.hit_policy == "collect" and table.aggregation == "sum"
    assert [rule.enabled for rule in table.rules] == [True, True, True, False]
    assert reloaded.run(input_data=input_data).output["end"]["output"] == expected


def test_the_editor_payload_loads_with_null_cells():
    data = {
        "nodes": {
            "start": {"type": "dynamiq.nodes.utils.Input", "name": "start"},
            "table": {
                "type": "dynamiq.nodes.operators.DecisionTable",
                "name": "table",
                "hit_policy": "collect",
                "aggregation": "sum",
                "input_columns": [{"id": "c1", "name": "fico", "type": "int"}],
                "output_columns": [{"id": "o1", "name": "points", "type": "int"}],
                "rules": [
                    {"id": "r1", "name": "prime", "when": [">= 700"], "then": ["10"]},
                    {"id": "r2", "name": "off", "when": [None], "then": [None], "enabled": False},
                    {"id": "r3", "name": "any", "when": [None], "then": ["1"]},
                ],
                "depends": [{"node": "start"}],
                "input_transformer": {"path": None, "selector": {"fico": "$.start.output.fico"}},
            },
        },
        "flows": {"flow": {"name": "flow", "nodes": ["start", "table"]}},
        "workflows": {"workflow": {"flow": "flow"}},
    }

    workflow = Workflow.from_yaml_file_data(WorkflowYAMLLoader.parse(data, init_components=True))
    result = workflow.run(input_data={"fico": 720})

    assert result.output["table"]["output"] == {
        "points": 11,
        "matched_rules": [{"id": "r1", "name": "prime"}, {"id": "r3", "name": "any"}],
    }


def test_a_failing_table_with_return_behavior_leaves_the_flow_running():
    start = Input(id="start", name="start")
    table = adjustments_table(
        hit_policy="unique",
        error_handling=ErrorHandling(behavior=Behavior.RETURN),
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(
            selector={"fico": "$.start.output.fico", "ltv": "$.start.output.ltv", "program": "$.start.output.program"}
        ),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=table)],
        input_transformer=InputTransformer(selector={"llpa": "$.adjustments.output.llpa"}),
    )
    workflow = Workflow(flow=Flow(nodes=[start, table, end]))

    result = workflow.run(input_data={"fico": 700, "ltv": 85, "program": "FHA"})

    # Three rules match the input, which a unique table refuses; with a return behavior the table fails
    # alone and the nodes after it still run, seeing no value from it.
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["adjustments"]["status"] == RunnableStatus.FAILURE
    assert "all match, but a unique table allows one" in result.output["adjustments"]["error"]["message"]
    assert result.output["end"]["status"] == RunnableStatus.SUCCESS
    assert result.output["end"]["output"] == {"llpa": None}


def test_a_trace_keeps_the_first_rules_and_the_total_while_the_dump_keeps_them_all(mock_tracing_client):
    rules = [DecisionRule(id=f"r{i}", name=f"rule {i}", when=[f">= {i}", "", ""], then=["0.1", ""]) for i in range(60)]
    table = adjustments_table(hit_policy="first", rules=rules)
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    table.run(input_data={"fico": 700, "ltv": 85, "program": "FHA"}, config=RunnableConfig(callbacks=[tracing]))

    traced = next(run for run in tracing.runs.values() if run.name == "adjustments").metadata["node"]
    assert len(traced["rules"]) == 50
    assert traced["rules_count"] == 60
    assert len(table.to_dict()["rules"]) == 60
    assert "rules_count" not in table.to_dict()
