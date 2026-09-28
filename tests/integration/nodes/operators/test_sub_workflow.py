import time
from typing import ClassVar

import pytest
from pydantic import PrivateAttr

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.flows import Flow
from dynamiq.nodes import InputTransformer, Node, NodeGroup, OutputTransformer
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import ErrorHandling, NodeDependency, NodeOutputReference
from dynamiq.nodes.operators import Choice, ChoiceOption, DecisionTable, Expression, Map, Pass, SubWorkflow
from dynamiq.nodes.types import (
    ChoiceCondition,
    ConditionOperator,
    DecisionRule,
    ExpressionItem,
    NamedField,
    SubWorkflowField,
)
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.dry_run import DryRunConfig


class Sleeper(Node):
    """Stands in for an inner node that takes longer than the caller allows."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "sleeper"
    seconds: float = 0.3

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(self.seconds)
        return {"done": True}


class Flaky(Node):
    """Fails the first time it runs, as a transient error would; counted on the class since each attempt runs a copy."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "flaky"
    calls: ClassVar[list[int]] = []

    def execute(self, input_data, config=None, **kwargs):
        self.calls.append(1)
        if len(self.calls) == 1:
            raise ValueError("transient")
        return {"done": True}


def flow_around(node: Node, flow_id: str) -> Flow:
    start = Input(id="in", name="in")
    node.depends = [NodeDependency(node=start)]
    end = Output(
        id="out",
        name="out",
        depends=[NodeDependency(node=node)],
        input_transformer=InputTransformer(selector={"done": f"$.{node.id}.output.done"}),
    )
    return Flow(id=flow_id, nodes=[start, node, end])


def eligibility_flow() -> Flow:
    """Input → decision table → Output, the shape every workflow built in the editor has."""
    start = Input(id="start", name="start")
    table = DecisionTable(
        id="table",
        name="eligibility",
        input_columns=[NamedField(id="fico", name="fico", type="int")],
        output_columns=[NamedField(id="decision", name="decision", type="string")],
        rules=[
            DecisionRule(id="r1", name="prime", when=[">= 740"], then=["approve"]),
            DecisionRule(id="r2", name="rest", when=[""], then=["decline"]),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"fico": "$.start.output.fico"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=table)],
        input_transformer=InputTransformer(
            selector={"decision": "$.table.output.decision", "rules": "$.table.output.matched_rules"}
        ),
    )
    return Flow(id="eligibility-flow", name="Eligibility", nodes=[start, table, end])


def sub_workflow(**overrides) -> SubWorkflow:
    fields = {
        "id": "check",
        "name": "eligibility_check",
        "flow": eligibility_flow(),
        "workflow_id": "wf-eligibility",
        "input_fields": [SubWorkflowField(name="fico", type="int", required=True)],
        "output_fields": [SubWorkflowField(name="decision", type="string")],
    }
    return SubWorkflow(**(fields | overrides))


def outer_workflow() -> Workflow:
    start = Input(id="outer-start", name="start")
    check = sub_workflow(
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"fico": "$.outer-start.output.fico"}),
    )
    end = Output(
        id="outer-end",
        name="end",
        depends=[NodeDependency(node=check)],
        input_transformer=InputTransformer(selector={"decision": "$.check.output.decision"}),
    )
    return Workflow(id="outer", flow=Flow(id="outer-flow", nodes=[start, check, end]))


def test_runs_the_flow_and_returns_its_output_node_result_traced_under_the_node(mock_tracing_client):
    tracing = TracingCallbackHandler(client=mock_tracing_client())
    workflow = outer_workflow()

    result = workflow.run(input_data={"fico": 760}, config=RunnableConfig(callbacks=[tracing]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["check"]["output"] == {"decision": "approve", "rules": [{"id": "r1", "name": "prime"}]}
    assert result.output["outer-end"]["output"] == {"decision": "approve"}

    runs = {str(run.id): run for run in tracing.runs.values()}
    check_run = next(run for run in runs.values() if run.name == "eligibility_check")
    inner_flow_run = next(
        run for run in runs.values() if run.type == RunType.FLOW and run.parent_run_id == check_run.id
    )
    inner_node_runs = [run for run in runs.values() if run.parent_run_id == inner_flow_run.id]
    assert inner_flow_run.metadata["flow"]["id"] == "eligibility-flow"
    assert sorted(run.name for run in inner_node_runs) == ["eligibility", "end", "start"]
    assert len(check_run.executions) == 1


def test_a_missing_required_input_fails_before_the_flow_runs():
    result = sub_workflow().run(input_data={"program": "FHA"}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message == "Sub-workflow 'eligibility_check': required inputs missing: fico"


def test_a_failing_inner_node_fails_the_sub_workflow():
    broken = Expression(id="broken", name="broken", expressions=[ExpressionItem(key="x", expression="1 / 0")])
    node = SubWorkflow(id="check", name="check", flow=Flow(id="broken-flow", nodes=[broken]))

    result = node.run(input_data={}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert result.error.message == "Sub-workflow 'check' failed: broken: division by zero"


def test_a_flow_that_calls_itself_is_refused():
    node = SubWorkflow(id="loop", name="loop")
    node.flow = Flow(id="loop-flow", nodes=[node])

    result = node.run(input_data={}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert "flow 'loop-flow' is already running above this node" in result.error.message


def test_a_flow_without_an_output_node_returns_every_node_output():
    node = SubWorkflow(
        id="check",
        name="check",
        flow=Flow(
            id="bare",
            nodes=[Expression(id="double", name="double", expressions=[ExpressionItem(key="x", expression="x * 2")])],
        ),
    )

    assert node.run(input_data={"x": 2}, config=RunnableConfig(callbacks=[])).output == {"double": {"x": 4}}


def test_map_runs_the_sub_workflow_per_item_in_parallel_without_sharing_flow_state():
    node = sub_workflow()
    batch = Map(id="batch", name="batch", node=node, max_workers=4)
    applications = [{"fico": 760}, {"fico": 600}] * 4

    result = batch.run(input_data={"input": applications}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert [item["decision"] for item in result.output["output"]] == ["approve", "decline"] * 4
    # Clones report the rules by their original ids, the ones the table was authored with.
    assert result.output["output"][0]["rules"] == [{"id": "r1", "name": "prime"}]
    assert node.flow.id == "eligibility-flow" and [n.id for n in node.flow.nodes] == ["start", "table", "end"]
    assert node.run(input_data={"fico": 760}, config=RunnableConfig(callbacks=[])).output["decision"] == "approve"


def test_two_nodes_holding_the_same_flow_run_in_parallel_without_sharing_its_state():
    flow = eligibility_flow()
    start = Input(id="outer-start", name="start")
    prime = sub_workflow(
        id="prime",
        name="prime",
        flow=flow,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"fico": "$.outer-start.output.prime"}),
    )
    weak = sub_workflow(
        id="weak",
        name="weak",
        flow=flow,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"fico": "$.outer-start.output.weak"}),
    )
    end = Output(
        id="outer-end",
        name="end",
        depends=[NodeDependency(node=prime), NodeDependency(node=weak)],
        input_transformer=InputTransformer(
            selector={"prime": "$.prime.output.decision", "weak": "$.weak.output.decision"}
        ),
    )
    workflow = Workflow(flow=Flow(id="outer-flow", nodes=[start, prime, weak, end]))

    # Both branches are ready at once, so the flow runs them in parallel threads against one flow object;
    # each run has to execute its own copy or the two overwrite each other's results.
    for _ in range(10):
        result = workflow.run(input_data={"prime": 760, "weak": 600}, config=RunnableConfig(callbacks=[]))

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["outer-end"]["output"] == {"prime": "approve", "weak": "decline"}


def test_a_timeout_fails_the_node_and_a_later_run_is_unaffected():
    node = SubWorkflow(
        id="check",
        name="check",
        flow=flow_around(Sleeper(id="slow"), "slow-flow"),
        error_handling=ErrorHandling(timeout_seconds=0.05),
    )

    result = node.run(input_data={}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.FAILURE
    assert result.error.type.__name__ == "TimeoutError"
    # The timed-out run carries on in its own thread on its own copy of the flow, so it cannot
    # disturb the next run.
    node.error_handling = ErrorHandling(timeout_seconds=5)
    assert node.run(input_data={}, config=RunnableConfig(callbacks=[])).output == {"done": True}


def test_a_retry_runs_the_flow_again_and_traces_every_attempt(mock_tracing_client):
    Flaky.calls.clear()
    node = SubWorkflow(
        id="check",
        name="check",
        flow=flow_around(Flaky(id="flaky"), "flaky-flow"),
        error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0),
    )
    tracing = TracingCallbackHandler(client=mock_tracing_client())

    result = node.run(input_data={}, config=RunnableConfig(callbacks=[tracing]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"done": True}
    assert len(Flaky.calls) == 2
    node_run = next(run for run in tracing.runs.values() if run.name == "check")
    assert len(node_run.executions) == 2
    assert [run.name for run in tracing.runs.values() if run.name == "flaky"] == ["flaky", "flaky"]


def test_the_parent_checkpoint_settings_do_not_reach_the_flow():
    node = sub_workflow()
    # A resume id names a checkpoint of the parent's run; the flow the node runs has none to load.
    config = RunnableConfig(callbacks=[], checkpoint=CheckpointConfig(enabled=True, resume_from="parent-checkpoint"))

    result = node.run(input_data={"fico": 760}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["decision"] == "approve"


def test_a_map_of_maps_keeps_the_inner_flow_wired():
    batches = Map(id="batches", name="batches", node=Map(id="batch", name="batch", node=sub_workflow(), max_workers=2))

    result = batches.run(
        input_data={"input": [{"input": [{"fico": 760}, {"fico": 600}]}, {"input": [{"fico": 700}]}]},
        config=RunnableConfig(callbacks=[]),
    )

    # The node is cloned and re-identified once per level, so the inner flow's paths move twice.
    assert result.status == RunnableStatus.SUCCESS
    assert [[item["decision"] for item in batch["output"]] for batch in result.output["output"]] == [
        ["approve", "decline"],
        ["decline"],
    ]


def test_a_map_keeps_the_choice_gates_inside_the_flow():
    start = Input(id="start", name="start")
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="opt-hi",
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.score", value=50
                ),
            ),
            ChoiceOption(id="opt-lo"),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"score": "$.start.output.score"}),
    )
    hi = Expression(
        id="hi",
        name="hi",
        expressions=[ExpressionItem(key="band", expression="'HIGH'")],
        depends=[NodeDependency(node=route, option="opt-hi")],
    )
    lo = Expression(
        id="lo",
        name="lo",
        expressions=[ExpressionItem(key="band", expression="'LOW'")],
        depends=[NodeDependency(node=route, option="opt-lo")],
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=hi), NodeDependency(node=lo)],
        input_transformer=InputTransformer(selector={"hi": "$.hi.output.band", "lo": "$.lo.output.band"}),
    )
    node = SubWorkflow(id="bands", name="bands", flow=Flow(id="bands-flow", nodes=[start, route, hi, lo, end]))
    batch = Map(id="batch", name="batch", node=node, max_workers=2)

    result = batch.run(input_data={"input": [{"score": 10}, {"score": 90}]}, config=RunnableConfig(callbacks=[]))

    # The cloned Choice's option ids change with the clone; a gate on the old id would let every branch run.
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"hi": None, "lo": "LOW"}, {"hi": "HIGH", "lo": None}]


def test_a_map_keeps_the_gates_of_two_choices_sharing_an_option_id():
    """Two Choices in one flow each own an option called `go`; a gate follows the option of its own Choice."""
    start = Input(id="start", name="start")

    def route(node_id: str, key: str) -> Choice:
        return Choice(
            id=node_id,
            name=node_id,
            options=[
                ChoiceOption(
                    id="go",
                    condition=ChoiceCondition(
                        operator=ConditionOperator.NUMERIC_GREATER_THAN, variable=f"$.start.output.{key}", value=0
                    ),
                ),
                ChoiceOption(id="default"),
            ],
            depends=[NodeDependency(node=start)],
        )

    route_a, route_b = route("route_a", "a"), route("route_b", "b")
    a_go = Expression(
        id="a_go",
        name="a_go",
        expressions=[ExpressionItem(key="band", expression="'A'")],
        depends=[NodeDependency(node=route_a, option="go")],
    )
    b_go = Expression(
        id="b_go",
        name="b_go",
        expressions=[ExpressionItem(key="band", expression="'B'")],
        depends=[NodeDependency(node=route_b, option="go")],
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=a_go), NodeDependency(node=b_go)],
        input_transformer=InputTransformer(selector={"a": "$.a_go.output.band", "b": "$.b_go.output.band"}),
    )
    review = SubWorkflow(
        id="review", name="review", flow=Flow(id="review-flow", nodes=[start, route_a, route_b, a_go, b_go, end])
    )
    batch = Map(id="batch", name="batch", node=review, max_workers=2)

    result = batch.run(input_data={"input": [{"a": 1, "b": 0}, {"a": 0, "b": 1}]}, config=RunnableConfig(callbacks=[]))

    # A gate moved to the other Choice's option finds no result under it and would let the branch run.
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"a": "A", "b": None}, {"a": None, "b": "B"}]


def test_a_map_keeps_nested_flows_that_spell_a_node_id_alike_wired():
    """The outer and the inner flow both hold a `start`; every reader follows its own flow's copy."""
    inner_start = Input(id="start", name="start")
    double = Expression(
        id="double",
        name="double",
        input_fields=[NamedField(name="x")],
        expressions=[ExpressionItem(key="doubled", expression="x * 2")],
        depends=[NodeDependency(node=inner_start)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x"}),
    )
    inner_end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=double)],
        input_transformer=InputTransformer(selector={"doubled": "$.double.output.doubled"}),
    )
    outer_start = Input(id="start", name="start")
    inner = SubWorkflow(
        id="inner",
        name="inner",
        flow=Flow(id="inner-flow", nodes=[inner_start, double, inner_end]),
        depends=[NodeDependency(node=outer_start)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x"}),
    )
    outer_end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=inner)],
        input_transformer=InputTransformer(selector={"x": "$.start.output.x", "doubled": "$.inner.output.doubled"}),
    )
    outer = SubWorkflow(id="outer", name="outer", flow=Flow(id="outer-flow", nodes=[outer_start, inner, outer_end]))
    batch = Map(id="batch", name="batch", node=outer, max_workers=2)

    result = batch.run(input_data={"input": [{"x": 2}, {"x": 5}]}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"x": 2, "doubled": 4}, {"x": 5, "doubled": 10}]


def test_a_map_leaves_a_selector_that_names_an_input_key_alone():
    start = Input(id="start", name="start")
    table = DecisionTable(
        id="table",
        name="table",
        # The column's id reads like the input key the selector names.
        input_columns=[NamedField(id="fico", name="fico", type="int")],
        output_columns=[NamedField(id="decision", name="decision", type="string")],
        rules=[
            DecisionRule(id="r1", name="prime", when=[">= 740"], then=["approve"]),
            DecisionRule(id="r2", name="rest", when=[""], then=["decline"]),
        ],
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"fico": "$.fico"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=table)],
        input_transformer=InputTransformer(selector={"decision": "$.table.output.decision"}),
    )
    node = SubWorkflow(id="check", name="check", flow=Flow(id="check-flow", nodes=[start, table, end]))
    batch = Map(id="batch", name="batch", node=node, max_workers=2)

    result = batch.run(input_data={"input": [{"fico": 760}, {"fico": 600}]}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"decision": "approve"}, {"decision": "decline"}]


def test_a_map_keeps_the_output_references_inside_the_flow():
    start = Input(id="start", name="start")
    calc = Expression(
        id="calc",
        name="calc",
        expressions=[ExpressionItem(key="doubled", expression="score * 2")],
        depends=[NodeDependency(node=start)],
        input_mapping={"score": NodeOutputReference(node=start, output_key="score")},
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=calc)],
        input_transformer=InputTransformer(selector={"doubled": "$.calc.output.doubled"}),
    )
    node = SubWorkflow(id="double", name="double", flow=Flow(id="double-flow", nodes=[start, calc, end]))
    batch = Map(id="batch", name="batch", node=node, max_workers=2)

    result = batch.run(input_data={"input": [{"score": 5}, {"score": 8}]}, config=RunnableConfig(callbacks=[]))

    # A reference wired with `.inputs(...)` must point at the copied node, or the clone reads a node that never runs.
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"doubled": 10}, {"doubled": 16}]


def test_a_sub_workflow_held_as_an_agent_tool_dumps_with_its_flow_and_reloads(tmp_path):
    agent = Agent(
        id="agent",
        name="agent",
        llm=OpenAI(id="llm", name="llm", model="gpt-4o-mini", connection=connections.OpenAI(api_key="test")),
        tools=[sub_workflow()],
    )
    workflow = Workflow(id="agentic", flow=Flow(id="agent-flow", nodes=[agent]))
    path = tmp_path / "agent.yaml"

    data = workflow.to_yaml_file_data()
    workflow.to_yaml_file(str(path))
    reloaded = Workflow.from_yaml_file(str(path), init_components=True)

    # A flow held in a list field is emitted like one held directly, so the file reads back.
    assert set(data.flows) == {"agent-flow", "eligibility-flow"}
    assert {"start", "table", "end"} <= set(data.nodes)
    reloaded_agent = next(node for node in reloaded.flow.nodes if node.id == "agent")
    assert reloaded_agent.tools[0].flow.id == "eligibility-flow"
    assert [node.id for node in reloaded_agent.tools[0].flow.nodes] and reloaded.flow.id == "agent-flow"


def test_the_dump_names_the_sub_flow_and_a_double_round_trip_still_runs(tmp_path):
    workflow = outer_workflow()
    first = tmp_path / "outer.yaml"
    workflow.to_yaml_file(first)
    text = first.read_text()

    assert "flow: eligibility-flow" in text
    assert "  eligibility-flow:" in text

    loaded = Workflow.from_yaml_file(str(first), init_components=True)
    second = tmp_path / "outer_again.yaml"
    loaded.to_yaml_file(second)
    reloaded = Workflow.from_yaml_file(str(second), init_components=True)

    check = next(node for node in reloaded.flow.nodes if isinstance(node, SubWorkflow))
    assert check.workflow_id == "wf-eligibility"
    assert check.flow.id == "eligibility-flow"
    assert sorted(node.id for node in check.flow.nodes) == ["end", "start", "table"]
    assert [field.required for field in check.input_fields] == [True]
    assert reloaded.run(input_data={"fico": 500}).output["outer-end"]["output"] == {"decision": "decline"}


def test_a_map_holding_a_sub_workflow_round_trips_through_yaml(tmp_path):
    workflow = Workflow(
        id="batch-workflow",
        flow=Flow(id="batch-flow", nodes=[Map(id="batch", name="batch", node=sub_workflow(), max_workers=2)]),
    )
    path = tmp_path / "batch.yaml"
    workflow.to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    result = loaded.run(input_data={"input": [{"fico": 760}, {"fico": 500}]})

    assert [item["decision"] for item in result.output["batch"]["output"]["output"]] == ["approve", "decline"]


def test_a_sub_workflow_two_levels_deep_round_trips_through_yaml(tmp_path):
    mid_start = Input(id="mid-start", name="start")
    inner = sub_workflow(
        id="inner",
        name="inner",
        depends=[NodeDependency(node=mid_start)],
        input_transformer=InputTransformer(selector={"fico": "$.mid-start.output.fico"}),
    )
    mid_end = Output(
        id="mid-end",
        name="end",
        depends=[NodeDependency(node=inner)],
        input_transformer=InputTransformer(selector={"decision": "$.inner.output.decision"}),
    )
    outer_start = Input(id="outer-start", name="start")
    mid = sub_workflow(
        id="mid",
        name="mid",
        flow=Flow(id="mid-flow", name="Mid", nodes=[mid_start, inner, mid_end]),
        depends=[NodeDependency(node=outer_start)],
        input_transformer=InputTransformer(selector={"fico": "$.outer-start.output.fico"}),
    )
    outer_end = Output(
        id="outer-end",
        name="end",
        depends=[NodeDependency(node=mid)],
        input_transformer=InputTransformer(selector={"decision": "$.mid.output.decision"}),
    )
    workflow = Workflow(id="outer", flow=Flow(id="outer-flow", nodes=[outer_start, mid, outer_end]))
    path = tmp_path / "nested.yaml"

    workflow.to_yaml_file(str(path))
    reloaded = Workflow.from_yaml_file(str(path), init_components=True)

    # The innermost flow is built first, so the node two levels down finds the flow it holds.
    mid_node = next(node for node in reloaded.flow.nodes if node.id == "mid")
    inner_node = next(node for node in mid_node.flow.nodes if node.id == "inner")
    assert inner_node.flow.id == "eligibility-flow"
    result = reloaded.run(input_data={"fico": 760}, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["outer-end"]["output"] == {"decision": "approve"}


def test_a_dump_refuses_flows_that_reuse_ids():
    outer_start = Input(id="outer-start", name="start")
    first = sub_workflow(id="first", name="first", depends=[NodeDependency(node=outer_start)])
    second = sub_workflow(id="second", name="second", depends=[NodeDependency(node=outer_start)])
    workflow = Workflow(flow=Flow(id="outer-flow", nodes=[outer_start, first, second]))

    # Each node holds its own copy of the eligibility flow, so its id is taken twice.
    with pytest.raises(ValueError, match="Flow id 'eligibility-flow' is used by two different flows"):
        workflow.to_yaml_file_data()

    # With distinct flow ids the node ids under them still collide in the id-keyed nodes section.
    second.flow = Flow(id="eligibility-flow-2", name="Eligibility", nodes=eligibility_flow().nodes)
    with pytest.raises(ValueError, match="Node id 'start' is used by two different nodes"):
        workflow.to_yaml_file_data()


def test_a_flow_that_holds_itself_is_a_loader_error(tmp_path):
    path = tmp_path / "loop.yaml"
    path.write_text(
        "nodes:\n"
        "  again:\n"
        "    type: dynamiq.nodes.operators.SubWorkflow\n"
        "    flow: loop-flow\n"
        "flows:\n"
        "  loop-flow:\n"
        "    nodes: [again]\n"
        "workflows:\n"
        "  workflow:\n"
        "    flow: loop-flow\n"
    )

    with pytest.raises(Exception, match="Flow 'loop-flow' holds itself through loop-flow -> loop-flow"):
        Workflow.from_yaml_file(str(path), init_components=True)


def test_a_missing_referenced_flow_is_a_loader_error(tmp_path):
    path = tmp_path / "dangling.yaml"
    path.write_text(
        "nodes:\n"
        "  check:\n"
        "    type: dynamiq.nodes.operators.SubWorkflow\n"
        "    flow: missing-flow\n"
        "flows:\n"
        "  outer:\n"
        "    nodes: [check]\n"
        "workflows:\n"
        "  workflow:\n"
        "    flow: outer\n"
    )

    with pytest.raises(Exception, match="Flow 'missing-flow' for node 'check' not found"):
        Workflow.from_yaml_file(str(path), init_components=True)


def test_a_map_leaves_a_dependency_condition_that_reads_the_result_alone():
    """An Output named `output` gated on `$.output.score`: run directly, under a Map, and directly again."""
    start = Input(id="input", name="input")
    end = Output(
        id="output",
        name="output",
        depends=[
            NodeDependency(
                node=start,
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.output.score", value=5
                ),
            )
        ],
        input_transformer=InputTransformer(selector={"score": "$.input.output.score"}),
    )
    flow = Flow(id="gated-flow", nodes=[start, end])
    config = RunnableConfig(callbacks=[])

    before = Workflow(flow=flow).run(input_data={"score": 10}, config=config)
    assert before.status == RunnableStatus.SUCCESS
    assert before.output["output"]["output"] == {"score": 10}

    batch = Map(id="batch", name="batch", node=SubWorkflow(id="sw", name="sw", flow=flow), max_workers=2)
    result = batch.run(input_data={"input": [{"score": 10}, {"score": 20}]}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"score": 10}, {"score": 20}]
    # The flow the Map copied is as it was: the gate still reads the result, and runs again.
    assert end.depends[0].condition.variable == "$.output.score"
    after = Workflow(flow=flow).run(input_data={"score": 10}, config=config)
    assert after.status == RunnableStatus.SUCCESS
    assert after.output["output"]["output"] == {"score": 10}


def test_a_map_keeps_a_sub_workflow_selecting_from_every_inner_output_wired():
    """A flow without an Output node returns every node's output by id; the node's own selector reads them."""
    inner_start = Input(id="inner_start", name="inner_start")
    inner_calc = Expression(
        id="inner_calc",
        name="inner_calc",
        depends=[NodeDependency(node=inner_start)],
        input_fields=[NamedField(name="x")],
        expressions=[ExpressionItem(key="doubled", expression="x * 2")],
        input_transformer=InputTransformer(selector={"x": "$.inner_start.output.x"}),
    )
    node = SubWorkflow(
        id="keyed",
        name="keyed",
        flow=Flow(id="keyed_flow", nodes=[inner_start, inner_calc]),
        output_transformer=OutputTransformer(selector={"result": "$.inner_calc.doubled"}),
    )
    config = RunnableConfig(callbacks=[])

    assert node.run(input_data={"x": 4}, config=config).output == {"result": 8}

    batch = Map(id="batch", name="batch", node=node, max_workers=2)
    result = batch.run(input_data={"input": [{"x": 4}, {"x": 5}]}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"result": 8}, {"result": 10}]


def test_a_map_keeps_a_flow_wired_in_the_bracket_form():
    start = Input(id="start", name="start")
    calc = Expression(
        id="calc",
        name="calc",
        depends=[NodeDependency(node=start)],
        input_fields=[NamedField(name="x")],
        expressions=[ExpressionItem(key="doubled", expression="x * 2")],
        input_transformer=InputTransformer(selector={"x": "$['start'].output.x"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=calc)],
        input_transformer=InputTransformer(selector={"doubled": "$['calc'].output.doubled"}),
    )
    node = SubWorkflow(id="sw", name="sw", flow=Flow(id="bracket_flow", nodes=[start, calc, end]))
    config = RunnableConfig(callbacks=[])

    assert node.run(input_data={"x": 4}, config=config).output == {"doubled": 8}

    batch = Map(id="batch", name="batch", node=node, max_workers=2)
    result = batch.run(input_data={"input": [{"x": 4}, {"x": 5}]}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"doubled": 8}, {"doubled": 10}]


def test_a_map_keeps_a_choice_gate_that_names_a_node_by_id():
    """Input → Choice on `$.start.output.score` → Pass → Output, cloned per Map item."""
    start = Input(id="start", name="start")
    route = Choice(
        id="route",
        name="route",
        options=[
            ChoiceOption(
                id="opt-hi",
                name="hi",
                condition=ChoiceCondition(
                    operator=ConditionOperator.NUMERIC_GREATER_THAN, variable="$.start.output.score", value=50
                ),
            ),
            ChoiceOption(id="opt-lo", name="lo"),
        ],
        depends=[NodeDependency(node=start)],
    )
    hi = Pass(
        id="hi",
        name="hi",
        depends=[NodeDependency(node=route, option="opt-hi")],
        input_transformer=InputTransformer(selector={"score": "$.start.output.score"}),
    )
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=hi)],
        input_transformer=InputTransformer(selector={"score": "$.hi.output.score"}),
    )
    review = SubWorkflow(id="review", name="review", flow=Flow(id="review-flow", nodes=[start, route, hi, end]))
    batch = Map(id="batch", name="batch", node=review, max_workers=2)

    result = batch.run(input_data={"input": [{"score": 90}, {"score": 95}]}, config=RunnableConfig(callbacks=[]))

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["output"] == [{"score": 90}, {"score": 95}]


STORE: dict[str, str] = {}
CLEANED: list[int] = []


class StoreWriter(Node):
    """Stands in for a writer: ingests into the module store and removes what it wrote when cleaned up."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "store_writer"
    _written: list[str] = PrivateAttr(default_factory=list)

    def execute(self, input_data, config=None, **kwargs):
        documents = list(input_data["documents"])
        for document in documents:
            STORE[document] = self.id
        # Rebound rather than extended: a copy of this node shares the list object with the original.
        self._written = [*self._written, *documents]
        return {"count": len(documents)}

    def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
        CLEANED.append(id(self))
        for document in self._written:
            STORE.pop(document, None)
        self._written = []


class StoreReader(Node):
    """Stands in for a retriever: reads whatever the store holds when it runs."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "store_reader"

    def execute(self, input_data, config=None, **kwargs):
        return {"documents": sorted(STORE)}


class Breaks(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "breaks"

    def execute(self, input_data, config=None, **kwargs):
        raise RuntimeError("broken")


def ingestion_flow(then: Node | None = None) -> Flow:
    """Input → writer [→ then] → Output: an ingestion workflow run as a sub-workflow."""
    start = Input(id="in", name="in")
    writer = StoreWriter(
        id="writer",
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector={"documents": "$.in.output.documents"}),
    )
    last = writer
    if then is not None:
        then.depends = [NodeDependency(node=writer)]
        last = then
    end = Output(
        id="out",
        name="out",
        depends=[NodeDependency(node=last)],
        input_transformer=InputTransformer(selector={"count": "$.writer.output.count"}),
    )
    return Flow(id="ingestion-flow", nodes=[start, writer, *([then] if then is not None else []), end])


def reading_after(ingest: Node, selector: dict[str, str]) -> Workflow:
    """Input → ingest → reader → Output: the caller reads what the sub-workflow ingested."""
    start = Input(id="start", name="start")
    ingest.depends = [NodeDependency(node=start)]
    ingest.input_transformer = InputTransformer(selector=selector)
    reader = StoreReader(id="reader", depends=[NodeDependency(node=ingest)])
    end = Output(
        id="end",
        name="end",
        depends=[NodeDependency(node=reader)],
        input_transformer=InputTransformer(selector={"seen": "$.reader.output.documents"}),
    )
    return Workflow(id="caller", flow=Flow(id="caller-flow", nodes=[start, ingest, reader, end]))


@pytest.fixture
def empty_store():
    STORE.clear()
    CLEANED.clear()
    yield
    STORE.clear()
    CLEANED.clear()


def test_a_dry_run_cleans_the_sub_flow_writes_once_the_caller_ends(empty_store):
    inner = ingestion_flow()
    ingest = SubWorkflow(id="ingest", name="ingest", flow=inner)
    workflow = reading_after(ingest, {"documents": "$.start.output.documents"})

    result = workflow.run_sync(
        {"documents": ["b", "a"]}, config=RunnableConfig(dry_run=DryRunConfig(enabled=True), callbacks=[])
    )

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["end"]["output"] == {"seen": ["a", "b"]}
    assert STORE == {}
    # The copy that ran is what is cleaned up, once; the template never ran and holds nothing.
    template_writer = next(node for node in inner.nodes if isinstance(node, StoreWriter))
    assert len(CLEANED) == 1 and id(template_writer) not in CLEANED
    assert template_writer._written == []


def test_a_dry_run_cleans_what_a_failing_sub_flow_wrote(empty_store):
    ingest = SubWorkflow(id="ingest", name="ingest", flow=ingestion_flow(then=Breaks(id="breaks")))
    workflow = reading_after(ingest, {"documents": "$.start.output.documents"})

    result = workflow.run_sync(
        {"documents": ["a"]}, config=RunnableConfig(dry_run=DryRunConfig(enabled=True), callbacks=[])
    )

    assert result.status == RunnableStatus.FAILURE
    assert STORE == {}
    assert len(CLEANED) == 1


def test_a_map_dry_run_cleans_every_item_once_the_caller_ends(empty_store):
    ingest = SubWorkflow(id="ingest", name="ingest", flow=ingestion_flow())
    batch = Map(id="batch", name="batch", node=ingest, max_workers=3)
    workflow = reading_after(batch, {"input": "$.start.output.items"})

    result = workflow.run_sync(
        {"items": [{"documents": ["a"]}, {"documents": ["b"]}, {"documents": ["c"]}]},
        config=RunnableConfig(dry_run=DryRunConfig(enabled=True), callbacks=[]),
    )

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["end"]["output"] == {"seen": ["a", "b", "c"]}
    assert STORE == {}
    assert len(CLEANED) == 3
