"""Which spans a run produces and how they nest."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from opentelemetry import trace

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.operators import Map
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import (
    RUN_ID,
    RUN_TYPE,
    AsyncEcho,
    AsyncInstrumented,
    Calling,
    Echo,
    Instrumented,
    parent_span_id,
    span_named,
    spans_named,
)

pytest.importorskip("opentelemetry.sdk")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

FLOW_RUN_ID = "dynamiq.flow.run_id"


def two_step_workflow(first: Node | None = None, second: Node | None = None) -> Workflow:
    first = first or Echo(id="first", name="first")
    second = second or Echo(id="second", name="second")
    second.depends_on(first)
    return Workflow(name="steps", flow=Flow(nodes=[first, second]))


def test_workflow_span_is_the_parent_of_its_node_spans(exporter, traced):
    result = two_step_workflow().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    spans = exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == ["first", "invoke_workflow steps", "second"]
    workflow_span = span_named(exporter, "invoke_workflow steps")
    assert workflow_span.parent is None
    assert workflow_span.attributes[RUN_TYPE] == "workflow"
    for name in ("first", "second"):
        node_span = span_named(exporter, name)
        assert parent_span_id(node_span) == workflow_span.context.span_id
        assert node_span.context.trace_id == workflow_span.context.trace_id
        assert node_span.attributes[RUN_TYPE] == "node"


def test_flow_of_a_workflow_has_no_span_and_is_named_on_the_workflow_span(exporter, traced):
    two_step_workflow().run(input_data={"value": 1}, config=traced)

    workflow_span = span_named(exporter, "invoke_workflow steps")
    run_ids = {span.attributes[RUN_ID] for span in exporter.get_finished_spans()}
    assert len(run_ids) == 3
    # The flow run has an id of its own, which no span carries as its run id.
    assert workflow_span.attributes[FLOW_RUN_ID] not in run_ids


def test_workflow_joins_the_trace_that_is_current(exporter, provider, traced):
    with provider.get_tracer("application").start_as_current_span("request") as request:
        two_step_workflow().run(input_data={"value": 1}, config=traced)

    workflow_span = span_named(exporter, "invoke_workflow steps")
    assert parent_span_id(workflow_span) == request.get_span_context().span_id
    assert {span.context.trace_id for span in exporter.get_finished_spans()} == {request.get_span_context().trace_id}


def test_workflow_span_is_not_current_after_or_between_runs(exporter, traced):
    # Only node spans are made current, and only while the node runs.
    two_step_workflow().run(input_data={"value": 1}, config=traced)

    assert not trace.get_current_span().get_span_context().is_valid


def test_standalone_flow_has_a_root_span(exporter, traced):
    flow = Flow(name="alone", nodes=[Echo(id="only", name="only")])

    result = flow.run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    flow_span = span_named(exporter, "invoke_workflow alone")
    assert flow_span.parent is None
    assert flow_span.attributes[RUN_TYPE] == "flow"
    assert FLOW_RUN_ID not in flow_span.attributes
    assert parent_span_id(span_named(exporter, "only")) == flow_span.context.span_id


@pytest.mark.asyncio
async def test_standalone_async_flow_has_a_root_span(exporter, traced):
    # Flow.run_async names the flow run itself as its parent run when it has none.
    flow = Flow(name="alone", nodes=[AsyncEcho(id="only", name="only")])

    result = await flow.run_async(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    flow_span = span_named(exporter, "invoke_workflow alone")
    assert flow_span.parent is None
    assert parent_span_id(span_named(exporter, "only")) == flow_span.context.span_id
    assert len(exporter.get_finished_spans()) == 2


@pytest.mark.asyncio
async def test_async_workflow_span_is_the_parent_of_its_node_spans(exporter, traced):
    # One node with a native async execute, one that runs in the flow's thread pool.
    workflow = two_step_workflow(AsyncEcho(id="first", name="first"), Echo(id="second", name="second"))

    result = await workflow.run_async(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    workflow_span = span_named(exporter, "invoke_workflow steps")
    assert len(exporter.get_finished_spans()) == 3
    for name in ("first", "second"):
        assert parent_span_id(span_named(exporter, name)) == workflow_span.context.span_id


def test_standalone_node_has_a_root_span(exporter, traced):
    result = Echo().run(input_data={"value": 1}, config=traced)

    assert result.output == {"value": 1}
    (span,) = exporter.get_finished_spans()
    assert span.name == "echo"
    assert span.parent is None
    assert span.attributes[RUN_TYPE] == "node"


@pytest.mark.asyncio
@pytest.mark.parametrize("node_class", [Echo, AsyncEcho], ids=["thread", "native"])
async def test_standalone_async_node_has_a_root_span(exporter, traced, node_class):
    result = await node_class(name="node").run_async(input_data={"value": 1}, config=traced)

    assert result.output == {"value": 1}
    (span,) = exporter.get_finished_spans()
    assert span.name == "node"
    assert span.parent is None


@pytest.mark.parametrize("timeout_seconds", [None, 5], ids=["inline", "timeout-thread"])
def test_span_started_inside_a_node_is_a_child_of_the_node_span(exporter, provider, traced, timeout_seconds):
    node = Instrumented(error_handling=ErrorHandling(timeout_seconds=timeout_seconds))
    node._tracer = provider.get_tracer("client-library")

    node.run(input_data={}, config=traced)

    node_span = span_named(exporter, "instrumented")
    assert parent_span_id(span_named(exporter, "inner")) == node_span.context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [None, 5], ids=["inline", "wait-for"])
async def test_span_started_inside_an_async_node_is_a_child_of_the_node_span(
    exporter, provider, traced, timeout_seconds
):
    node = AsyncInstrumented(error_handling=ErrorHandling(timeout_seconds=timeout_seconds))
    node._tracer = provider.get_tracer("client-library")

    await node.run_async(input_data={}, config=traced)

    node_span = span_named(exporter, "async-instrumented")
    assert parent_span_id(span_named(exporter, "inner")) == node_span.context.span_id


@pytest.mark.parametrize("parent_run_id_passed", [True, False], ids=["by-run-id", "by-current-span"])
def test_node_run_by_a_node_is_its_child(exporter, traced, parent_run_id_passed):
    outer = Calling(inner=Echo(name="inner"), parent_run_id_passed=parent_run_id_passed)

    outer.run(input_data={}, config=traced)

    outer_span = span_named(exporter, "calling")
    inner_span = span_named(exporter, "inner")
    assert parent_span_id(inner_span) == outer_span.context.span_id
    assert inner_span.context.trace_id == outer_span.context.trace_id


def test_every_map_item_is_a_child_of_the_map_span(exporter, traced):
    mapper = Map(id="map", name="each", node=Echo(name="item"), max_workers=8)

    result = mapper.run(input_data={"input": [{"value": index} for index in range(20)]}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    map_span = span_named(exporter, "each")
    item_spans = spans_named(exporter, "item")
    assert len(item_spans) == 20
    assert {parent_span_id(span) for span in item_spans} == {map_span.context.span_id}
    assert len({span.attributes[RUN_ID] for span in item_spans}) == 20


def test_parallel_branches_are_children_of_the_workflow_span(exporter, traced):
    branches = [Echo(id=f"branch-{index}", name=f"branch-{index}") for index in range(8)]
    join = Echo(id="join", name="join")
    join.depends_on(branches)
    workflow = Workflow(name="fan", flow=Flow(nodes=[*branches, join]))

    result = workflow.run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    workflow_span = span_named(exporter, "invoke_workflow fan")
    node_spans = [span for span in exporter.get_finished_spans() if span is not workflow_span]
    assert len(node_spans) == 9
    assert {parent_span_id(span) for span in node_spans} == {workflow_span.context.span_id}


@pytest.mark.asyncio
async def test_parallel_async_branches_are_children_of_the_workflow_span(exporter, traced):
    branches = [(AsyncEcho if index % 2 else Echo)(id=f"branch-{index}", name=f"branch-{index}") for index in range(8)]
    workflow = Workflow(name="fan", flow=Flow(nodes=branches))

    result = await workflow.run_async(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    workflow_span = span_named(exporter, "invoke_workflow fan")
    node_spans = [span for span in exporter.get_finished_spans() if span is not workflow_span]
    assert len(node_spans) == 8
    assert {parent_span_id(span) for span in node_spans} == {workflow_span.context.span_id}


def test_two_workflows_at_once_keep_their_own_trees(exporter, provider):
    def run(index: int):
        workflow = Workflow(name=f"wf-{index}", flow=Flow(nodes=[Echo(id="n", name=f"node-{index}")]))
        return workflow.run(
            input_data={"value": index}, config=RunnableConfig(tracing=TracingConfig(tracer_provider=provider))
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(16)))

    assert all(result.status == RunnableStatus.SUCCESS for result in results)
    for index in range(16):
        workflow_span = span_named(exporter, f"invoke_workflow wf-{index}")
        node_span = span_named(exporter, f"node-{index}")
        assert workflow_span.parent is None
        assert parent_span_id(node_span) == workflow_span.context.span_id
