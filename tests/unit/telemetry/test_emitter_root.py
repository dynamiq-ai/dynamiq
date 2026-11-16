"""Which span of a run tree says that its run has no Dynamiq parent run.

A backend receives the spans of a trace children first, so a span whose parent has not arrived looks like a top-most
one. ``dynamiq.run.is_root`` is how it tells a run that has no parent run from one whose parent is late.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from dynamiq import Workflow
from dynamiq.executors.pool import ProcessExecutor
from dynamiq.flows import Flow
from dynamiq.nodes.operators import Map
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import AsyncEcho, Calling, Echo, Failing, parent_span_id, span_named

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
sampling = pytest.importorskip("opentelemetry.sdk.trace.sampling")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

IS_ROOT = "dynamiq.run.is_root"


def two_step_workflow(name: str = "steps") -> Workflow:
    first = Echo(id="first", name="first")
    second = Echo(id="second", name="second")
    second.depends_on(first)
    return Workflow(name=name, flow=Flow(nodes=[first, second]))


def root_names(exporter) -> list[str]:
    """Names of the spans that carry the marker. Its only value is ``True``: a run that has a parent has none."""
    marked = [span for span in exporter.get_finished_spans() if IS_ROOT in span.attributes]
    assert all(span.attributes[IS_ROOT] is True for span in marked)
    return sorted(span.name for span in marked)


def test_workflow_run_marks_the_workflow_span_only(exporter, traced):
    result = two_step_workflow().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert len(exporter.get_finished_spans()) == 3
    assert root_names(exporter) == ["invoke_workflow steps"]


@pytest.mark.asyncio
async def test_async_workflow_run_marks_the_workflow_span_only(exporter, traced):
    workflow = Workflow(name="steps", flow=Flow(nodes=[AsyncEcho(id="first", name="first"), Echo(id="second")]))

    result = await workflow.run_async(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert len(exporter.get_finished_spans()) == 3
    assert root_names(exporter) == ["invoke_workflow steps"]


def test_standalone_flow_marks_the_flow_span_only(exporter, traced):
    flow = Flow(name="alone", nodes=[Echo(id="only", name="only")])

    flow.run(input_data={"value": 1}, config=traced)

    assert len(exporter.get_finished_spans()) == 2
    assert root_names(exporter) == ["invoke_workflow alone"]


@pytest.mark.asyncio
async def test_standalone_async_flow_marks_the_flow_span_only(exporter, traced):
    # Flow.run_async names the flow run itself as its parent run when it has none.
    flow = Flow(name="alone", nodes=[AsyncEcho(id="only", name="only")])

    await flow.run_async(input_data={"value": 1}, config=traced)

    assert len(exporter.get_finished_spans()) == 2
    assert root_names(exporter) == ["invoke_workflow alone"]


def test_standalone_node_is_a_root(exporter, traced):
    Echo().run(input_data={"value": 1}, config=traced)

    assert root_names(exporter) == ["echo"]


@pytest.mark.asyncio
@pytest.mark.parametrize("node_class", [Echo, AsyncEcho], ids=["thread", "native"])
async def test_standalone_async_node_is_a_root(exporter, traced, node_class):
    await node_class(name="node").run_async(input_data={"value": 1}, config=traced)

    assert root_names(exporter) == ["node"]


def test_workflow_inside_a_span_of_the_application_is_still_a_root(exporter, provider, traced):
    with provider.get_tracer("application").start_as_current_span("request") as request:
        two_step_workflow().run(input_data={"value": 1}, config=traced)

    workflow_span = span_named(exporter, "invoke_workflow steps")
    # Not the root of the trace, and still the top-most Dynamiq run in it.
    assert parent_span_id(workflow_span) == request.get_span_context().span_id
    assert len(exporter.get_finished_spans()) == 4
    assert root_names(exporter) == ["invoke_workflow steps"]


def test_every_top_level_run_of_one_trace_is_a_root(exporter, provider, traced):
    with provider.get_tracer("application").start_as_current_span("request"):
        two_step_workflow("one").run(input_data={"value": 1}, config=traced)
        two_step_workflow("two").run(input_data={"value": 2}, config=traced)

    assert len({span.context.trace_id for span in exporter.get_finished_spans()}) == 1
    assert root_names(exporter) == ["invoke_workflow one", "invoke_workflow two"]


def test_node_run_by_a_node_is_not_a_root(exporter, traced):
    Calling(inner=Calling(name="middle", inner=Echo(name="inner"))).run(input_data={}, config=traced)

    assert len(exporter.get_finished_spans()) == 3
    assert root_names(exporter) == ["calling"]


def test_node_run_without_a_parent_run_id_is_a_root_below_the_span_of_its_caller(exporter, traced):
    # Code that runs a node without naming the run it belongs to starts a run of its own, which callbacks also see
    # as one without a parent. Its span is still nested, through the span that is current.
    outer = Calling(inner=Echo(name="inner"), parent_run_id_passed=False)

    outer.run(input_data={}, config=traced)

    assert parent_span_id(span_named(exporter, "inner")) == span_named(exporter, "calling").context.span_id
    assert root_names(exporter) == ["calling", "inner"]


def test_map_items_are_not_roots(exporter, traced):
    mapper = Map(id="map", name="each", node=Echo(name="item"), max_workers=4)

    mapper.run(input_data={"input": [{"value": index} for index in range(6)]}, config=traced)

    assert len(exporter.get_finished_spans()) == 7
    assert root_names(exporter) == ["each"]


def test_failed_run_keeps_the_marker(exporter, traced):
    failing = Failing(id="failing", name="failing")

    result = Workflow(name="wf", flow=Flow(nodes=[failing])).run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert root_names(exporter) == ["invoke_workflow wf"]


def test_node_run_by_a_worker_process_is_not_a_root(exporter, provider, traced):
    # The worker has no registry entry for the flow run that submitted the node, only its run id.
    with provider.get_tracer("application").start_as_current_span("workflow in the parent process") as parent:
        context = parent.get_span_context()
        carrier = {"traceparent": f"00-{context.trace_id:032x}-{context.span_id:016x}-01"}
    outcome = {}

    def work():
        outcome["result"] = ProcessExecutor._run_node(
            node_data=ProcessExecutor.serialize_node(Echo(name="worker")),
            trace_carrier=carrier,
            input_data={"value": 1},
            config=traced,
            parent_run_id=uuid4(),
        )

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()

    assert outcome["result"].output == {"value": 1}
    assert parent_span_id(span_named(exporter, "worker")) == context.span_id
    assert root_names(exporter) == []


def test_concurrent_runs_have_one_root_each(exporter, provider):
    def run(index: int):
        config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))
        return two_step_workflow(f"wf-{index}").run(input_data={"value": index}, config=config)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(16)))

    assert all(result.status == RunnableStatus.SUCCESS for result in results)
    assert root_names(exporter) == sorted(f"invoke_workflow wf-{index}" for index in range(16))


def test_sampler_sees_the_marker_when_the_span_starts(exporter):
    # A backend, or a sampler in front of it, may keep or drop a trace by its top-most runs.
    seen = {}

    class Recording(sampling.Sampler):
        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, **kwargs):
            seen[name] = dict(attributes or {})
            return sampling.SamplingResult(sampling.Decision.RECORD_AND_SAMPLE, attributes)

        def get_description(self) -> str:
            return "Recording"

    provider = sdk_trace.TracerProvider(sampler=Recording())
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    two_step_workflow().run(input_data={"value": 1}, config=config)

    assert seen["invoke_workflow steps"][IS_ROOT] is True
    assert IS_ROOT not in seen["first"]
    assert IS_ROOT not in seen["second"]
