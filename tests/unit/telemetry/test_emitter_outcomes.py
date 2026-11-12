"""How the outcome of a run ends its span, and that no way out of a run leaves a span open, registered or current.

Every test runs under ``no_span_leaks``, which checks the registry and the current context after the run.
"""

import asyncio
import threading
import time
from typing import Any, ClassVar

import pytest
from opentelemetry import context as context_api
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, PrivateAttr

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ErrorHandling
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import registry, semconv
from dynamiq.telemetry.types import ErrorSpec
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import (
    AsyncWaiting,
    Echo,
    Failing,
    Flaky,
    Pausing,
    Waiting,
    parent_span_id,
    span_named,
    status_of,
)

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

ERROR_TYPE = "error.type"
CHECKPOINT_ID = "dynamiq.checkpoint.id"
ATTEMPT = "dynamiq.execution.attempt"


class StrictInput(BaseModel):
    value: int


class Strict(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "strict"
    input_schema: ClassVar[type[StrictInput]] = StrictInput

    def execute(self, input_data: StrictInput, config: RunnableConfig = None, **kwargs):
        return {"value": input_data.value}


class Reporting(Node):
    """Reports data while it executes, as LLM nodes report usage."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "reporting"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self.run_on_node_execute_run(config.callbacks, usage_data={"total_tokens": 3}, **kwargs)
        return {"reported": True}


class LateCalling(Node):
    """Outlives its timeout, then runs another node from the thread that was left behind."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "late-calling"
    inner: Node
    delay_seconds: float = 0.5
    _finished: threading.Event = PrivateAttr(default_factory=threading.Event)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        try:
            time.sleep(self.delay_seconds)
            return self.inner.run(input_data={"value": 1}, config=config, parent_run_id=kwargs["run_id"]).output
        finally:
            self._finished.set()


class Abort(BaseException):
    """An exception run code does not turn into a result, as KeyboardInterrupt and SystemExit are not."""


class Aborting(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "aborting"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        raise Abort("stop everything")

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        raise Abort("stop everything")


class RaisingOnStart(BaseCallbackHandler):
    """A user callback that fails where flow and workflow callbacks are not guarded."""

    def __init__(self, hook: str):
        self.hook = hook

    def on_workflow_start(self, serialized, input_data, **kwargs):
        if self.hook == "workflow":
            raise RuntimeError("callback failed")

    def on_flow_start(self, serialized, input_data, **kwargs):
        if self.hook == "flow":
            raise RuntimeError("callback failed")


async def until_registered(count: int) -> None:
    """Waits until the run under test has started ``count`` spans, however busy the machine is."""
    deadline = time.monotonic() + 10
    while registry.size() != count:
        assert time.monotonic() < deadline, f"{registry.size()} spans registered, expected {count}"
        await asyncio.sleep(0.005)


def cancel_once_registered(config: RunnableConfig, count: int) -> threading.Thread:
    """Cancels the run of ``config`` from another thread once it has started ``count`` spans."""

    def wait_and_cancel():
        deadline = time.monotonic() + 10
        while registry.size() != count and time.monotonic() < deadline:
            time.sleep(0.005)
        config.cancellation.token.cancel()

    thread = threading.Thread(target=wait_and_cancel)
    thread.start()
    return thread


def execution_attempts(span) -> list[int]:
    return [event.attributes[ATTEMPT] for event in span.events if event.name == semconv.EXECUTION_EVENT_NAME]


def assert_failed(span, error_type: str) -> None:
    assert status_of(span) == "failed"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == error_type


def assert_canceled(span) -> None:
    assert status_of(span) == "canceled"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "CanceledException"


def test_succeeded_node_span_has_unset_status_and_no_events(exporter, traced):
    Echo().run(input_data={"value": 1}, config=traced)

    span = span_named(exporter, "echo")
    assert status_of(span) == "succeeded"
    assert span.status.status_code == StatusCode.UNSET
    assert ERROR_TYPE not in span.attributes
    # The span is the attempt of a run that succeeds at once.
    assert execution_attempts(span) == []


def test_failed_node_span_has_error_status_and_type(exporter, traced):
    result = Failing().run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert_failed(span_named(exporter, "failing"), "ValueError")


def test_failure_is_recorded_as_the_mapping_describes_it(exporter, traced, monkeypatch):
    monkeypatch.setattr(
        semconv,
        "error_spec",
        lambda error, outcome, *, settings: ErrorSpec(
            error_type="Described", description=str(error), event_attributes={"exception.type": "Described"}
        ),
    )

    Failing().run(input_data={}, config=traced)

    span = span_named(exporter, "failing")
    assert span.attributes[ERROR_TYPE] == "Described"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"
    (event,) = (event for event in span.events if event.name == semconv.EXCEPTION_EVENT_NAME)
    assert event.attributes["exception.type"] == "Described"


def test_input_of_a_run_that_fails_before_it_starts_is_recorded(exporter, traced, monkeypatch):
    recorded = []

    def input_attributes(span_type, entity, input_data, *, run_kwargs, settings, state):
        recorded.append(input_data)
        return {"test.input": "recorded"}

    monkeypatch.setattr(semconv, "input_attributes", input_attributes)

    result = Strict().run(input_data={"value": "not a number"}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    span = span_named(exporter, "strict")
    assert status_of(span) == "failed"
    assert span.attributes["test.input"] == "recorded"
    assert recorded == [{"value": "not a number"}]


def test_retried_node_is_one_span_with_an_event_per_attempt(exporter, traced):
    node = Flaky(fail_times=1, error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01))

    result = node.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    (span,) = exporter.get_finished_spans()
    assert status_of(span) == "succeeded"
    assert span.status.status_code == StatusCode.UNSET
    assert execution_attempts(span) == [1, 2]


@pytest.mark.asyncio
async def test_retried_async_node_is_one_span_with_an_event_per_attempt(exporter, traced):
    node = Flaky(fail_times=2, error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0.01))
    context_before = context_api.get_current()

    result = await node.run_async(input_data={}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    (span,) = exporter.get_finished_spans()
    assert status_of(span) == "succeeded"
    assert execution_attempts(span) == [1, 2, 3]
    # The fixture cannot see the context of the task the test runs in.
    assert context_api.get_current() is context_before


def test_node_out_of_retries_is_one_failed_span(exporter, traced):
    node = Flaky(fail_times=5, error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0.01))

    result = node.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    (span,) = exporter.get_finished_spans()
    assert_failed(span, "ValueError")
    assert execution_attempts(span) == [1, 2]


def test_node_behind_a_failed_node_is_skipped(exporter, traced):
    first = Failing(id="first", name="first")
    second = Echo(id="second", name="second")
    second.depends_on(first)
    workflow = Workflow(name="wf", flow=Flow(nodes=[first, second]))

    result = workflow.run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    skipped = span_named(exporter, "second")
    assert status_of(skipped) == "skipped"
    assert skipped.status.status_code == StatusCode.UNSET
    assert ERROR_TYPE not in skipped.attributes
    assert_failed(span_named(exporter, "first"), "ValueError")
    assert_failed(span_named(exporter, "invoke_workflow wf"), "FlowNodeFailureException")


def test_canceled_run_ends_its_spans_as_canceled(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Waiting(id="waiting")]))
    # The workflow span, the flow run that shares it and the node span: the node is running.
    canceling = cancel_once_registered(traced, 3)

    result = workflow.run(input_data={}, config=traced)
    canceling.join()

    assert result.status == RunnableStatus.CANCELED
    assert_canceled(span_named(exporter, "waiting"))
    assert_canceled(span_named(exporter, "invoke_workflow wf"))


@pytest.mark.asyncio
async def test_task_cancellation_ends_the_node_span_as_canceled(exporter, traced):
    context_before = context_api.get_current()
    task = asyncio.create_task(AsyncWaiting().run_async(input_data={}, config=traced))
    await until_registered(1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # No hook reports this outcome: the cancellation passes through the node's run as an exception.
    assert_canceled(span_named(exporter, "async-waiting"))
    assert registry.size() == 0
    assert context_api.get_current() is context_before


@pytest.mark.asyncio
async def test_task_cancellation_of_a_node_running_in_a_thread_ends_its_span_as_canceled(exporter, traced):
    # A node without a native async execute runs in a thread, which the canceled task signals and waits for.
    task = asyncio.create_task(Waiting().run_async(input_data={}, config=traced))
    await until_registered(1)

    task.cancel()
    result = await task

    assert result.status == RunnableStatus.CANCELED
    assert_canceled(span_named(exporter, "waiting"))
    assert registry.size() == 0


@pytest.mark.asyncio
async def test_task_cancellation_of_an_async_workflow_ends_every_span(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[AsyncWaiting(id="waiting")]))
    task = asyncio.create_task(workflow.run_async(input_data={}, config=traced))
    await until_registered(3)

    task.cancel()
    result = await task

    assert result.status == RunnableStatus.CANCELED
    assert_canceled(span_named(exporter, "async-waiting"))
    assert_canceled(span_named(exporter, "invoke_workflow wf"))
    assert registry.size() == 0


def test_canceled_standalone_flow_ends_its_spans_as_canceled(exporter, traced):
    flow = Flow(name="alone", nodes=[Waiting(id="waiting")])
    # The flow span and the node span: the node is running.
    canceling = cancel_once_registered(traced, 2)

    result = flow.run(input_data={}, config=traced)
    canceling.join()

    assert result.status == RunnableStatus.CANCELED
    assert_canceled(span_named(exporter, "waiting"))
    assert_canceled(span_named(exporter, "invoke_workflow alone"))


@pytest.mark.asyncio
async def test_task_cancellation_of_a_standalone_async_flow_ends_every_span(exporter, traced):
    flow = Flow(name="alone", nodes=[AsyncWaiting(id="waiting")])
    task = asyncio.create_task(flow.run_async(input_data={}, config=traced))
    await until_registered(2)

    task.cancel()
    result = await task

    assert result.status == RunnableStatus.CANCELED
    assert_canceled(span_named(exporter, "async-waiting"))
    assert_canceled(span_named(exporter, "invoke_workflow alone"))
    assert registry.size() == 0


def test_timed_out_node_span_ends_as_failed(exporter, traced):
    node = Waiting(seconds=0.5, error_handling=ErrorHandling(timeout_seconds=0.05))

    result = node.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert_failed(span_named(exporter, "waiting"), "TimeoutError")
    assert registry.size() == 0


@pytest.mark.asyncio
async def test_timed_out_async_node_span_ends_as_failed(exporter, traced):
    node = AsyncWaiting(error_handling=ErrorHandling(timeout_seconds=0.05))
    context_before = context_api.get_current()

    result = await node.run_async(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert_failed(span_named(exporter, "async-waiting"), "TimeoutError")
    assert registry.size() == 0
    assert context_api.get_current() is context_before


def test_run_started_by_a_timed_out_node_is_a_child_of_its_ended_span(exporter, traced):
    node = LateCalling(inner=Echo(name="late"), error_handling=ErrorHandling(timeout_seconds=0.05))

    result = node.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    timed_out = span_named(exporter, "late-calling")
    assert_failed(timed_out, "TimeoutError")
    assert node._finished.wait(timeout=5)
    late = span_named(exporter, "late")
    assert parent_span_id(late) == timed_out.context.span_id
    assert status_of(late) == "succeeded"


def test_paused_run_ends_its_spans_as_paused(exporter, traced):
    backend = InMemory()
    flow = Flow(nodes=[Pausing(id="gate", name="gate")], checkpoint=CheckpointConfig(enabled=True, backend=backend))
    workflow = Workflow(name="wf", flow=flow)

    result = workflow.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert result.error.type is RunPausedException
    workflow_span = span_named(exporter, "invoke_workflow wf")
    for span in (span_named(exporter, "gate"), workflow_span):
        assert status_of(span) == "paused"
        assert span.status.status_code == StatusCode.UNSET
        assert ERROR_TYPE not in span.attributes
    assert workflow_span.attributes[CHECKPOINT_ID] == backend.get_latest_by_flow(flow.id).id


def test_paused_standalone_flow_names_its_checkpoint(exporter, traced):
    backend = InMemory()
    flow = Flow(
        name="alone",
        nodes=[Pausing(id="gate", name="gate")],
        checkpoint=CheckpointConfig(enabled=True, backend=backend),
    )

    flow.run(input_data={}, config=traced)

    flow_span = span_named(exporter, "invoke_workflow alone")
    assert status_of(flow_span) == "paused"
    assert flow_span.attributes[CHECKPOINT_ID] == backend.get_latest_by_flow(flow.id).id


def test_workflow_whose_flow_raises_before_it_starts_ends_its_span_as_failed(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Echo()]))

    with pytest.raises(ValueError, match="Checkpoint not found"):
        workflow.run(input_data={}, config=traced, resume_from="missing")

    (span,) = exporter.get_finished_spans()
    assert span.name == "invoke_workflow wf"
    assert_failed(span, "ValueError")


@pytest.mark.asyncio
async def test_async_workflow_whose_flow_raises_before_it_starts_ends_its_span_as_failed(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Echo()]))

    with pytest.raises(ValueError, match="Checkpoint not found"):
        await workflow.run_async(input_data={}, config=traced, resume_from="missing")

    (span,) = exporter.get_finished_spans()
    assert_failed(span, "ValueError")
    assert registry.size() == 0


def test_standalone_flow_that_raises_before_it_starts_has_no_span(exporter, traced):
    with pytest.raises(ValueError, match="Checkpoint not found"):
        Flow(nodes=[Echo()]).run(input_data={}, config=traced, resume_from="missing")

    assert exporter.get_finished_spans() == ()


def test_exception_that_run_code_does_not_handle_ends_every_span_as_failed(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Aborting(id="aborting")]))

    with pytest.raises(Abort):
        workflow.run(input_data={}, config=traced)

    # No node, flow or workflow hook reports this outcome: each span is ended on the way out.
    assert_failed(span_named(exporter, "aborting"), "Abort")
    assert_failed(span_named(exporter, "invoke_workflow wf"), "Abort")


def test_exception_that_run_code_does_not_handle_ends_a_standalone_flow_span(exporter, traced):
    with pytest.raises(Abort):
        Flow(name="alone", nodes=[Aborting(id="aborting")]).run(input_data={}, config=traced)

    assert_failed(span_named(exporter, "aborting"), "Abort")
    assert_failed(span_named(exporter, "invoke_workflow alone"), "Abort")


@pytest.mark.asyncio
async def test_exception_that_async_run_code_does_not_handle_ends_every_span(exporter, traced):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Aborting(id="aborting")]))

    with pytest.raises(Abort):
        await workflow.run_async(input_data={}, config=traced)

    assert_failed(span_named(exporter, "aborting"), "Abort")
    assert_failed(span_named(exporter, "invoke_workflow wf"), "Abort")
    assert registry.size() == 0


@pytest.mark.asyncio
async def test_exception_that_async_run_code_does_not_handle_ends_a_standalone_flow_span(exporter, traced):
    with pytest.raises(Abort):
        await Flow(name="alone", nodes=[Aborting(id="aborting")]).run_async(input_data={}, config=traced)

    assert_failed(span_named(exporter, "aborting"), "Abort")
    assert_failed(span_named(exporter, "invoke_workflow alone"), "Abort")
    assert registry.size() == 0


@pytest.mark.parametrize("hook", ["workflow", "flow"])
def test_start_callback_that_raises_ends_the_workflow_span_as_failed(exporter, provider, hook):
    config = RunnableConfig(callbacks=[RaisingOnStart(hook)], tracing=TracingConfig(tracer_provider=provider))
    workflow = Workflow(name="wf", flow=Flow(nodes=[Echo()]))

    with pytest.raises(RuntimeError, match="callback failed"):
        workflow.run(input_data={}, config=config)

    (span,) = exporter.get_finished_spans()
    assert span.name == "invoke_workflow wf"
    assert_failed(span, "RuntimeError")


def test_start_callback_that_raises_ends_the_flow_span_as_failed(exporter, provider):
    config = RunnableConfig(callbacks=[RaisingOnStart("flow")], tracing=TracingConfig(tracer_provider=provider))

    with pytest.raises(RuntimeError, match="callback failed"):
        Flow(name="alone", nodes=[Echo()]).run(input_data={}, config=config)

    (span,) = exporter.get_finished_spans()
    assert span.name == "invoke_workflow alone"
    assert_failed(span, "RuntimeError")


def reporting_workflow() -> Workflow:
    first = Flaky(
        id="first", name="first", fail_times=1, error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0.01)
    )
    second = Reporting(id="second", name="second")
    second.depends_on(first)
    return Workflow(name="wf", flow=Flow(nodes=[first, second]))


@pytest.mark.parametrize(
    "mapping",
    [
        "run_attributes",
        "conversation_of",
        "input_attributes",
        "execute_run_attributes",
        "hook_events",
        "execute_output_attributes",
        "execution_event",
        "output_attributes",
        "outcome_attributes",
    ],
)
def test_mapping_that_raises_costs_neither_the_run_nor_the_span(exporter, traced, monkeypatch, mapping):
    def broken(*args, **kwargs):
        raise RuntimeError("mapping bug")

    monkeypatch.setattr(semconv, mapping, broken)

    result = reporting_workflow().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["second"]["output"] == {"reported": True}
    assert sorted(span.name for span in exporter.get_finished_spans()) == ["first", "invoke_workflow wf", "second"]


@pytest.mark.parametrize("mapping", ["error_spec", "skip_attributes", "failure_attributes", "outcome_attributes"])
def test_mapping_that_raises_on_a_failed_run_still_ends_every_span(exporter, traced, monkeypatch, mapping):
    def broken(*args, **kwargs):
        raise RuntimeError("mapping bug")

    monkeypatch.setattr(semconv, mapping, broken)
    first = Failing(id="first", name="first")
    second = Echo(id="second", name="second")
    second.depends_on(first)

    result = Workflow(name="wf", flow=Flow(nodes=[first, second])).run(input_data={}, config=traced)

    assert result.status == RunnableStatus.FAILURE
    assert sorted(span.name for span in exporter.get_finished_spans()) == ["first", "invoke_workflow wf", "second"]
    if mapping == "error_spec":
        # Without the mapping's description of the error, the status still says that the run failed.
        assert span_named(exporter, "first").status.status_code == StatusCode.ERROR


def test_span_spec_that_raises_leaves_the_run_untraced_and_unharmed(exporter, traced, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("mapping bug")

    monkeypatch.setattr(semconv, "node_span_spec", broken)

    result = reporting_workflow().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert [span.name for span in exporter.get_finished_spans()] == ["invoke_workflow wf"]


class RaisingProcessor(sdk_trace.SpanProcessor):
    """A span processor with a bug, which the SDK lets through to the code that starts or ends a span."""

    def __init__(self, hook: str):
        self.hook = hook

    def on_start(self, span, parent_context=None):
        if self.hook == "on_start":
            raise RuntimeError("processor bug")

    def on_end(self, span):
        if self.hook == "on_end":
            raise RuntimeError("processor bug")


@pytest.mark.parametrize("hook", ["on_start", "on_end"])
def test_span_processor_that_raises_does_not_fail_the_run(hook):
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(RaisingProcessor(hook))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    result = reporting_workflow().run(input_data={"value": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["second"]["output"] == {"reported": True}
