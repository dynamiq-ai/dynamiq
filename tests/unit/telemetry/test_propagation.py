"""Trace context that leaves the process: to worker processes, into checkpoints and to MCP servers."""

import pickle
import threading
from concurrent import futures
from typing import Any
from uuid import uuid4

import pytest
from opentelemetry import baggage
from opentelemetry import context as context_api
from opentelemetry import trace
from pydantic import PrivateAttr

from dynamiq import Workflow
from dynamiq.executors.pool import ProcessExecutor
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import propagation
from dynamiq.types.cancellation import CancellationToken
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import Echo, parent_span_id, span_named

pytest.importorskip("opentelemetry.sdk")

pytestmark = pytest.mark.usefixtures("no_span_leaks")


def traceparent_of(span) -> str:
    context = span.get_span_context()
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"


class Capturing(Node):
    """Keeps the carrier and the traceparent of the run that started it."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "capturing"
    _carrier: Any = PrivateAttr(default=None)
    _traceparent: Any = PrivateAttr(default=None)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._carrier = propagation.carrier_for_run(kwargs["parent_run_id"])
        self._traceparent = propagation.traceparent_for_run(kwargs["parent_run_id"])
        return {}


def run_in_new_thread(function, **kwargs):
    """Calls ``function`` on a thread of its own, which like a worker process starts with an empty context."""
    outcome = {}

    def target():
        outcome["result"] = function(**kwargs)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    return outcome["result"]


def test_carrier_of_a_flow_run_names_the_workflow_span(exporter, traced):
    node = Capturing(id="capturing")

    Workflow(name="wf", flow=Flow(nodes=[node])).run(input_data={}, config=traced)

    # The node's parent run is the flow run, which the workflow span stands for.
    workflow_span = span_named(exporter, "invoke_workflow wf")
    assert node._carrier == {"traceparent": traceparent_of(workflow_span)}
    assert node._traceparent == traceparent_of(workflow_span)


def test_untraced_run_has_no_carrier():
    node = Capturing(id="capturing")

    Workflow(flow=Flow(nodes=[node])).run(input_data={})

    assert node._carrier is None
    assert node._traceparent is None
    assert propagation.carrier_for_run(uuid4()) is None
    assert propagation.carrier_for_run(None) is None


def test_attached_carrier_makes_the_span_of_the_carrier_current(provider):
    with provider.get_tracer("application").start_as_current_span("parent") as parent:
        carrier = {"traceparent": traceparent_of(parent)}
    context_before = context_api.get_current()

    with propagation.attached_carrier(carrier):
        current = trace.get_current_span().get_span_context()
        assert (current.trace_id, current.span_id) == (
            parent.get_span_context().trace_id,
            parent.get_span_context().span_id,
        )
        assert current.is_remote

    assert context_api.get_current() is context_before


@pytest.mark.parametrize("carrier", [None, {}, {"traceparent": "garbage"}])
def test_attached_carrier_without_a_trace_context_changes_nothing(carrier):
    with propagation.attached_carrier(carrier):
        assert not trace.get_current_span().get_span_context().is_valid


@pytest.fixture
def submitted_to_pool(monkeypatch) -> list[dict]:
    """Replaces the process pool with one that runs a submitted call on a thread, which like a worker process
    starts with an empty context, and records what was submitted."""
    submitted = []

    class InlinePool:
        def __init__(self, max_workers=None):
            self.pool = futures.ThreadPoolExecutor(max_workers=1)

        def submit(self, function, **kwargs):
            submitted.append(kwargs)
            return self.pool.submit(function, **kwargs)

        def shutdown(self, wait=True):
            self.pool.shutdown(wait=wait)

    monkeypatch.setattr(futures, "ProcessPoolExecutor", InlinePool)
    return submitted


def test_process_executor_hands_the_trace_context_to_its_workers(exporter, traced, submitted_to_pool):
    workflow = Workflow(name="wf", flow=Flow(nodes=[Echo(id="worker", name="worker")], executor=ProcessExecutor))

    result = workflow.run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    workflow_span = span_named(exporter, "invoke_workflow wf")
    (call,) = submitted_to_pool
    assert call["trace_carrier"] == {"traceparent": traceparent_of(workflow_span)}
    assert parent_span_id(span_named(exporter, "worker")) == workflow_span.context.span_id


def test_process_executor_hands_no_trace_context_from_an_untraced_run(submitted_to_pool):
    result = Workflow(flow=Flow(nodes=[Echo(id="worker")], executor=ProcessExecutor)).run(input_data={"value": 1})

    assert result.status == RunnableStatus.SUCCESS
    (call,) = submitted_to_pool
    assert call["trace_carrier"] is None


def test_node_run_by_a_worker_continues_the_trace_of_its_carrier(exporter, provider, traced):
    with provider.get_tracer("application").start_as_current_span("workflow in the parent process") as parent:
        carrier = {"traceparent": traceparent_of(parent)}

    # The worker has neither the context nor the registry of the parent process: the parent run is unknown here.
    result = run_in_new_thread(
        ProcessExecutor._run_node,
        node_data=ProcessExecutor.serialize_node(Echo(name="worker")),
        trace_carrier=carrier,
        input_data={"value": 1},
        config=traced,
        parent_run_id=uuid4(),
    )

    assert result.output == {"value": 1}
    worker_span = span_named(exporter, "worker")
    assert parent_span_id(worker_span) == parent.get_span_context().span_id
    assert worker_span.context.trace_id == parent.get_span_context().trace_id


def test_worker_runs_a_node_without_a_carrier():
    result = run_in_new_thread(
        ProcessExecutor._run_node,
        node_data=ProcessExecutor.serialize_node(Echo()),
        input_data={"value": 1},
    )

    assert result.output == {"value": 1}


def test_tracing_config_is_pickled_without_its_provider(provider):
    tracing = TracingConfig(tracer_provider=provider, session_id="session-1", tags=["a"])

    copy = pickle.loads(pickle.dumps(tracing))  # nosec B301

    assert copy.tracer_provider is None
    assert copy.session_id == "session-1"
    assert copy.tags == ["a"]
    assert tracing.tracer_provider is provider


def test_provider_does_not_keep_a_run_config_from_being_pickled(provider, monkeypatch):
    # The cancellation token of a run config holds a lock and is not picklable today, which is why this test
    # makes it so: once the token is, the provider must not be what keeps the config out of a worker process.
    monkeypatch.setattr(CancellationToken, "__reduce__", lambda self: (CancellationToken, ()), raising=False)
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, session_id="session-1"))

    copy = pickle.loads(pickle.dumps(config))  # nosec B301

    assert copy.tracing.tracer_provider is None
    assert copy.tracing.session_id == "session-1"


def test_traceparent_parses_back_to_the_span_it_was_built_from(provider):
    with provider.get_tracer("application").start_as_current_span("parent") as parent:
        span_context = propagation.span_context_from_traceparent(traceparent_of(parent))

    assert span_context.trace_id == parent.get_span_context().trace_id
    assert span_context.span_id == parent.get_span_context().span_id


@pytest.mark.parametrize("value", [None, "", "garbage", 7, "00-" + "0" * 32 + "-" + "0" * 16 + "-01", "00-abc-def-01"])
def test_value_that_is_not_a_traceparent_parses_to_none(provider, value):
    # Inside a span, to show that a bad value does not come back as the span that is current.
    with provider.get_tracer("application").start_as_current_span("current"):
        assert propagation.span_context_from_traceparent(value) is None


def test_mcp_request_meta_carries_the_current_trace_context_and_baggage(provider):
    assert propagation.mcp_request_meta() is None

    with provider.get_tracer("application").start_as_current_span("tool call") as span:
        assert propagation.mcp_request_meta() == {"traceparent": traceparent_of(span)}
        token = context_api.attach(baggage.set_baggage("tenant", "acme"))
        try:
            assert propagation.mcp_request_meta() == {"traceparent": traceparent_of(span), "baggage": "tenant=acme"}
        finally:
            context_api.detach(token)
