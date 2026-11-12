"""Pause and resume: the checkpoint carries the paused span, and the resumed run links to it."""

import pytest

from dynamiq import Workflow
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import semconv
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import Echo, Pausing, span_named

pytest.importorskip("opentelemetry.sdk")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

FLOW_RUN_ID = "dynamiq.flow.run_id"


def traceparent_of(span) -> str:
    context = span.context
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"


def gated_flow(backend, name: str = "gated") -> Flow:
    """A step, then a gate that pauses the run until the test opens it."""
    before = Echo(id="before", name="before")
    gate = Pausing(id="gate", name="gate")
    gate.depends_on(before)
    return Flow(name=name, nodes=[before, gate], checkpoint=CheckpointConfig(enabled=True, backend=backend))


def open_gate(flow: Flow) -> None:
    flow.nodes[1]._pause = False


@pytest.fixture
def resume_on_spans(monkeypatch):
    """Makes a resumed span show what it resumed from."""

    def resume_attributes(run_kwargs):
        return {
            "test.resumed_from": run_kwargs["resumed_from_checkpoint"],
            "test.original_run_id": run_kwargs["original_run_id"],
        }

    monkeypatch.setattr(semconv, "resume_attributes", resume_attributes)


def test_resumed_workflow_links_to_the_workflow_span_of_the_paused_run(exporter, provider, resume_on_spans):
    backend = InMemory()
    flow = gated_flow(backend)
    workflow = Workflow(name="wf", flow=flow)

    paused = workflow.run(
        input_data={"value": 1}, config=RunnableConfig(tracing=TracingConfig(tracer_provider=provider))
    )

    assert paused.error.type is RunPausedException
    paused_span = span_named(exporter, "invoke_workflow wf")
    checkpoint = backend.get_latest_by_flow(flow.id)
    assert checkpoint.metadata["traceparent"] == traceparent_of(paused_span)

    exporter.clear()
    open_gate(flow)
    # The tracing settings are not in the checkpoint, so the resuming caller passes them again.
    resumed = workflow.run(
        input_data=None,
        config=RunnableConfig(tracing=TracingConfig(tracer_provider=provider)),
        resume_from=checkpoint.id,
    )

    assert resumed.status == RunnableStatus.SUCCESS
    # A resumed run is a new run with new spans; the step that completed before the pause has none.
    assert sorted(span.name for span in exporter.get_finished_spans()) == ["gate", "invoke_workflow wf"]
    resumed_span = span_named(exporter, "invoke_workflow wf")
    assert resumed_span.context.trace_id != paused_span.context.trace_id
    (link,) = resumed_span.links
    assert link.context.trace_id == paused_span.context.trace_id
    assert link.context.span_id == paused_span.context.span_id
    assert dict(link.attributes) == semconv.RESUME_LINK_ATTRIBUTES
    assert resumed_span.attributes["test.resumed_from"] == checkpoint.id
    assert resumed_span.attributes["test.original_run_id"] == paused_span.attributes[FLOW_RUN_ID]
    # The checkpoint the run resumed from still names the paused span, for a second resume from it.
    assert backend.load(checkpoint.id).metadata["traceparent"] == traceparent_of(paused_span)
    assert backend.get_latest_by_flow(flow.id).metadata["traceparent"] == traceparent_of(resumed_span)


def test_resumed_standalone_flow_links_to_the_flow_span_of_the_paused_run(exporter, traced, resume_on_spans):
    backend = InMemory()
    flow = gated_flow(backend, name="alone")

    flow.run(input_data={"value": 1}, config=traced)
    paused_span = span_named(exporter, "invoke_workflow alone")
    checkpoint = backend.get_latest_by_flow(flow.id)

    exporter.clear()
    open_gate(flow)
    resumed = flow.run(input_data=None, config=traced, resume_from=checkpoint.id)

    assert resumed.status == RunnableStatus.SUCCESS
    resumed_span = span_named(exporter, "invoke_workflow alone")
    (link,) = resumed_span.links
    assert link.context.span_id == paused_span.context.span_id
    assert resumed_span.attributes["test.resumed_from"] == checkpoint.id


@pytest.mark.asyncio
async def test_resumed_async_workflow_links_to_the_paused_run(exporter, traced):
    backend = InMemory()
    flow = gated_flow(backend)
    workflow = Workflow(name="wf", flow=flow)

    await workflow.run_async(input_data={"value": 1}, config=traced)
    paused_span = span_named(exporter, "invoke_workflow wf")
    checkpoint = backend.get_latest_by_flow(flow.id)

    exporter.clear()
    open_gate(flow)
    resumed = await workflow.run_async(input_data=None, config=traced, resume_from=checkpoint.id)

    assert resumed.status == RunnableStatus.SUCCESS
    (link,) = span_named(exporter, "invoke_workflow wf").links
    assert link.context.span_id == paused_span.context.span_id


@pytest.mark.parametrize("stored", [None, "not-a-traceparent", "00-" + "0" * 32 + "-" + "0" * 16 + "-01", 7])
def test_resume_from_a_checkpoint_without_a_usable_traceparent_has_no_link(exporter, traced, stored):
    backend = InMemory()
    flow = gated_flow(backend)
    workflow = Workflow(name="wf", flow=flow)
    # Paused without tracing, as a run of an earlier release or of a process without a provider is.
    workflow.run(input_data={"value": 1})
    checkpoint = backend.get_latest_by_flow(flow.id)
    assert "traceparent" not in checkpoint.metadata
    if stored is not None:
        checkpoint.metadata["traceparent"] = stored
        backend.update(checkpoint)

    open_gate(flow)
    resumed = workflow.run(input_data=None, config=traced, resume_from=checkpoint.id)

    assert resumed.status == RunnableStatus.SUCCESS
    assert span_named(exporter, "invoke_workflow wf").links == ()


def test_tracing_settings_are_not_saved_in_a_checkpoint(provider):
    backend = InMemory()
    flow = gated_flow(backend)
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, session_id="session-1"))

    Workflow(flow=flow).run(input_data={"value": 1}, config=config)

    assert "tracing" not in config.to_checkpoint_dict()
    assert "tracing" not in backend.get_latest_by_flow(flow.id).original_config
