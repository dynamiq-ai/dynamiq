"""What the lifecycle hands to the mapping functions and where it puts what they return.

The mapping functions are replaced here, so that the tests hold whatever the real ones emit.
"""

from typing import Any

import pytest
from opentelemetry.trace import SpanKind

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import interop, semconv
from dynamiq.telemetry.types import CaptureSettings, SpanSpec, SpanType
from dynamiq.types.telemetry import ContentCaptureMode, TracingConfig
from tests.unit.telemetry.conftest import Calling, Echo, span_named

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

pytestmark = pytest.mark.usefixtures("no_span_leaks")


class Reporting(Node):
    """Reports data while it executes, as LLM nodes report usage and agents report hook events."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "reporting"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self.run_on_node_execute_run(
            config.callbacks, usage_data={"total_tokens": 3}, hook_events=[{"name": "guard"}], **kwargs
        )
        return {"reported": True}


class SwitchingProvider(Node):
    """Runs a node with a config that names another tracer provider."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "switching"
    other: Any = None

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        nested = RunnableConfig(tracing=TracingConfig(tracer_provider=self.other, session_id="nested"))
        return Echo(name="inner").run(input_data, config=nested, parent_run_id=kwargs["run_id"]).output


@pytest.fixture
def settings_on_spans(monkeypatch):
    """Makes every span show the run-level settings it was given."""

    def run_attributes(settings: CaptureSettings) -> dict[str, Any]:
        shown = {
            "test.session": settings.session_id,
            "test.user": settings.user_id,
            "test.source": settings.source_id,
            "test.tags": list(settings.tags),
            "test.content": settings.content_enabled,
        }
        return {key: value for key, value in shown.items() if value not in (None, [])}

    monkeypatch.setattr(semconv, "run_attributes", run_attributes)


@pytest.fixture
def conversation_from_input(monkeypatch):
    """Makes every run declare the session and user of its input, as agents do."""

    def conversation_of(span_type, entity, input_data):
        return input_data.get("session_id"), input_data.get("user_id")

    monkeypatch.setattr(semconv, "conversation_of", conversation_of)


def test_mapping_functions_get_the_node_and_its_data_as_they_are(exporter, traced, monkeypatch):
    seen = {}

    def input_attributes(span_type, entity, input_data, *, run_kwargs, settings, state):
        seen["input"] = (span_type, entity, input_data, run_kwargs["run_id"])
        state["from_input"] = True
        return {"test.input": "set"}

    def output_attributes(span_type, entity, output_data, *, run_kwargs, settings, state):
        seen["output"] = (span_type, entity, output_data, dict(state))
        return {"test.output": "set"}

    monkeypatch.setattr(semconv, "input_attributes", input_attributes)
    monkeypatch.setattr(semconv, "output_attributes", output_attributes)
    node = Echo()

    node.run(input_data={"value": 1}, config=traced)

    span = span_named(exporter, "echo")
    span_type, entity, input_data, run_id = seen["input"]
    assert entity is node
    assert input_data == {"value": 1}
    assert str(run_id) == span.attributes["dynamiq.run.id"]
    # The state one mapping call leaves is there for the later calls of the same span.
    assert seen["output"] == (span_type, node, {"value": 1}, {"from_input": True})
    assert span.attributes["test.input"] == "set"
    assert span.attributes["test.output"] == "set"


def test_workflow_mapping_gets_the_workflow_and_a_standalone_flow_gets_the_flow(exporter, traced, monkeypatch):
    entities = []

    def input_attributes(span_type, entity, input_data, *, run_kwargs, settings, state):
        if span_type == SpanType.WORKFLOW:
            entities.append(entity)
        return {}

    monkeypatch.setattr(semconv, "input_attributes", input_attributes)
    workflow = Workflow(flow=Flow(nodes=[Echo()]))
    flow = Flow(nodes=[Echo()])

    workflow.run(input_data={"value": 1}, config=traced)
    flow.run(input_data={"value": 1}, config=traced)

    assert len(entities) == 2
    assert entities[0] is workflow
    assert entities[1] is flow


def test_data_reported_during_execution_reaches_the_span(exporter, traced, monkeypatch):
    def execute_run_attributes(span_type, entity, *, run_kwargs, settings, state):
        return {"test.total_tokens": run_kwargs["usage_data"]["total_tokens"]}

    def hook_events(run_kwargs, *, settings):
        return [("test.hook", {"test.hook.name": event["name"]}) for event in run_kwargs["hook_events"]]

    monkeypatch.setattr(semconv, "execute_run_attributes", execute_run_attributes)
    monkeypatch.setattr(semconv, "hook_events", hook_events)

    Reporting().run(input_data={}, config=traced)

    span = span_named(exporter, "reporting")
    assert span.attributes["test.total_tokens"] == 3
    (event,) = (event for event in span.events if event.name == "test.hook")
    assert event.attributes["test.hook.name"] == "guard"


def test_settings_of_the_run_config_reach_every_span_of_the_tree(exporter, provider, settings_on_spans):
    tracing = TracingConfig(
        tracer_provider=provider,
        session_id="session-1",
        user_id="user-1",
        source_id="source-1",
        tags=["a", "b"],
        content_capture=ContentCaptureMode.NO_CONTENT,
    )
    workflow = Workflow(name="wf", flow=Flow(nodes=[Calling(id="outer", name="outer", inner=Echo(name="inner"))]))

    workflow.run(input_data={}, config=RunnableConfig(tracing=tracing))

    for name in ("invoke_workflow wf", "outer", "inner"):
        attributes = span_named(exporter, name).attributes
        assert attributes["test.session"] == "session-1"
        assert attributes["test.user"] == "user-1"
        assert attributes["test.source"] == "source-1"
        assert attributes["test.tags"] == ("a", "b")
        assert attributes["test.content"] is False


def test_session_a_run_declares_reaches_its_span_and_the_runs_below_it(
    exporter, traced, settings_on_spans, conversation_from_input
):
    outer = Calling(name="outer", inner=Echo(name="inner"))
    workflow = Workflow(name="wf", flow=Flow(nodes=[outer]))

    workflow.run(input_data={"session_id": "session-1", "user_id": "user-1"}, config=traced)

    for name in ("outer", "inner"):
        attributes = span_named(exporter, name).attributes
        assert attributes["test.session"] == "session-1"
        assert attributes["test.user"] == "user-1"
    # The workflow span started before the run that declares them.
    assert "test.session" not in span_named(exporter, "invoke_workflow wf").attributes


def test_session_of_the_run_config_wins_over_a_declared_one(
    exporter, provider, settings_on_spans, conversation_from_input
):
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, session_id="configured"))
    outer = Calling(name="outer", inner=Echo(name="inner"))

    outer.run(input_data={"session_id": "declared", "user_id": "user-1"}, config=config)

    for name in ("outer", "inner"):
        attributes = span_named(exporter, name).attributes
        assert attributes["test.session"] == "configured"
        assert attributes["test.user"] == "user-1"


def test_nearest_run_that_declares_a_session_names_it_for_the_runs_below(
    exporter, traced, settings_on_spans, conversation_from_input
):
    middle = Calling(name="middle", inner=Echo(name="inner"))
    outer = Calling(name="outer", inner=middle, inner_input={"session_id": "middle-session"})

    outer.run(input_data={"session_id": "outer-session", "user_id": "outer-user"}, config=traced)

    assert span_named(exporter, "outer").attributes["test.session"] == "outer-session"
    for name in ("middle", "inner"):
        attributes = span_named(exporter, name).attributes
        assert attributes["test.session"] == "middle-session"
        assert attributes["test.user"] == "outer-user"


def test_child_run_keeps_the_provider_and_settings_of_its_tree(exporter, traced, settings_on_spans):
    other_exporter = sdk_trace.export.in_memory_span_exporter.InMemorySpanExporter()
    other = sdk_trace.TracerProvider()
    other.add_span_processor(sdk_trace.export.SimpleSpanProcessor(other_exporter))

    result = SwitchingProvider(other=other).run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert other_exporter.get_finished_spans() == ()
    inner = span_named(exporter, "inner")
    assert inner.parent.span_id == span_named(exporter, "switching").context.span_id
    assert "test.session" not in inner.attributes


@pytest.fixture
def chat_spans(monkeypatch):
    """Makes every node span a chat span, as the spans of LLM nodes are."""

    def node_span_spec(node, *, run_id, run_kwargs):
        return SpanSpec(span_type=SpanType.CHAT, name=node.name, kind=SpanKind.CLIENT)

    monkeypatch.setattr(semconv, "node_span_spec", node_span_spec)


def test_overlapping_instrumentation_is_checked_for_at_every_recorded_chat_span(exporter, traced, mocker, chat_spans):
    check = mocker.spy(interop, "warn_if_litellm_otel_callback")

    Echo().run(input_data={"value": 1}, config=traced)
    Echo().run(input_data={"value": 1}, config=traced)

    # At every span, not the first only: the client library is imported by the first model call, after its span.
    assert check.call_count == 2
    assert all(span.kind == SpanKind.CLIENT for span in exporter.get_finished_spans())


def test_spans_that_are_not_chat_spans_are_not_checked(exporter, traced, mocker):
    check = mocker.spy(interop, "warn_if_litellm_otel_callback")

    Echo().run(input_data={"value": 1}, config=traced)

    assert check.call_count == 0


def test_check_for_overlapping_instrumentation_that_raises_does_not_fail_the_run(
    exporter, traced, monkeypatch, chat_spans
):
    def broken():
        raise RuntimeError("check bug")

    monkeypatch.setattr(interop, "warn_if_litellm_otel_callback", broken)

    result = Echo().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert span_named(exporter, "echo").attributes["dynamiq.run.status"] == "succeeded"
