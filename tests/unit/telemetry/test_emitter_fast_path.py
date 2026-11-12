"""What a run costs and records when it is not traced, sampled out or switched off."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from opentelemetry import trace

import dynamiq
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import registry, semconv, settings
from dynamiq.types.telemetry import TracingConfig
from dynamiq.utils import utils
from tests.unit.telemetry.conftest import Echo, Parent, Prompting, parent_span_id, span_named

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
sampling = pytest.importorskip("opentelemetry.sdk.trace.sampling")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

REPO_ROOT = Path(dynamiq.__file__).parent.parent
# The functions the lifecycle calls the mapping module through.
MAPPING_FUNCTIONS = (
    "scope_attributes",
    "workflow_span_spec",
    "node_span_spec",
    "run_attributes",
    "conversation_of",
    "input_attributes",
    "prompt_attributes",
    "execute_run_attributes",
    "hook_events",
    "execute_output_attributes",
    "output_attributes",
    "outcome_attributes",
    "skip_attributes",
    "failure_attributes",
    "resume_attributes",
    "error_spec",
    "execution_event",
)
# Built before the sampler decides, and with the tracer of a provider: the only mapping a sampled-out run costs.
SPAN_SPECS = {"node_span_spec", "workflow_span_spec"}
SCOPE = {"scope_attributes"}


def two_step_workflow() -> Workflow:
    first = Echo(id="first", name="first")
    second = Echo(id="second", name="second")
    second.depends_on(first)
    return Workflow(name="steps", flow=Flow(nodes=[first, second]))


@pytest.fixture
def mapping_calls(mocker) -> dict:
    """Spies on the functions of the mapping module, by name."""
    return {name: mocker.spy(semconv, name) for name in MAPPING_FUNCTIONS}


@pytest.fixture
def format_value_calls(mocker) -> list:
    """Records calls of format_value through every dynamiq module that imported it."""
    calls = []
    original = utils.format_value

    def recording(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("dynamiq") and getattr(module, "format_value", None) is original:
            mocker.patch.object(module, "format_value", recording)
    return calls


@pytest.fixture
def no_global_provider():
    # The tests of an untraced run rely on what a process has until an application installs an SDK provider.
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


def test_untraced_node_run_serializes_nothing(mocker, mapping_calls, format_value_calls, no_global_provider):
    to_dict = mocker.spy(Node, "to_dict")

    result = Echo().run(input_data={"value": 1})

    assert result.status == RunnableStatus.SUCCESS
    assert result.input == {"value": 1}
    assert result.output == {"value": 1}
    assert {name for name, spy in mapping_calls.items() if spy.call_count} == set()
    assert to_dict.call_count == 0
    assert format_value_calls == []
    assert registry.size() == 0


def test_untraced_workflow_run_calls_no_mapping(mapping_calls, no_global_provider):
    result = two_step_workflow().run(input_data={"value": 1})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["second"]["output"] == {"value": 1}
    assert {name for name, spy in mapping_calls.items() if spy.call_count} == set()
    assert registry.size() == 0


def run_reporting_prompts(config: RunnableConfig | None):
    """A run whose LLM node runs twice below it, the second prompt continuing the first: what prompt compaction
    acts on."""
    system, user = {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}
    prompts = [[system, user], [system, user, {"role": "assistant", "content": "Hello"}]]

    def script(config, run_id):
        for prompt in prompts:
            result = Prompting(id="llm").run(input_data={"messages": prompt}, config=config, parent_run_id=run_id)
            assert result.status == RunnableStatus.SUCCESS

    parent = Parent()
    parent._script = script
    return parent.run(input_data={}, config=config)


@pytest.mark.parametrize("on_the_config", [False, True], ids=["environment", "config"])
def test_untraced_run_with_prompt_compaction_on_keeps_and_compares_no_prompt(
    mocker, monkeypatch, mapping_calls, format_value_calls, no_global_provider, on_the_config
):
    monkeypatch.setenv("DYNAMIQ_TRACING_PROMPT_COMPACTION_ENABLED", "true")
    to_dict = mocker.spy(Node, "to_dict")
    resolve_settings = mocker.spy(settings, "resolve_settings")
    keep_prompt = mocker.spy(registry, "keep_prompt")
    config = RunnableConfig(tracing=TracingConfig(prompt_compaction_enabled=True)) if on_the_config else None

    result = run_reporting_prompts(config)

    assert result.status == RunnableStatus.SUCCESS
    assert {name for name, spy in mapping_calls.items() if spy.call_count} == set()
    # The setting is not even read: settings are resolved for a run tree that has a tracer.
    assert resolve_settings.call_count == 0
    assert keep_prompt.call_count == 0
    assert to_dict.call_count == 0
    assert format_value_calls == []
    assert registry.size() == 0


def test_sampled_out_run_with_prompt_compaction_on_keeps_and_compares_no_prompt(exporter, mocker, mapping_calls):
    provider = sdk_trace.TracerProvider(sampler=sampling.ALWAYS_OFF)
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    keep_prompt = mocker.spy(registry, "keep_prompt")
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, prompt_compaction_enabled=True))

    result = run_reporting_prompts(config)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()
    assert {name for name, spy in mapping_calls.items() if spy.call_count} - SCOPE == {"node_span_spec"}
    assert keep_prompt.call_count == 0
    assert registry.size() == 0


def test_sampled_out_run_exports_nothing_and_maps_only_span_specs(exporter, mapping_calls):
    provider = sdk_trace.TracerProvider(sampler=sampling.ALWAYS_OFF)
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    result = two_step_workflow().run(input_data={"value": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()
    assert {name for name, spy in mapping_calls.items() if spy.call_count} - SCOPE == SPAN_SPECS
    assert registry.size() == 0


def test_run_under_a_sampled_out_parent_is_sampled_out(exporter):
    # The default sampler follows the parent, so an unsampled request keeps its whole run out of the export.
    unsampled = sdk_trace.TracerProvider(sampler=sampling.ALWAYS_OFF)
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    with unsampled.get_tracer("application").start_as_current_span("request"):
        two_step_workflow().run(input_data={"value": 1}, config=config)

    assert exporter.get_finished_spans() == ()


def test_sampled_out_node_still_parents_its_children(exporter):
    # A sampler that drops one span by name: its child must stay in the trace, under the dropped span's id.
    class DropFirst(sampling.Sampler):
        def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, **kwargs):
            decision = sampling.Decision.DROP if name == "outer" else sampling.Decision.RECORD_AND_SAMPLE
            return sampling.SamplingResult(decision, attributes)

        def get_description(self) -> str:
            return "DropFirst"

    class Calling(Echo):
        def execute(self, input_data, config=None, **kwargs):
            return Echo(name="inner").run(input_data, config=config, parent_run_id=kwargs["run_id"]).output

    provider = sdk_trace.TracerProvider(sampler=DropFirst())
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    with provider.get_tracer("application").start_as_current_span("request") as request:
        Calling(name="outer").run(input_data={"value": 1}, config=config)

    assert sorted(span.name for span in exporter.get_finished_spans()) == ["inner", "request"]
    inner = span_named(exporter, "inner")
    assert inner.context.trace_id == request.get_span_context().trace_id
    assert parent_span_id(inner) != request.get_span_context().span_id


def test_tracing_disabled_on_the_config_emits_no_spans(exporter, provider, mapping_calls):
    config = RunnableConfig(tracing=TracingConfig(enabled=False, tracer_provider=provider))

    result = two_step_workflow().run(input_data={"value": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()
    assert {name for name, spy in mapping_calls.items() if spy.call_count} == set()


@pytest.mark.parametrize("value", ["false", "FALSE", "0"])
def test_tracing_disabled_in_the_environment_emits_no_spans(exporter, traced, monkeypatch, value):
    monkeypatch.setenv("DYNAMIQ_TRACING_ENABLED", value)

    result = two_step_workflow().run(input_data={"value": 1}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()


def test_sdk_disabled_in_the_environment_emits_no_spans_and_maps_nothing(exporter, monkeypatch, mapping_calls):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(sdk_trace.export.SimpleSpanProcessor(exporter))
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider))

    two_step_workflow().run(input_data={"value": 1}, config=config)

    assert exporter.get_finished_spans() == ()
    # A disabled SDK is found out through the tracer it hands out, which is asked for once per provider.
    assert {name for name, spy in mapping_calls.items() if spy.call_count} - SCOPE == set()


def test_run_is_traced_through_the_global_provider():
    # The global provider can be set once per process, so this path runs in a process of its own.
    script = textwrap.dedent(
        """
        import json

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from dynamiq import Workflow
        from dynamiq.flows import Flow
        from dynamiq.nodes.utils import Input, Output
        from dynamiq.telemetry import registry

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        start, end = Input(id="start", name="start"), Output(id="end", name="end")
        end.depends_on(start)
        result = Workflow(name="global", flow=Flow(nodes=[start, end])).run(input_data={"value": 1})
        spans = exporter.get_finished_spans()
        print(json.dumps({
            "status": result.status.value,
            "names": sorted(span.name for span in spans),
            "roots": [span.name for span in spans if span.parent is None],
            "scopes": sorted({span.instrumentation_scope.name for span in spans}),
            "registry": registry.size(),
        }))
        """
    )

    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {
        "status": "success",
        "names": ["end", "invoke_workflow global", "start"],
        "roots": ["invoke_workflow global"],
        "scopes": ["dynamiq"],
        "registry": 0,
    }
