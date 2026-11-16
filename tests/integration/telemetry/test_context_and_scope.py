"""Instrumentation scope, trace joining, other instrumentation inside a node, switches, async and hooks."""

import asyncio
from importlib.metadata import version
from typing import ClassVar

from pydantic import BaseModel, Field

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.hooks import CallLimitHook
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.telemetry import TracingConfig
from tests.helpers.agent_hooks import SearchTool, build_agent, final_step, tool_step
from tests.integration.telemetry.conftest import attrs_json
from tests.integration.telemetry.test_llm_workflow import build_workflow


class FetchInput(BaseModel):
    url: str = Field(default="", description="Address to fetch.")


class Instrumented(Node):
    """Starts a span of its own while it executes, as an HTTP client or a database driver would."""

    group: NodeGroup = NodeGroup.TOOLS
    name: str = "instrumented"
    # a class attribute: an agent clones its tools, and a provider cannot be copied into the clone
    providers: ClassVar[list] = []
    description: str = "Fetches an address."
    input_schema: ClassVar[type[FetchInput]] = FetchInput

    def execute(self, input_data, config=None, **kwargs):
        with self.providers[-1].get_tracer("http-client").start_as_current_span("http GET"):
            return {"content": "fetched"}


def test_instrumentation_scope(mock_llm_executor, config, trace):
    build_workflow().run(input_data={"topic": "hello"}, config=config)

    for span in trace().spans:
        scope = span.instrumentation_scope
        assert scope.name == "dynamiq"
        assert scope.version == version("dynamiq")
        assert scope.schema_url == "https://opentelemetry.io/schemas/1.41.1"
        assert scope.attributes["dynamiq.trace.shape_version"] == 1


def test_run_joins_the_active_trace(mock_llm_executor, provider, config, trace):
    tracer = provider.get_tracer("app")

    with tracer.start_as_current_span("request") as request:
        build_workflow().run(input_data={"topic": "hello"}, config=config)

    spans = trace()
    root = spans.one("invoke_workflow Workflow")
    assert root.parent.span_id == request.get_span_context().span_id
    assert {s.context.trace_id for s in spans.spans} == {request.get_span_context().trace_id}
    assert len(spans.roots()) == 1 and spans.roots()[0].name == "request"


def test_each_top_level_run_starts_its_own_trace(mock_llm_executor, config, trace):
    workflow = build_workflow()

    workflow.run(input_data={"topic": "one"}, config=config)
    workflow.run(input_data={"topic": "two"}, config=config)

    roots = trace().named("invoke_workflow Workflow")
    assert len(roots) == 2
    assert roots[0].context.trace_id != roots[1].context.trace_id
    assert all(r.parent is None for r in roots)


def test_spans_of_other_instrumentation_inside_a_node_are_its_children(provider, config, trace):
    Instrumented.providers.append(provider)
    node = Instrumented()

    result = Workflow(flow=Flow(nodes=[node])).run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    tool = spans.one("execute_tool instrumented")
    inner = spans.one("http GET")
    assert inner.parent.span_id == tool.context.span_id
    assert inner.instrumentation_scope.name == "http-client"
    assert spans.parent_of(tool).name == "invoke_workflow Workflow"


def test_spans_of_instrumentation_inside_a_tool_of_an_agent_nest_under_the_tool(script_llm, provider, config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    Instrumented.providers.append(provider)
    agent = build_agent(tools=[Instrumented(name="fetch")], inference_mode=mode)
    script_llm(mode, [tool_step("fetch"), final_step("ok")])

    agent.run(input_data={"input": "go"}, config=config)

    spans = trace()
    assert spans.parent_of(spans.one("http GET")).name == "execute_tool fetch"
    assert spans.parent_of(spans.one("execute_tool fetch")).name == "invoke_agent assistant"


def test_tracing_disabled_for_a_run_emits_nothing(mock_llm_executor, provider, exporter):
    config = RunnableConfig(tracing=TracingConfig(tracer_provider=provider, enabled=False))

    result = build_workflow().run(input_data={"topic": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()


def test_tracing_disabled_by_the_environment_emits_nothing(mock_llm_executor, config, exporter, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_TRACING_ENABLED", "false")

    result = build_workflow().run(input_data={"topic": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()


def test_run_without_a_tracing_config_and_without_a_global_provider_is_untraced(mock_llm_executor, exporter):
    result = build_workflow().run(input_data={"topic": "hello"}, config=RunnableConfig())

    assert result.status == RunnableStatus.SUCCESS
    assert exporter.get_finished_spans() == ()


def test_async_agent_run_has_the_same_tree(script_llm, config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    agent = build_agent(tools=[SearchTool(result="forty-two")], inference_mode=mode)
    script_llm(mode, [tool_step("search", query="q"), final_step("done")])

    result = asyncio.run(agent.run_async(input_data={"input": "hi", "session_id": "S"}, config=config))

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert spans.shape() == [
        "invoke_agent assistant",
        sorted([["chat gpt-4o", []], ["execute_tool search", []], ["chat gpt-4o", []]], key=repr),
    ]
    for span in spans.spans:
        assert span.attributes["gen_ai.conversation.id"] == "S"
        assert span.attributes["dynamiq.run.status"] == "succeeded"
    assert attrs_json(spans.one("execute_tool search"), "gen_ai.tool.call.result") == {"content": "forty-two"}


def test_hook_events_are_span_events_of_the_agent(script_llm, config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    agent = build_agent(
        tools=[SearchTool()], inference_mode=mode, hooks=[CallLimitHook(tools=["search"], max_per_run=1, name="limit")]
    )
    script_llm(mode, [tool_step("search", query="a"), tool_step("search", query="b"), final_step("done")])

    agent.run(input_data={"input": "hi"}, config=config)

    spans = trace()
    root = spans.one("invoke_agent assistant")
    events = [e for e in root.events if e.name == "dynamiq.agent.hook"]
    assert events, "the blocked call is reported as a hook event"
    for event in events:
        assert event.attributes["dynamiq.hook.name"] == "limit"
        assert event.attributes["dynamiq.hook.type"]
        assert event.attributes["dynamiq.hook.point"] == "before_tool"
        assert event.attributes["dynamiq.hook.decision"]
    assert "dynamiq.hook.details" in events[-1].attributes
    for span in spans.spans:
        if span is not root:
            assert not [e for e in span.events if e.name == "dynamiq.agent.hook"]
