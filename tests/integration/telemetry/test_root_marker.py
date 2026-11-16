"""``dynamiq.run.is_root`` across real run trees: one marked span per run that no Dynamiq run started."""

from typing import Any
from unittest.mock import MagicMock

from litellm import ModelResponse
from litellm.exceptions import RateLimitError

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.base import FallbackConfig
from dynamiq.nodes.operators.sub_workflow import SubWorkflow
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.helpers.agent_hooks import SearchTool, build_agent, final_step, tool_step
from tests.integration.telemetry.rebuild import rebuild

IS_ROOT = "dynamiq.run.is_root"


class Detached(Node):
    """Runs a node without naming its own run as the parent run."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "detached"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        return Input(id="inner", name="inner").run(input_data=input_data, config=config).output


def root_names(spans: list) -> list[str]:
    marked = [span for span in spans if IS_ROOT in span.attributes]
    assert all(span.attributes[IS_ROOT] is True for span in marked)
    return sorted(span.name for span in marked)


def test_agent_run_marks_the_agent_and_none_of_its_children(script_llm, config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    child = build_agent(tools=[SearchTool()], inference_mode=mode, id="child", name="Researcher")
    parent = build_agent(
        tools=[SubAgentTool(agent=child, name="Researcher", description="Researches.")],
        inference_mode=mode,
        id="parent",
        name="Boss",
    )
    steps = [
        tool_step("Researcher", input="dig"),
        tool_step("search", query="zz"),
        final_step("child answer"),
        final_step("parent answer"),
    ]
    script_llm(mode, steps)

    result = parent.run(input_data={"input": "hello"}, config=config)

    assert result.output["content"] == "parent answer"
    spans = trace().spans
    # two model calls and the sub-agent under the agent; two model calls and a tool under the sub-agent
    assert len(spans) == 7
    assert root_names(spans) == ["invoke_agent Boss"]


def test_workflow_with_an_agent_marks_the_workflow_only(script_llm, config, trace):
    mode = InferenceMode.XML
    agent = build_agent(tools=[SearchTool()], inference_mode=mode)
    script_llm(mode, [tool_step("search", query="q"), final_step("ok")])

    result = Workflow(flow=Flow(nodes=[agent])).run(input_data={"input": "hello"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace().spans
    assert len(spans) == 5
    assert root_names(spans) == ["invoke_workflow Workflow"]


def test_sub_workflow_is_not_a_root(config, trace):
    inner = Flow(id="inner-flow", name="inner", nodes=[Input(id="inner-in", name="inner-in")])
    start = Input(id="start", name="start")
    sub = SubWorkflow(id="sub", name="sub", flow=inner)
    sub.depends_on(start)
    end = Output(id="end", name="end")
    end.depends_on(sub)

    result = Workflow(name="outer", flow=Flow(nodes=[start, sub, end])).run(input_data={"x": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    # the flow of the sub-workflow has a span of its own, started by the node that holds it
    assert spans.one("invoke_workflow inner").attributes["dynamiq.run.type"] == "flow"
    assert len(spans.spans) == 6
    assert root_names(spans.spans) == ["invoke_workflow outer"]


def llm_with_fallback() -> OpenAI:
    """An LLM whose model call is rate limited and whose fallback LLM answers."""

    def llm(model: str) -> OpenAI:
        prompt = Prompt(messages=[Message(role="user", content="Hi")])
        return OpenAI(model=model, connection=connections.OpenAI(api_key="test-key"), prompt=prompt)

    primary, fallback = llm("gpt-4o-mini"), llm("gpt-4o")
    primary.fallback = FallbackConfig(llm=fallback, enabled=True)
    primary._completion = MagicMock(
        side_effect=RateLimitError(message="slow down", model="gpt-4o-mini", llm_provider="openai")
    )
    reply = ModelResponse()
    reply["choices"][0]["message"]["content"] = "from the fallback"
    fallback._completion = MagicMock(return_value=reply)
    return primary


def test_fallback_of_an_llm_run_on_its_own_is_a_root_of_its_own(config, trace):
    # The fallback runs beside the failed run, with that run's parent run, and here that run has none.
    result = llm_with_fallback().run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert root_names(trace().spans) == ["chat gpt-4o", "chat gpt-4o-mini"]


def test_fallback_of_an_llm_in_a_workflow_is_not_a_root(config, trace):
    result = Workflow(flow=Flow(nodes=[llm_with_fallback()])).run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    spans = trace().spans
    assert len(spans) == 3
    assert root_names(spans) == ["invoke_workflow Workflow"]


def test_rebuilt_root_has_no_parent_run_even_below_another_run_span(make_config, trace):
    # The run-based handler sees a run that names no parent run as a top-most one. A rebuild from spans agrees
    # through the marker; by the span tree alone the run would be a child of the node that ran it.
    handler = TracingCallbackHandler()

    result = Detached().run(input_data={"v": 1}, config=make_config(callbacks=[handler]))

    assert result.status == RunnableStatus.SUCCESS
    spans = trace()
    assert spans.parent_of(spans.one("inner")) is spans.one("detached")
    assert root_names(spans.spans) == ["detached", "inner"]
    rebuilt = rebuild(spans.spans)
    assert {run.name: run.parent_run_id for run in rebuilt.values()} == {"detached": None, "inner": None}
    assert {run.name: run.parent_run_id for run in handler.runs.values()} == {"detached": None, "inner": None}


def test_rebuilt_workflow_below_a_span_of_the_application_has_no_parent_run(provider, config, trace):
    with provider.get_tracer("application").start_as_current_span("request"):
        Workflow(flow=Flow(nodes=[Input(id="start", name="start")])).run(input_data={"v": 1}, config=config)

    spans = trace().spans
    assert root_names(spans) == ["invoke_workflow Workflow"]
    rebuilt = rebuild(spans)
    workflow_run = next(run for run in rebuilt.values() if run.type == "workflow")
    assert workflow_run.parent_run_id is None
    assert [run.parent_run_id for run in rebuilt.values() if run.type == "node"] == [workflow_run.id]
