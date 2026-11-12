"""Content capture: with it off no span holds a string that came from the run's data."""

import json

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableStatus
from dynamiq.types.telemetry import ContentCaptureMode
from tests.helpers.agent_hooks import ExplodingTool, SearchTool, build_agent, final_step, tool_step
from tests.integration.telemetry.conftest import all_strings
from tests.integration.telemetry.test_llm_workflow import build_workflow

QUESTION = "zebra-question-7731"
TOOL_RESULT = "zebra-tool-result-7731"
ANSWER = "zebra-final-answer-7731"
ROLE = "help the user"  # the agent's role is part of its system prompt
TEMPLATE_VALUE = "zebra-template-value-7731"
FAILURE = "zebra-failure-message-7731"
SECRETS = [QUESTION, TOOL_RESULT, ANSWER, TEMPLATE_VALUE, FAILURE]

CONTENT_ATTRIBUTES = (
    "dynamiq.input",
    "dynamiq.output",
    "dynamiq.node.config",
    "dynamiq.human_feedback",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.tool.definitions",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
    "gen_ai.retrieval.query.text",
    "gen_ai.retrieval.documents",
    "gen_ai.system_instructions",
)


class Exploding(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "exploding"

    def execute(self, input_data, config=None, **kwargs):
        raise ValueError(FAILURE)


def run_rich_scenario(script_llm, config):
    """An agent with a tool, a plain LLM workflow and a failing node, all run with the same config."""
    agent = build_agent(
        tools=[SearchTool(result=TOOL_RESULT), ExplodingTool(body=FAILURE)],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )
    steps = [tool_step("search", query=QUESTION), tool_step("http", query=QUESTION), final_step(ANSWER)]
    script_llm(InferenceMode.FUNCTION_CALLING, steps)
    agent_result = agent.run(input_data={"input": QUESTION, "session_id": "s", "user_id": "u"}, config=config)
    assert agent_result.status == RunnableStatus.SUCCESS

    llm_result = build_workflow().run(input_data={"topic": TEMPLATE_VALUE}, config=config)
    assert llm_result.status == RunnableStatus.SUCCESS

    start = Input(id="in", name="start")
    broken = Exploding(id="broken", depends=[NodeDependency(start)])
    out = Output(id="out", name="end", depends=[NodeDependency(broken)])
    failed = Workflow(flow=Flow(nodes=[start, broken, out])).run(input_data={"v": QUESTION}, config=config)
    assert failed.status == RunnableStatus.FAILURE


def strings_with(spans, needles):
    return {(span.name, needle) for span in spans for text in all_strings(span) for needle in needles if needle in text}


def test_content_is_on_by_default(script_llm, config, trace):
    run_rich_scenario(script_llm, config)

    found = {needle for _, needle in strings_with(trace().spans, SECRETS + [ROLE])}
    assert found == set(SECRETS + [ROLE])


@pytest.fixture(params=["config", "env", "env-legacy", "event-only"])
def content_off(request, make_config, monkeypatch):
    """Four ways to turn content capture off."""
    match request.param:
        case "config":
            return make_config(content_capture=ContentCaptureMode.NO_CONTENT)
        case "env":
            monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "NO_CONTENT")
        case "env-legacy":
            monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
        case "event-only":
            # Dynamiq emits no log-record events, so EVENT_ONLY records nothing
            monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT", "EVENT_ONLY")
    return make_config()


def test_no_content_leaves_nothing_of_the_data_on_any_span(script_llm, content_off, trace):
    run_rich_scenario(script_llm, content_off)

    spans = trace().spans
    assert len(spans) > 10
    assert strings_with(spans, SECRETS + [ROLE]) == set()
    for span in spans:
        for key in CONTENT_ATTRIBUTES:
            assert key not in span.attributes, f"{key} on {span.name}"
        for event in span.events:
            for key in ("exception.message", "exception.stacktrace", "dynamiq.hook.details"):
                assert key not in (event.attributes or {})
        if span.status.status_code.name == "ERROR":
            assert not span.status.description


def test_no_content_keeps_the_structure(script_llm, content_off, trace):
    run_rich_scenario(script_llm, content_off)

    spans = trace()
    chat = spans.named("chat gpt-4o")[0]
    assert chat.attributes["gen_ai.request.model"] == "gpt-4o"
    assert chat.attributes["gen_ai.usage.input_tokens"] > 0
    assert chat.attributes["dynamiq.usage.total_tokens"] > 0
    assert chat.attributes["dynamiq.node.name"]
    tool = spans.one("execute_tool search")
    assert tool.attributes["gen_ai.tool.name"] == "search"
    assert tool.attributes["dynamiq.run.status"] == "succeeded"
    failed = spans.one("exploding")
    assert failed.attributes["dynamiq.run.status"] == "failed"
    assert failed.attributes["error.type"] == "ValueError"
    (event,) = (e for e in failed.events if e.name == "exception")
    assert event.attributes["exception.type"] == "ValueError"
    (root,) = (s for s in spans.spans if s.attributes.get("dynamiq.run.status") == "failed" and s.parent is None)
    assert root.attributes["error.type"] == "FlowNodeFailureException"
    # ids and names stay, the error message is content
    assert json.loads(root.attributes["dynamiq.failed_nodes"]) == [{"id": "broken", "name": "exploding"}]


def test_node_config_is_off_by_default_and_follows_content(script_llm, make_config, trace):
    agent = build_agent(tools=[SearchTool()], inference_mode=InferenceMode.FUNCTION_CALLING)
    script_llm(InferenceMode.FUNCTION_CALLING, [final_step("x")])

    agent.run(input_data={"input": "q"}, config=make_config())
    assert not [s for s in trace().spans if "dynamiq.node.config" in s.attributes]


def test_node_config_when_enabled_has_no_prompt_messages_or_secrets(mock_llm_executor, make_config, trace):
    build_workflow().run(input_data={"topic": TEMPLATE_VALUE}, config=make_config(node_config_enabled=True))

    chat = trace().one("chat gpt-4o")
    config = json.loads(chat.attributes["dynamiq.node.config"])
    assert config["model"] == "openai/gpt-4o"
    assert "messages" not in (config.get("prompt") or {})
    assert "test-key" not in chat.attributes["dynamiq.node.config"]
    for span in trace().spans:
        assert "test-key" not in " ".join(all_strings(span))


def test_node_config_is_not_recorded_without_content(mock_llm_executor, make_config, trace):
    config = make_config(node_config_enabled=True, content_capture=ContentCaptureMode.NO_CONTENT)

    build_workflow().run(input_data={"topic": TEMPLATE_VALUE}, config=config)

    assert not [s for s in trace().spans if "dynamiq.node.config" in s.attributes]


def test_content_max_length_truncates_strings_but_keeps_json_valid(mock_llm_executor, make_config, trace):
    long_value = "x" * 500

    build_workflow().run(input_data={"topic": long_value}, config=make_config(content_max_length=40))

    chat = trace().one("chat gpt-4o")
    messages = json.loads(chat.attributes["gen_ai.input.messages"])
    text = messages[1]["parts"][0]["content"]
    assert 0 < len(text) < 100
    assert json.loads(chat.attributes["dynamiq.input"])["topic"] != long_value


def test_redacted_keys_are_replaced_at_any_depth(config, trace):
    class Secretive(Node):
        group: NodeGroup = NodeGroup.UTILS
        name: str = "secretive"

        def execute(self, input_data, config=None, **kwargs):
            return {"nested": {"mcp_http_headers": {"Authorization": "Bearer zebra-out"}}, "visible": "yes"}

    Workflow(flow=Flow(nodes=[Secretive()])).run(
        input_data={"mcp_http_headers": {"Authorization": "Bearer zebra-in"}, "other": 1}, config=config
    )

    for span in trace().spans:
        assert "zebra-" not in " ".join(all_strings(span))
    node = trace().one("secretive")
    assert json.loads(node.attributes["dynamiq.input"]) == {"mcp_http_headers": "***", "other": 1}
    assert json.loads(node.attributes["dynamiq.output"]) == {"nested": {"mcp_http_headers": "***"}, "visible": "yes"}
