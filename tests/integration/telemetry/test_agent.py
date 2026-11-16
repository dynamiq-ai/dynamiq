"""Agents with tools in every inference mode: the loop appears as chat and tool spans under the agent span."""

import pytest
from opentelemetry.trace import SpanKind

from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from tests.helpers.agent_hooks import (
    ALL_MODES,
    EmailTool,
    SearchTool,
    build_agent,
    final_step,
    parallel_step,
    tool_step,
)
from tests.integration.telemetry.conftest import OPERATION, attrs_json

USAGE = (11, 7)


def make_agent(mode, tools, **kwargs) -> Agent:
    # The mode is fixed at construction: the agent builds its tool schemas and prompt for it then.
    return build_agent(tools=tools, inference_mode=mode, **kwargs)


def run_agent(agent: Agent, script_llm, mode, steps, config, question="hello", **input_extra):
    assert agent.inference_mode == mode
    script_llm(mode, steps, usage=USAGE)
    return agent.run(input_data={"input": question, **input_extra}, config=config)


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_agent_loop_with_a_tool(mode, script_llm, config, trace):
    agent = make_agent(mode, [SearchTool(result="forty-two")], description="Answers questions.")

    result = run_agent(agent, script_llm, mode, [tool_step("search", query="q1"), final_step("done!")], config)

    assert result.output["content"] == "done!"
    spans = trace()
    (root,) = spans.roots()
    assert root.name == "invoke_agent assistant"
    assert root.kind == SpanKind.INTERNAL
    # loop iterations have no span: the llm and tool runs of all iterations are children of the agent, in order
    children = spans.children(root)
    assert [s.name for s in children] == ["chat gpt-4o", "execute_tool search", "chat gpt-4o"]
    assert [s.kind for s in children] == [SpanKind.CLIENT, SpanKind.INTERNAL, SpanKind.CLIENT]
    assert len(spans.spans) == 4

    a = root.attributes
    assert a[OPERATION] == "invoke_agent"
    assert a["gen_ai.agent.name"] == "assistant"
    assert a["gen_ai.agent.id"] == "agent"
    assert a["gen_ai.agent.description"] == "Answers questions."
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.request.model"] == "gpt-4o"
    assert attrs_json(root, "gen_ai.input.messages") == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert attrs_json(root, "gen_ai.output.messages") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "done!"}], "finish_reason": "stop"}
    ]
    assert a["dynamiq.run.status"] == "succeeded"
    assert a["dynamiq.node.type"] == "dynamiq.nodes.agents.Agent"
    assert a["dynamiq.node.group"] == "agents"
    # no token usage on agent spans
    assert not [key for key in a if "usage" in key]


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_tool_span_arguments_and_result(mode, script_llm, config, trace):
    agent = make_agent(mode, [SearchTool(result="forty-two")])

    run_agent(agent, script_llm, mode, [tool_step("search", query="q1"), final_step("done!")], config)

    tool = trace().one("execute_tool search")
    a = tool.attributes
    assert a[OPERATION] == "execute_tool"
    assert a["gen_ai.tool.name"] == "search"
    assert a["gen_ai.tool.type"] == "function"
    assert attrs_json(tool, "gen_ai.tool.call.arguments") == {"query": "q1", "mode": "slow"}
    assert attrs_json(tool, "gen_ai.tool.call.result") == {"content": "forty-two"}
    # the same content is not repeated under the generic names
    assert "dynamiq.input" not in a
    assert "dynamiq.output" not in a
    assert "gen_ai.tool.call.id" not in a
    assert "gen_ai.tool.description" not in a
    assert a["dynamiq.node.name"] == "search"
    assert a["dynamiq.node.group"] == "tools"


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_chat_spans_of_the_loop(mode, script_llm, config, trace):
    agent = make_agent(mode, [SearchTool(result="forty-two")])

    run_agent(agent, script_llm, mode, [tool_step("search", query="q1"), final_step("done!")], config)

    spans = trace()
    first, second = (s for s in spans.children(spans.roots()[0]) if s.name == "chat gpt-4o")
    tool = spans.one("execute_tool search")
    for chat in (first, second):
        assert chat.attributes[OPERATION] == "chat"
        assert chat.attributes["gen_ai.usage.input_tokens"] == USAGE[0]
        assert chat.attributes["gen_ai.usage.output_tokens"] == USAGE[1]
        assert chat.attributes["dynamiq.usage.total_tokens"] == sum(USAGE)
        assert attrs_json(chat, "gen_ai.input.messages")[0]["role"] == "system"
    # the whole conversation is sent on every model call: the second request is longer than the first
    first_messages = attrs_json(first, "gen_ai.input.messages")
    second_messages = attrs_json(second, "gen_ai.input.messages")
    assert len(second_messages) > len(first_messages)
    # the first request holds the question, the second one the observation of the tool
    assert "hello" in str(first_messages)
    assert "forty-two" in str(second_messages)
    assert "forty-two" not in str(first_messages)
    # run.depends orders the loop: the tool follows the first model call and the second call follows the tool
    assert list(tool.attributes["dynamiq.run.depends"]) == [first.attributes["dynamiq.node.id"]]
    assert list(second.attributes["dynamiq.run.depends"]) == [tool.attributes["dynamiq.node.id"]]
    assert "dynamiq.run.depends" not in first.attributes
    # the same llm node ran twice: two spans, one node id
    assert first.attributes["dynamiq.node.id"] == second.attributes["dynamiq.node.id"]
    assert first.attributes["dynamiq.run.id"] != second.attributes["dynamiq.run.id"]


def test_function_calling_messages_carry_tool_calls_and_responses(script_llm, config, trace):
    agent = make_agent(InferenceMode.FUNCTION_CALLING, [SearchTool(result="forty-two"), EmailTool()])
    steps = [tool_step("search", query="q1"), final_step("x")]

    run_agent(agent, script_llm, InferenceMode.FUNCTION_CALLING, steps, config)

    spans = trace()
    first, second = (s for s in spans.children(spans.roots()[0]) if s.name == "chat gpt-4o")
    (output,) = attrs_json(first, "gen_ai.output.messages")
    assert output["finish_reason"] == "tool_call"
    (call,) = output["parts"]
    assert call["type"] == "tool_call"
    assert call["name"] == "search"
    assert call["arguments"]["query"] == "q1"  # the parsed object, not a string
    assert list(first.attributes["gen_ai.response.finish_reasons"]) == ["tool_call"]
    messages = attrs_json(second, "gen_ai.input.messages")
    responses = [p for m in messages for p in m["parts"] if p["type"] == "tool_call_response"]
    assert [r["id"] for r in responses] == [call["id"]]
    assert "forty-two" in str(responses[0]["response"])
    calls = [p for m in messages for p in m["parts"] if p["type"] == "tool_call"]
    assert [c["id"] for c in calls] == [call["id"]]
    tool_definitions = attrs_json(first, "gen_ai.tool.definitions")
    assert {"search", "send-email"} <= {d["name"] for d in tool_definitions}
    assert all(d["type"] == "function" and "parameters" in d for d in tool_definitions)


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda m: m.value)
def test_session_and_user_ids_come_from_the_agent_input(mode, script_llm, config, trace):
    agent = make_agent(mode, [SearchTool()])

    run_agent(
        agent,
        script_llm,
        mode,
        [tool_step("search", query="q"), final_step("ok")],
        config,
        session_id="S-1",
        user_id="U-1",
    )

    spans = trace().spans
    assert len(spans) == 4
    for span in spans:
        assert span.attributes["gen_ai.conversation.id"] == "S-1"
        assert span.attributes["session.id"] == "S-1"
        assert span.attributes["user.id"] == "U-1"


def test_run_config_session_wins_over_the_agent_input(script_llm, make_config, trace):
    agent = make_agent(InferenceMode.FUNCTION_CALLING, [SearchTool()])

    run_agent(
        agent,
        script_llm,
        InferenceMode.FUNCTION_CALLING,
        [final_step("ok")],
        make_config(session_id="from-config", user_id="cfg-user"),
        session_id="from-input",
        user_id="input-user",
    )

    for span in trace().spans:
        assert span.attributes["gen_ai.conversation.id"] == "from-config"
        assert span.attributes["user.id"] == "cfg-user"


def test_no_session_attributes_when_none_is_known(script_llm, config, trace):
    agent = make_agent(InferenceMode.FUNCTION_CALLING, [SearchTool()])

    run_agent(agent, script_llm, InferenceMode.FUNCTION_CALLING, [final_step("ok")], config)

    for span in trace().spans:
        assert "gen_ai.conversation.id" not in span.attributes
        assert "session.id" not in span.attributes
        assert "user.id" not in span.attributes


def test_parallel_tool_calls_are_siblings_under_the_agent(script_llm, config, trace):
    agent = make_agent(InferenceMode.FUNCTION_CALLING, [SearchTool(), EmailTool()])
    steps = [parallel_step(("search", {"query": "a"}), ("send-email", {"to": "x@y.z", "body": "b"})), final_step("fin")]

    run_agent(agent, script_llm, InferenceMode.FUNCTION_CALLING, steps, config)

    spans = trace()
    root = spans.roots()[0]
    tools = [s for s in spans.children(root) if s.name.startswith("execute_tool")]
    assert sorted(s.name for s in tools) == ["execute_tool search", "execute_tool send-email"]
    by_name = {s.attributes["gen_ai.tool.name"]: s for s in tools}
    assert attrs_json(by_name["search"], "gen_ai.tool.call.arguments")["query"] == "a"
    assert attrs_json(by_name["send-email"], "gen_ai.tool.call.arguments") == {"to": "x@y.z", "body": "b"}
    assert attrs_json(by_name["send-email"], "gen_ai.tool.call.result") == {"content": "sent to x@y.z"}
    chats = [s for s in spans.children(root) if s.name == "chat gpt-4o"]
    assert len(chats) == 2
    assert len({s.context.trace_id for s in spans.spans}) == 1


def test_failed_tool_is_failed_while_the_agent_recovers(script_llm, config, trace):
    from tests.helpers.agent_hooks import ExplodingTool

    agent = make_agent(InferenceMode.FUNCTION_CALLING, [ExplodingTool()])
    steps = [tool_step("http", query="x"), final_step("recovered")]

    result = run_agent(agent, script_llm, InferenceMode.FUNCTION_CALLING, steps, config)

    assert result.output["content"] == "recovered"
    spans = trace()
    tool = spans.one("execute_tool http")
    assert tool.attributes["dynamiq.run.status"] == "failed"
    assert tool.attributes["error.type"]
    assert [e.name for e in tool.events if e.name == "exception"] == ["exception"]
    assert spans.roots()[0].attributes["dynamiq.run.status"] == "succeeded"


def test_sub_agent_as_a_tool(script_llm, config, trace):
    mode = InferenceMode.FUNCTION_CALLING
    child = make_agent(mode, [SearchTool()], id="child", name="Researcher", description="Researches.")
    parent = make_agent(
        mode,
        [SubAgentTool(agent=child, name="Researcher", description="Researches.")],
        id="parent",
        name="Boss",
    )
    steps = [
        tool_step("Researcher", input="dig"),
        tool_step("search", query="zz"),
        final_step("child answer"),
        final_step("parent answer"),
    ]

    result = run_agent(parent, script_llm, InferenceMode.FUNCTION_CALLING, steps, config, session_id="S-1")

    assert result.output["content"] == "parent answer"
    spans = trace()
    (root,) = spans.roots()
    assert root.name == "invoke_agent Boss"
    assert [s.name for s in spans.children(root)] == ["chat gpt-4o", "invoke_agent Researcher", "chat gpt-4o"]
    sub = spans.one("invoke_agent Researcher")
    assert sub.kind == SpanKind.INTERNAL
    assert sub.attributes[OPERATION] == "invoke_agent"
    assert sub.attributes["gen_ai.agent.name"] == "Researcher"
    assert sub.attributes["gen_ai.agent.description"] == "Researches."
    assert [s.name for s in spans.children(sub)] == ["chat gpt-4o", "execute_tool search", "chat gpt-4o"]
    assert attrs_json(sub, "gen_ai.output.messages")[0]["parts"] == [{"type": "text", "content": "child answer"}]
    # the sub-agent gets its own session id derived from the parent's, and its spans carry the one it was given
    child_session = sub.attributes["gen_ai.conversation.id"]
    assert child_session.startswith("S-1")
    for span in spans.spans:
        in_child = span is sub or spans.parent_of(span) is sub
        assert span.attributes["gen_ai.conversation.id"] == (child_session if in_child else "S-1")
        assert span.attributes["session.id"] == span.attributes["gen_ai.conversation.id"]
    # the child's run depends on the model call of the parent that asked for it
    parent_first_chat = spans.children(root)[0]
    assert list(sub.attributes["dynamiq.run.depends"]) == [parent_first_chat.attributes["dynamiq.node.id"]]


@pytest.mark.parametrize(
    "mode", [InferenceMode.XML, InferenceMode.DEFAULT, InferenceMode.STRUCTURED_OUTPUT], ids=lambda m: m.value
)
def test_streaming_agent_has_the_same_tree(mode, script_llm, config, trace):
    from dynamiq.types.streaming import StreamingConfig

    agent = make_agent(mode, [SearchTool(result="forty-two")], streaming=StreamingConfig(enabled=True))

    result = run_agent(agent, script_llm, mode, [tool_step("search", query="q1"), final_step("done!")], config)

    assert result.output["content"] == "done!"
    spans = trace()
    (root,) = spans.roots()
    assert [s.name for s in spans.children(root)] == ["chat gpt-4o", "execute_tool search", "chat gpt-4o"]
    for chat in spans.named("chat gpt-4o"):
        assert chat.attributes["gen_ai.request.stream"] is True
        assert attrs_json(chat, "gen_ai.output.messages")[0]["role"] == "assistant"
        assert chat.attributes["dynamiq.run.status"] == "succeeded"
    assert attrs_json(root, "gen_ai.output.messages")[0]["parts"] == [{"type": "text", "content": "done!"}]
