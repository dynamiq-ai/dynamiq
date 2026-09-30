"""Before/after tool hooks, driven by a real agent loop."""

import json
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.hooks import ToolHook
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus

CALLS: list[dict] = []


class SearchSchema(BaseModel):
    query: str = Field(default="", description="Search query.")
    mode: str = Field(default="slow", description="Search mode.")


class SearchTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search"
    description: str = "Searches the web."
    input_schema: ClassVar[type[SearchSchema]] = SearchSchema

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        CALLS.append(input_data.model_dump())
        return {"content": f"results for {input_data.query} ({input_data.mode})"}


@pytest.fixture
def test_llm():
    return OpenAI(
        connection=OpenAIConnection(api_key="test-api-key"),
        model="gpt-4o",
        max_tokens=100,
        temperature=0,
    )


@pytest.fixture(autouse=True)
def clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def build_agent(test_llm, hooks: list[ToolHook]) -> Agent:
    return Agent(
        id="agent",
        name="researcher",
        llm=test_llm,
        role="research",
        tools=[SearchTool()],
        tool_hooks=hooks,
    )


def drive_agent(agent: Agent, seen: list[str], calls: int = 1):
    """Make the agent call `search` `calls` times, then finish. Records every prompt it sees."""
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    steps = [
        json.dumps(
            {
                "thought": "search",
                "action": "search",
                "action_input": {"query": f"q{i}", "mode": "slow"},
            }
        )
        for i in range(calls)
    ]
    steps.append(json.dumps({"thought": "done", "action": "finish", "action_input": "All set."}))
    stream = iter(steps)

    def run(**kwargs):
        seen.append(str(kwargs.get("prompt")))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(stream)}
        return result

    return patch.object(agent.llm, "run", side_effect=run)


def run_agent(agent: Agent, seen: list[str], calls: int = 1):
    with drive_agent(agent, seen, calls):
        return agent.run(input_data={"input": "find things"})


def test_input_hook_rewrites_what_the_tool_receives(test_llm):
    hook = ToolHook(
        tools=["search"],
        input_transformer={"selector": {"query": "$.query", "mode": "fast"}},  # constant injected
    )
    result = run_agent(build_agent(test_llm, [hook]), [])

    assert result.status == RunnableStatus.SUCCESS
    assert CALLS == [{"query": "q0", "mode": "fast"}]


def test_output_hook_rewrites_what_the_llm_observes(test_llm):
    hook = ToolHook(tools=["search"], output_transformer={"selector": {"content": "redacted"}})
    seen: list[str] = []
    result = run_agent(build_agent(test_llm, [hook]), seen)

    assert result.status == RunnableStatus.SUCCESS
    assert CALLS  # the tool did run
    assert "results for q0" not in seen[-1]
    assert "redacted" in seen[-1]


def test_block_returns_message_as_observation_and_agent_continues(test_llm):
    hook = ToolHook(tools=["search"], block=True, block_message="Search is disabled.")
    seen: list[str] = []
    result = run_agent(build_agent(test_llm, [hook]), seen)

    assert result.status == RunnableStatus.SUCCESS
    assert CALLS == []
    assert "Search is disabled." in seen[-1]


def test_block_with_stop_agent_fails_the_run(test_llm):
    hook = ToolHook(tools=["search"], block=True, block_message="Forbidden.", stop_agent=True)
    result = run_agent(build_agent(test_llm, [hook]), [])

    assert result.status == RunnableStatus.FAILURE
    assert CALLS == []
    assert "Forbidden." in str(result.error.message)


def test_hook_for_another_tool_is_ignored(test_llm):
    hook = ToolHook(tools=["other-tool"], block=True)
    result = run_agent(build_agent(test_llm, [hook]), [])

    assert result.status == RunnableStatus.SUCCESS
    assert len(CALLS) == 1


def test_empty_tools_matches_every_tool(test_llm):
    hook = ToolHook(block=True, block_message="No tools today.")
    seen: list[str] = []
    run_agent(build_agent(test_llm, [hook]), seen)

    assert CALLS == []
    assert "No tools today." in seen[-1]


def test_input_hooks_run_in_order_and_output_hooks_in_reverse():
    first = ToolHook(
        input_transformer={"selector": {"query": "first"}},
        output_transformer={"selector": {"content": "out-first"}},
    )
    second = ToolHook(
        input_transformer={"selector": {"query": "$.query"}},
        output_transformer={"selector": {"content": "$.content"}},
    )
    from dynamiq.nodes.agents.hooks import apply_after_tool_hooks, apply_before_tool_hooks

    # before: first then second -> "first" survives through second's passthrough
    assert apply_before_tool_hooks([first, second], "t", {"query": "orig"}) == {"query": "first"}
    # after: second then first -> first has the last word
    assert apply_after_tool_hooks([first, second], "t", "raw") == "out-first"


def test_output_hook_applies_once_when_second_call_is_a_cache_hit(test_llm):
    """Second identical call is served from the tool cache; the output hook still applies exactly once."""
    hook = ToolHook(tools=["search"], output_transformer={"selector": {"content": "redacted"}})
    agent = build_agent(test_llm, [hook])
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    same_call = json.dumps(
        {
            "thought": "s",
            "action": "search",
            "action_input": {"query": "same", "mode": "slow"},
        }
    )
    steps = iter(
        [
            same_call,
            same_call,
            json.dumps({"thought": "d", "action": "finish", "action_input": "ok"}),
        ]
    )
    seen: list[str] = []

    def run(**kwargs):
        seen.append(str(kwargs.get("prompt")))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(steps)}
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "go"})

    assert result.status == RunnableStatus.SUCCESS
    assert len(CALLS) == 1  # second call came from the cache
    assert "results for same" not in seen[-1]
    assert seen[-1].count("redacted") >= 2


class ProfileSchema(BaseModel):
    user: str = Field(default="", description="User to look up.")


class ProfileTool(Node):
    """Returns structured content, like most real tools."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "profile"
    description: str = "Looks up a profile."
    input_schema: ClassVar[type[ProfileSchema]] = ProfileSchema

    def execute(self, input_data: ProfileSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        return {"content": {"name": "Ann", "internal_notes": "s3cret"}}


def test_output_hook_selects_fields_from_structured_content(test_llm):
    hook = ToolHook(
        tools=["profile"],
        output_transformer={"selector": {"content": "$.content.name"}},
    )
    agent = Agent(
        id="agent",
        name="a",
        llm=test_llm,
        role="r",
        tools=[ProfileTool()],
        tool_hooks=[hook],
    )
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    steps = iter(
        [
            json.dumps({"thought": "p", "action": "profile", "action_input": {}}),
            json.dumps({"thought": "d", "action": "finish", "action_input": "ok"}),
        ]
    )
    seen: list[str] = []

    def run(**kwargs):
        seen.append(str(kwargs.get("prompt")))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(steps)}
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "who"})

    assert result.status == RunnableStatus.SUCCESS
    assert "Ann" in seen[-1]
    assert "s3cret" not in seen[-1]


def test_output_hook_selects_from_json_text_and_serializes_objects_back():
    from dynamiq.nodes.agents.hooks import apply_after_tool_hooks

    raw = json.dumps({"public": {"name": "Ann"}, "internal": {"notes": "s3cret"}})
    pick_name = ToolHook(output_transformer={"selector": {"content": "$.content.public.name"}})
    pick_public = ToolHook(output_transformer={"selector": {"content": "$.content.public"}})

    assert apply_after_tool_hooks([pick_name], "t", raw) == "Ann"
    assert json.loads(apply_after_tool_hooks([pick_public], "t", raw)) == {"name": "Ann"}
    # plain text is left for the transformer as-is
    assert (
        apply_after_tool_hooks([ToolHook(output_transformer={"selector": {"content": "fixed"}})], "t", "x") == "fixed"
    )


def test_hook_activity_is_recorded_in_the_agent_trace(test_llm):
    from dynamiq.callbacks import TracingCallbackHandler

    hooks = [
        ToolHook(tools=["search"], input_transformer={"selector": {"query": "$.query", "mode": "fast"}}),
        ToolHook(tools=["search"], output_transformer={"selector": {"content": "redacted"}}),
    ]
    tracing = TracingCallbackHandler()
    agent = build_agent(test_llm, hooks)
    with drive_agent(agent, []):
        agent.run(input_data={"input": "go"}, config=RunnableConfig(callbacks=[tracing]))

    agent_run = next(run for run in tracing.runs.values() if "tool_hooks" in run.metadata)
    events = agent_run.metadata["tool_hooks"]
    assert [(e["phase"], e["hook"]) for e in events] == [("input", 0), ("output", 1)]
    assert events[0]["after"] == "{'query': 'q0', 'mode': 'fast'}"
    # the original output is never traced, only its size
    assert events[1]["after"] == "redacted" and "before" not in events[1] and events[1]["before_chars"] > 0


def test_block_is_traced_and_logged_before_raising(test_llm):
    from dynamiq.callbacks import TracingCallbackHandler

    tracing = TracingCallbackHandler()
    agent = build_agent(test_llm, [ToolHook(block=True, block_message="Nope.", stop_agent=True)])
    run_agent_config = RunnableConfig(callbacks=[tracing])
    with drive_agent(agent, []):
        agent.run(input_data={"input": "go"}, config=run_agent_config)

    events = next(run for run in tracing.runs.values() if "tool_hooks" in run.metadata).metadata["tool_hooks"]
    assert events == [{"tool": "search", "phase": "block", "hook": 0, "stop_agent": True, "message": "Nope."}]


class DenyBadQuery(ToolHook):
    """Python hook: conditional block on the tool input."""

    forbidden: str = "DROP TABLE"

    def should_block(self, tool_name, tool_input):
        return self.forbidden in str(tool_input)


class Exploding(ToolHook):
    def apply_before(self, tool_name, tool_input):
        raise RuntimeError("boom-before")

    def apply_after(self, tool_name, tool_result):
        raise RuntimeError("boom-after")

    def should_block(self, tool_name, tool_input):
        return False


def test_python_hook_blocks_conditionally(test_llm):
    agent = build_agent(test_llm, [DenyBadQuery(tools=["search"], forbidden="q0", block_message="Forbidden query.")])
    seen: list[str] = []
    result = run_agent(agent, seen)

    assert result.status == RunnableStatus.SUCCESS
    assert CALLS == []
    assert "Forbidden query." in seen[-1]


def test_python_hook_does_not_block_other_input(test_llm):
    agent = build_agent(test_llm, [DenyBadQuery(forbidden="DROP TABLE")])
    result = run_agent(agent, [])

    assert result.status == RunnableStatus.SUCCESS
    assert len(CALLS) == 1


def test_hook_error_block_policy_is_an_observation(test_llm):
    seen: list[str] = []
    result = run_agent(build_agent(test_llm, [Exploding(on_error="block")]), seen)

    assert result.status == RunnableStatus.SUCCESS  # the run continues; the LLM sees the failure
    assert CALLS == []  # before-hook crashed: call not executed
    assert "boom-before" in seen[-1] and "call not executed" in seen[-1]


def test_hook_error_after_phase_withholds_the_result(test_llm):
    class AfterOnly(Exploding):
        def apply_before(self, tool_name, tool_input):
            return tool_input

    seen: list[str] = []
    result = run_agent(build_agent(test_llm, [AfterOnly()]), seen)

    assert result.status == RunnableStatus.SUCCESS
    assert len(CALLS) == 1  # the tool ran
    assert "tool result withheld" in seen[-1]
    assert "results for q0" not in seen[-1]


def test_hook_error_stop_policy_fails_the_run(test_llm):
    result = run_agent(build_agent(test_llm, [Exploding(on_error="stop")]), [])

    assert result.status == RunnableStatus.FAILURE
    assert CALLS == []


def test_hook_error_skip_policy_keeps_original_data(test_llm):
    result = run_agent(build_agent(test_llm, [Exploding(on_error="skip")]), [])

    assert result.status == RunnableStatus.SUCCESS
    assert CALLS == [{"query": "q0", "mode": "slow"}]  # input untouched, tool ran


def test_hook_error_is_traced(test_llm):
    from dynamiq.callbacks import TracingCallbackHandler

    tracing = TracingCallbackHandler()
    agent = build_agent(test_llm, [Exploding(on_error="skip")])
    with drive_agent(agent, []):
        agent.run(input_data={"input": "go"}, config=RunnableConfig(callbacks=[tracing]))

    events = next(run for run in tracing.runs.values() if "tool_hooks" in run.metadata).metadata["tool_hooks"]
    errors = [e for e in events if e["phase"] == "error"]
    assert {(e["step"], e["policy"]) for e in errors} == {("input", "skip"), ("output", "skip")}
    assert "boom-before" in errors[0]["error"]


def test_invalid_jsonpath_is_rejected_at_construction():
    with pytest.raises(ValueError, match="not a valid JSONPath"):
        ToolHook(input_transformer={"selector": {"query": "$.[bad"}})
    with pytest.raises(ValueError, match="not a valid JSONPath"):
        ToolHook(output_transformer={"path": "$..[["})
    ToolHook(input_transformer={"selector": {"mode": "fast", "q": "$.query"}})  # constants + valid paths are fine


def test_subclass_dumps_type_and_base_stays_typeless():
    assert "type" not in ToolHook(block=True).model_dump()
    dumped = DenyBadQuery(forbidden="x").model_dump()
    assert dumped["type"] == f"{DenyBadQuery.__module__}.DenyBadQuery"
    assert dumped["forbidden"] == "x"


def test_agent_resolves_type_entries_into_subclasses(test_llm):
    dumped = DenyBadQuery(tools=["search"], forbidden="x").model_dump()
    agent = Agent(llm=test_llm, role="r", tools=[SearchTool()], tool_hooks=[dumped, {"tools": ["search"]}])

    assert isinstance(agent.tool_hooks[0], DenyBadQuery) and agent.tool_hooks[0].forbidden == "x"
    assert type(agent.tool_hooks[1]) is ToolHook


def test_agent_rejects_unknown_or_non_hook_type(test_llm):
    with pytest.raises(ValueError, match="could not be imported"):
        Agent(llm=test_llm, role="r", tool_hooks=[{"type": "no.such.module.Hook"}])
    with pytest.raises(ValueError, match="not a ToolHook subclass"):
        Agent(llm=test_llm, role="r", tool_hooks=[{"type": "collections.OrderedDict"}])
