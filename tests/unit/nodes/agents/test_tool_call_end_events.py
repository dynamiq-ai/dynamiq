"""Every tool call a client is told about is also told how it ended, including when its failure fails the run."""

import json
from typing import Any, ClassVar, Literal
from unittest.mock import patch

import pytest

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.hooks import BlockAs, ToolPolicyHook
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import check_cancellation
from dynamiq.types.streaming import InputStreamingTimeoutError, StreamingConfig, StreamingMode
from tests.helpers.agent_hooks import (
    ALL_MODES,
    EmailTool,
    ExplodingTool,
    SearchSchema,
    SearchTool,
    build_agent,
    final_step,
    parallel_step,
    scripted,
    scripted_streaming,
    tool_step,
)

STREAM_ALL = StreamingConfig(enabled=True, mode=StreamingMode.ALL)
BATCH = "run-parallel"


class Tool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    description: str = "A tool."
    input_schema: ClassVar[type[SearchSchema]] = SearchSchema


class CrashingTool(Tool):
    """Fails with an error its tool did not mark as one the model can recover from."""

    name: str = "crash"

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        raise RuntimeError("disk full at /data/customers/4111111111111111")


class CancelingTool(Tool):

    name: str = "cancel"

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        config.cancellation.token.cancel()
        check_cancellation(config)
        return {"content": "unreached"}


class PausingTool(Tool):
    """Waits for a person longer than its stream allows, which pauses the run."""

    name: str = "pause"

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        raise InputStreamingTimeoutError(node_id=self.id, timeout=1)


class QuietlyCancelingTool(Tool):
    """Ends normally while the run is canceled."""

    name: str = "quiet-cancel"

    def execute(self, input_data: SearchSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        config.cancellation.token.cancel()
        return {"content": "done"}


def one_at_a_time(*tools):
    """A parallel batch whose calls run in the order given, so that what follows a failure is known."""
    for tool in tools:
        tool.is_parallel_execution_allowed = True
    return build_agent(
        tools=list(tools), streaming=STREAM_ALL, parallel_tool_calls_enabled=True, max_parallel_tool_calls=1
    )


def run(agent, mode, steps):
    with scripted_streaming(agent, mode, steps) as (_, streamed):
        result = agent.run(input_data={"input": "go"})
    return result, streamed


def calls(streamed) -> list[tuple[str, str, dict]]:
    """Each tool call event in the order it was streamed: ("invoked" or "completed", tool name, content)."""
    events = []
    for step, content in streamed:
        if step == "reasoning" and isinstance(content, dict) and content.get("tool"):
            events.append(("invoked", content["tool"]["name"], content))
        elif step == "tool":
            events.append(("completed", content["name"], content))
    return events


def ends(streamed) -> dict[str, list[str]]:
    """How each invoked call ended, by tool name: the status of every end streamed for it, none if still open."""
    events = calls(streamed)
    return {
        name: [
            end["status"]
            for kind, _, end in events
            if kind == "completed" and end["tool_run_id"] == call["tool_run_id"]
        ]
        for kind, name, call in events
        if kind == "invoked"
    }


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_call_whose_failure_fails_the_run_is_streamed_as_failed(mode):
    agent = build_agent(tools=[CrashingTool()], streaming=STREAM_ALL)

    result, streamed = run(agent, mode, [tool_step("crash", query="x"), final_step("unreached")])

    assert result.status == RunnableStatus.FAILURE
    assert "disk full" in result.error.message
    assert ends(streamed) == {"crash": ["failure"]}


def test_the_failed_call_does_not_stream_the_error_text_no_hook_has_seen():
    agent = build_agent(tools=[CrashingTool()], streaming=STREAM_ALL)

    _, streamed = run(agent, InferenceMode.FUNCTION_CALLING, [tool_step("crash", query="x"), final_step("unreached")])

    assert "4111111111111111" not in json.dumps([content for _, _, content in calls(streamed)], default=str)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_failure_the_model_can_recover_from_is_still_streamed_once(mode):
    agent = build_agent(tools=[ExplodingTool()], streaming=STREAM_ALL)

    result, streamed = run(agent, mode, [tool_step("http", query="x"), final_step("ok")])

    assert result.status == RunnableStatus.SUCCESS
    assert ends(streamed) == {"http": ["failure"]}


def test_a_batch_whose_call_fails_the_run_ends_with_every_call_accounted_for():
    agent = one_at_a_time(CrashingTool(), SearchTool())
    steps = [parallel_step(("crash", {"query": "a"}), ("search", {"query": "b"})), final_step("unreached")]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert "disk full" in result.error.message
    assert ends(streamed) == {BATCH: ["failure"], "crash": ["failure"], "search": ["success"]}
    kind, name, batch_end = calls(streamed)[-1]
    assert (kind, name) == ("completed", BATCH)
    assert {entry["name"]: entry["status"] for entry in batch_end["result"]} == {
        "crash": "failure",
        "search": "success",
    }


def test_a_batch_that_fails_before_its_sequential_calls_reports_them_as_not_run():
    agent = build_agent(
        tools=[CrashingTool(is_parallel_execution_allowed=True), SearchTool(is_parallel_execution_allowed=False)],
        streaming=STREAM_ALL,
        parallel_tool_calls_enabled=True,
    )
    steps = [parallel_step(("crash", {"query": "a"}), ("search", {"query": "b"})), final_step("unreached")]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert ends(streamed) == {BATCH: ["failure"], "crash": ["failure"]}
    batch_end = calls(streamed)[-1][2]
    assert {entry["name"]: entry["status"] for entry in batch_end["result"]} == {"crash": "failure", "search": "skip"}


def test_a_batch_run_in_sequence_reports_what_ran_what_failed_and_what_never_started():
    agent = build_agent(tools=[SearchTool(), CrashingTool(), EmailTool()], streaming=STREAM_ALL)
    steps = [
        parallel_step(("search", {"query": "a"}), ("crash", {"query": "b"}), ("send-email", {"to": "c"})),
        final_step("unreached"),
    ]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert ends(streamed) == {BATCH: ["failure"], "search": ["success"], "crash": ["failure"]}
    batch_end = calls(streamed)[-1][2]
    assert {entry["name"]: entry["status"] for entry in batch_end["result"]} == {
        "search": "success",
        "crash": "failure",
        "send-email": "skip",
    }


def test_a_batch_with_a_failed_call_and_a_paused_one_is_left_open_and_fails_with_the_first_error():
    agent = one_at_a_time(CrashingTool(), PausingTool())
    steps = [parallel_step(("crash", {"query": "a"}), ("pause", {"query": "b"})), final_step("unreached")]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert "disk full" in result.error.message
    assert ends(streamed) == {BATCH: [], "crash": ["failure"], "pause": []}


def test_a_run_canceled_after_a_call_failed_it_still_ends_failed():
    agent = one_at_a_time(CrashingTool(), QuietlyCancelingTool())
    steps = [parallel_step(("crash", {"query": "a"}), ("quiet-cancel", {"query": "b"})), final_step("unreached")]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert "disk full" in result.error.message
    assert ends(streamed) == {BATCH: ["failure"], "crash": ["failure"], "quiet-cancel": ["success"]}


def test_a_call_that_ended_is_not_ended_again_when_streaming_its_result_fails():
    agent = build_agent(tools=[SearchTool()], streaming=STREAM_ALL)
    steps = [tool_step("search", query="a"), final_step("unreached")]

    with scripted_streaming(agent, InferenceMode.FUNCTION_CALLING, steps) as (_, streamed):
        record = Agent.stream_content

        def record_then_fail_on_the_result(self, content, source, step, config=None, **kwargs):
            record(self, content, source, step, config, **kwargs)
            if step == "tool":
                raise RuntimeError("stream closed")

        with patch.object(Agent, "stream_content", record_then_fail_on_the_result):
            result = agent.run(input_data={"input": "go"})

    assert result.status == RunnableStatus.FAILURE
    assert ends(streamed) == {"search": ["success"]}


def test_a_call_that_ended_is_not_ended_again_when_streaming_its_delegated_answer_fails():
    child = build_agent(tools=[], name="Researcher", id="child")
    parent = build_agent(
        tools=[SubAgentTool(agent=child, name="Researcher", description="researches")],
        delegation_allowed=True,
        streaming=STREAM_ALL,
    )
    steps = [tool_step("Researcher", input="q", delegate_final=True), final_step("unreached")]

    with scripted(child, InferenceMode.FUNCTION_CALLING, [final_step("found it")]):
        with scripted_streaming(parent, InferenceMode.FUNCTION_CALLING, steps) as (_, streamed):
            record = Agent.stream_content

            def record_then_fail_on_the_answer(self, content, source, step, config=None, **kwargs):
                record(self, content, source, step, config, **kwargs)
                if self is parent and step == "answer":
                    raise RuntimeError("stream closed")

            with patch.object(Agent, "stream_content", record_then_fail_on_the_answer):
                result = parent.run(input_data={"input": "go"})

    assert result.status == RunnableStatus.FAILURE
    assert ends(streamed) == {"Researcher": ["success"]}


@pytest.mark.parametrize(
    ("tool", "status"),
    [(CancelingTool(), RunnableStatus.CANCELED), (PausingTool(), RunnableStatus.FAILURE)],
    ids=["canceled", "waiting for a person"],
)
def test_a_call_that_has_not_ended_is_left_open(tool, status):
    agent = build_agent(tools=[tool], streaming=STREAM_ALL)

    result, streamed = run(
        agent, InferenceMode.FUNCTION_CALLING, [tool_step(tool.name, query="x"), final_step("unreached")]
    )

    assert result.status == status
    assert ends(streamed) == {tool.name: []}


def test_a_batch_canceled_between_its_calls_is_left_open():
    agent = build_agent(tools=[CancelingTool(), SearchTool()], streaming=STREAM_ALL)
    steps = [parallel_step(("cancel", {"query": "a"}), ("search", {"query": "b"})), final_step("unreached")]

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.CANCELED
    assert ends(streamed) == {BATCH: [], "cancel": []}


def test_a_call_on_which_a_hook_ends_the_run_with_an_answer_is_not_streamed_as_failed():
    hook = ToolPolicyHook(tools=["search"], deny=True, on_violation=BlockAs.ANSWER, message="I cannot search.")
    agent = build_agent([hook], tools=[SearchTool()], streaming=STREAM_ALL)

    result, streamed = run(agent, InferenceMode.FUNCTION_CALLING, [tool_step("search", query="x"), final_step("no")])

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == "I cannot search."
    assert ends(streamed) == {"search": []}
