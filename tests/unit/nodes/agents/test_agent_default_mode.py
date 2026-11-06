"""Replies in the DEFAULT (Thought/Action/Answer text) inference mode."""

from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from tests.helpers.agent_hooks import TOOL_CALLS, SearchTool, build_agent


def run_default(replies: list[str]):
    agent = build_agent(tools=[SearchTool()], inference_mode=InferenceMode.DEFAULT)
    requests: list[list] = []

    def run(**kwargs):
        requests.append(list(kwargs["prompt"].messages))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": replies[min(len(requests) - 1, len(replies) - 1)]}
        return result

    TOOL_CALLS.clear()
    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "Draft the FAQ."})
    return result, requests


def test_an_action_whose_input_contains_answer_is_run_as_an_action():
    action = 'Thought: I will look it up.\nAction: search\nAction Input: {"query": "Q: refunds?\\nAnswer: 30 days"}'

    result, _ = run_default([action, "Thought: Done.\nAnswer: Refunds take 30 days."])

    assert result.status == RunnableStatus.SUCCESS
    assert TOOL_CALLS == [("search", {"query": "Q: refunds?\nAnswer: 30 days", "mode": "slow"})]
    assert result.output["content"] == "Refunds take 30 days."


def test_an_action_line_with_no_input_before_the_answer_is_still_the_answer():
    result, _ = run_default(["Thought: I know this.\nAction: None\nAnswer: 42"])

    assert TOOL_CALLS == []
    assert result.output["content"] == "42"


def test_answer_text_inside_an_action_input_is_never_streamed_as_the_answer():
    agent = build_agent(
        tools=[SearchTool()],
        inference_mode=InferenceMode.DEFAULT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )
    replies = [
        'Thought: I will look it up.\nAction: search\nAction Input: {"query": "Q: refunds?\\nAnswer: 30 days"}',
        "Thought: Done.\nAnswer: Refunds take 30 days.",
    ]
    streamed: list[tuple[str, object]] = []
    calls = {"n": 0}

    def run(**kwargs):
        reply = replies[min(calls["n"], len(replies) - 1)]
        calls["n"] += 1
        serialized = {"group": "llms", "id": agent.llm.id}
        for callback in kwargs["config"].callbacks:
            if hasattr(callback, "accumulated_content"):
                for char in reply:
                    callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": {"content": char}}]})
                callback.on_node_execute_end(serialized, {})
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": reply}
        return result

    def record(self, content, source, step, config=None, **kwargs):
        if self is agent:
            streamed.append((step, content))
        return content

    with patch.object(agent.llm, "run", side_effect=run), patch.object(Agent, "stream_content", record):
        result = agent.run(input_data={"input": "Draft the FAQ."})

    assert result.output["content"] == "Refunds take 30 days."
    assert "".join(str(content) for step, content in streamed if step == "answer").strip() == "Refunds take 30 days."


@pytest.mark.parametrize(
    "action",
    [
        '**Thought:** Search.\n**Action:** search\n**Action Input:** {"query": "price"}',
        '### Thought\nSearch.\n### Action\nsearch\n### Action Input\n{"query": "price"}',
        'thought: Search.\naction: search\naction input: {"query": "price"}',
    ],
    ids=["bold", "headings", "lower-case"],
)
def test_labels_written_in_markdown_or_lower_case_are_read(action):
    result, _ = run_default([action, "**Thought:** Done.\n**Answer:** It costs $7."])

    assert TOOL_CALLS == [("search", {"query": "price", "mode": "slow"})]
    assert result.output["content"] == "It costs $7."


def test_an_answer_without_a_thought_is_still_the_answer():
    result, _ = run_default(["Answer: Paris"])

    assert result.output["content"] == "Paris"


def test_an_observation_the_model_wrote_itself_is_dropped():
    # Reasoning models ignore stop sequences and go on to invent the result and an answer.
    action = (
        'Thought: I will search.\nAction: search\nAction Input: {"query": "price"}\n'
        "Observation: the price is $5\nThought: Done.\nAnswer: $5"
    )

    result, requests = run_default([action, "Thought: Done.\nAnswer: It costs $7."])

    assert result.output["content"] == "It costs $7."
    assistant = [m.content for m in requests[1] if m.role == MessageRole.ASSISTANT]
    assert assistant and "the price is $5" not in assistant[-1]


def test_action_input_with_a_raw_newline_in_a_string_is_parsed():
    action = 'Thought: Search.\nAction: search\nAction Input: {"query": "line one\nline two"}'

    result, _ = run_default([action, "Thought: Done.\nAnswer: ok"])

    assert TOOL_CALLS == [("search", {"query": "line one\nline two", "mode": "slow"})]


def test_an_observation_the_model_wrote_itself_is_never_streamed():
    agent = build_agent(
        tools=[SearchTool()],
        inference_mode=InferenceMode.DEFAULT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )
    replies = [
        'Thought: I will search.\nAction: search\nAction Input: {"query": "price"}\n'
        "Observation: the price is $5\nThought: Done.\nAnswer: $5",
        "Thought: Done.\nAnswer: It costs $7.",
    ]
    streamed: list[tuple[str, object]] = []
    calls = {"n": 0}

    def run(**kwargs):
        reply = replies[min(calls["n"], len(replies) - 1)]
        calls["n"] += 1
        serialized = {"group": "llms", "id": agent.llm.id}
        for callback in kwargs["config"].callbacks:
            if hasattr(callback, "accumulated_content"):
                for char in reply:
                    callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": {"content": char}}]})
                callback.on_node_execute_end(serialized, {})
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": reply}
        return result

    def record(self, content, source, step, config=None, **kwargs):
        if self is agent:
            streamed.append((step, content))
        return content

    with patch.object(agent.llm, "run", side_effect=run), patch.object(Agent, "stream_content", record):
        result = agent.run(input_data={"input": "What does it cost?"})

    assert result.output["content"] == "It costs $7."
    answer = "".join(str(content) for step, content in streamed if step == "answer")
    assert "$5" not in answer
    assert "$7" in answer
