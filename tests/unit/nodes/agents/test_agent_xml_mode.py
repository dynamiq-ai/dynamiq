"""Replies in the XML inference mode."""

import json
from typing import ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from tests.helpers.agent_hooks import TOOL_CALLS, SearchTool, build_agent


def run_xml(replies: list[str]):
    agent = build_agent(tools=[SearchTool()], inference_mode=InferenceMode.XML)
    calls = {"n": 0}

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": replies[min(calls["n"], len(replies) - 1)]}
        calls["n"] += 1
        return result

    TOOL_CALLS.clear()
    with patch.object(agent.llm, "run", side_effect=run):
        return agent.run(input_data={"input": "What does it cost?"})


def test_an_action_runs_even_when_the_model_also_wrote_an_answer():
    # The answer in the same block was written before the tool's result existed.
    both = (
        "<output><thought>I will search.</thought><action>search</action>"
        '<action_input>{"query": "price"}</action_input><answer>It costs $5.</answer></output>'
    )

    result = run_xml([both, "<output><thought>Done.</thought><answer>It costs $7.</answer></output>"])

    assert result.status == RunnableStatus.SUCCESS
    assert [name for name, _ in TOOL_CALLS] == ["search"]
    assert result.output["content"] == "It costs $7."


def test_an_answer_whose_thought_mentions_an_action_tag_is_still_the_answer():
    answer = "<output><thought>No <action> is needed: I know it.</thought><answer>It costs $5.</answer></output>"

    result = run_xml([answer])

    assert TOOL_CALLS == []
    assert result.output["content"] == "It costs $5."


def test_answer_tags_inside_a_tool_input_reach_the_tool_intact():
    query = "Reply only inside <answer></answer> tags."
    action = (
        "<output><thought>Ask the search tool.</thought><action>search</action>"
        f'<action_input>{{"query": "{query}"}}</action_input></output>'
    )

    result = run_xml([action, "<output><thought>Done.</thought><answer>Found it.</answer></output>"])

    assert result.status == RunnableStatus.SUCCESS
    assert [(name, args["query"]) for name, args in TOOL_CALLS] == [("search", query)]


def test_an_answer_written_after_an_action_is_never_streamed():
    agent = build_agent(
        tools=[SearchTool()],
        inference_mode=InferenceMode.XML,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )
    replies = [
        "<output><thought>I will search.</thought><action>search</action>"
        '<action_input>{"query": "price"}</action_input><answer>It costs $5.</answer></output>',
        "<output><thought>Done.</thought><answer>It costs $7.</answer></output>",
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
    assert "$5" not in answer and "$7" in answer


@pytest.mark.parametrize(
    "reply, chunk_size",
    [
        pytest.param(
            "<output><thought>No <action> is needed: I know it.</thought><answer>It costs $5.</answer></output>",
            1,
            id="in-its-thought",
        ),
        # One chunk brings the start of the answer together with the tag its text mentions.
        pytest.param(
            "<output><thought>I know it.</thought><answer>It costs $5. Tools go in an <action> tag.</answer></output>",
            45,
            id="in-its-text",
        ),
    ],
)
def test_an_answer_that_mentions_an_action_tag_is_streamed(reply, chunk_size):
    agent = build_agent(
        tools=[SearchTool()],
        inference_mode=InferenceMode.XML,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )
    streamed: list[tuple[str, object]] = []

    def run(**kwargs):
        serialized = {"group": "llms", "id": agent.llm.id}
        for callback in kwargs["config"].callbacks:
            if hasattr(callback, "accumulated_content"):
                for start in range(0, len(reply), chunk_size):
                    delta = {"content": reply[start : start + chunk_size]}
                    callback.on_node_execute_stream(serialized, chunk={"choices": [{"delta": delta}]})
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

    assert result.output["content"].startswith("It costs $5.")
    assert "$5" in "".join(str(content) for step, content in streamed if step == "answer")


def test_a_reply_without_a_thought_is_still_read():
    # Required <thought> was the most frequent agent error in production: replies that act or answer without one.
    action = '<output><action>search</action><action_input>{"query": "price"}</action_input></output>'

    result = run_xml([action, "<output><answer>It costs $7.</answer></output>"])

    assert result.status == RunnableStatus.SUCCESS
    assert [name for name, _ in TOOL_CALLS] == ["search"]
    assert result.output["content"] == "It costs $7."


def test_an_action_without_action_input_runs_with_no_arguments():
    action = "<output><thought>List the results.</thought><action>search</action></output>"

    result = run_xml([action, "<output><thought>Done.</thought><answer>ok</answer></output>"])

    assert result.status == RunnableStatus.SUCCESS
    assert [name for name, _ in TOOL_CALLS] == ["search"]


class TicketSearchInput(BaseModel):
    query: str = Field(..., description="What to look for.")
    brief: str = Field(default="", description="Why the search is made.")


class TicketSearch(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "search-tickets"
    description: str = "Searches support tickets."
    input_schema: ClassVar[type[TicketSearchInput]] = TicketSearchInput

    def execute(self, input_data: TicketSearchInput, config: RunnableConfig | None = None, **kwargs) -> dict:
        TOOL_CALLS.append((self.name, input_data.model_dump()))
        return {"content": f"3 tickets about {input_data.query}"}


@pytest.mark.parametrize(
    "batch",
    [
        # Seen in production: the tool's one argument as a string, and another argument beside "name".
        [{"name": "search-tickets", "input": "mailbox limit", "brief": "Find limit reports"}],
        [{"name": "search-tickets", "arguments": {"query": "mailbox limit", "brief": "Find limit reports"}}],
        {"tools": [{"name": "search-tickets", "input": '{"query": "mailbox limit", "brief": "Find limit reports"}'}]},
    ],
    ids=["string-input-and-flattened-argument", "arguments-key-in-a-bare-list", "json-string-input"],
)
def test_a_parallel_batch_in_a_common_variant_shape_runs(batch):
    agent = build_agent(tools=[TicketSearch()], inference_mode=InferenceMode.XML, parallel_tool_calls_enabled=True)
    replies = [
        f"<output><thought>Search.</thought><action>run-parallel</action>"
        f"<action_input>{json.dumps(batch)}</action_input></output>",
        "<output><thought>Done.</thought><answer>Found 3.</answer></output>",
    ]
    calls = {"n": 0}

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": replies[min(calls["n"], len(replies) - 1)]}
        calls["n"] += 1
        return result

    TOOL_CALLS.clear()
    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "How many tickets mention the mailbox limit?"})

    assert result.status == RunnableStatus.SUCCESS
    assert TOOL_CALLS == [("search-tickets", {"query": "mailbox limit", "brief": "Find limit reports"})]
