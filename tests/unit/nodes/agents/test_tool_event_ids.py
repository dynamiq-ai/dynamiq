"""Tool events carry the tool's node id, so a client replaying history can address the tool itself."""

from typing import ClassVar
from unittest.mock import patch

from pydantic import BaseModel

from dynamiq import connections
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig


class _Input(BaseModel):
    i: int = 0


class ProbeTool(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "probe"
    description: str = "returns its input"
    input_schema: ClassVar[type[BaseModel]] = _Input

    def execute(self, input_data, config=None, **kwargs):
        return {"content": f"done-{input_data.i}"}


def _agent():
    llm = llms.OpenAI(
        id="llm",
        model="gpt-4o-mini",
        connection=connections.OpenAI(api_key="not-used"),
        is_postponed_component_init=True,
    )
    return Agent(
        id="agent",
        name="agent",
        llm=llm,
        tools=[ProbeTool(id="probe-id")],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )


def _streamed_events(tools_data):
    agent = _agent()
    with patch.object(Agent, "_stream_agent_event") as stream:
        agent._execute_tools(tools_data, "t", 1, RunnableConfig())
    return [(call[0][0], call[0][1]) for call in stream.call_args_list]


def test_single_tool_events_carry_the_tool_id():
    events = _streamed_events([{"name": "probe", "input": {"i": 1}, "thought": "t"}])

    reasoning = [data for data, kind in events if kind == "reasoning"]
    assert reasoning, "the tool call must stream a reasoning event"
    assert all(data.tool.id == "probe-id" for data in reasoning)


def test_parallel_batch_entries_carry_each_tool_id():
    events = _streamed_events(
        [
            {"name": "probe", "input": {"i": 1}, "thought": "t"},
            {"name": "probe", "input": {"i": 2}, "thought": "t"},
        ]
    )

    reasoning = next(d for d, kind in events if kind == "reasoning" and d.action == PARALLEL_TOOL_NAME)
    completion = next(d for d, kind in events if kind == "tool" and d.name == PARALLEL_TOOL_NAME)

    assert [entry["tool"]["id"] for entry in reasoning.action_input] == ["probe-id", "probe-id"]
    assert [entry["tool"]["id"] for entry in completion.result] == ["probe-id", "probe-id"]
    assert reasoning.tool.id is None, "the run-parallel wrapper is not a node and has no id"
