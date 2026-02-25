import os
import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.operators import Map
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode


def _openai_llm(model: str = "gpt-4o-mini") -> OpenAI:
    """Helper returning a real OpenAI LLM using env credentials."""
    return OpenAI(model=model, connection=connections.OpenAI())


def _child_researcher(llm: OpenAI) -> Agent:
    role = (
        "You are a Researcher sub-agent.\n"
        "- Always expect a single string input under the 'input' key.\n"
        "- Do NOT expect or require other keys like 'query' or 'metadata'.\n"
        "- Provide a concise factual summary based on the provided input.\n"
    )
    return Agent(
        name="Researcher",
        description=(
            "Sub-agent that summarizes research for a given topic. "
            "IMPORTANT: Accepts only {'input': '<topic>'} when invoked as a tool."
        ),
        role=role,
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=False,
        max_loops=6,
    )


def _child_writer(llm: OpenAI) -> Agent:
    role = (
        "You are a Writer sub-agent.\n"
        "- Always expect a single string input under the 'input' key.\n"
        "- Do NOT expect or require other keys like 'query' or 'metadata'.\n"
        "- Based on the 'input' topic, produce a clean, concise brief.\n"
    )
    return Agent(
        name="Writer",
        description=(
            "Sub-agent that drafts a concise brief based on a topic. "
            "IMPORTANT: Accepts only {'input': '<topic>'} when invoked as a tool."
        ),
        role=role,
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=False,
        max_loops=6,
    )


@pytest.mark.integration
def test_manager_with_subagents_parallel_calls():
    """Manager agent uses two sub-agents as tools with function calling + parallel enabled.

    Asserts successful run and traces/streams indicate agent-tool activity.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")

    llm = _openai_llm()

    shared_memory = Memory(backend=InMemory())

    researcher = _child_researcher(llm)
    writer = _child_writer(llm)
    researcher.memory = shared_memory
    writer.memory = shared_memory

    manager_role = (
        "You are a Manager agent coordinating sub-agents.\n"
        "- When invoking an agent tool, ALWAYS pass input as {'input': '<task>'}.\n"
        "- Do not include extraneous keys like 'query' or 'metadata' in tool inputs.\n"
        "- Prefer parallel calls when multiple subtasks are obvious.\n"
    )
    manager = Agent(
        name="Manager",
        description=("Manager that delegates to Researcher and Writer. Tools expect {'input': '<topic>'}."),
        role=manager_role,
        llm=llm,
        tools=[researcher, writer],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=True,
        memory=shared_memory,
        max_loops=10,
    )

    stream_cfg = StreamingConfig(enabled=True, mode=StreamingMode.ALL)
    manager.streaming = stream_cfg
    researcher.streaming = stream_cfg
    writer.streaming = stream_cfg

    wf = Workflow(flow=Flow(nodes=[manager]))

    streaming = StreamingIteratorCallbackHandler()
    tracing = TracingCallbackHandler()

    result = wf.run(
        input_data={
            "input": "Research facts about dolphins and about whales and write"
            " a short brief using both available agents.",
        },
        config=RunnableConfig(callbacks=[streaming, tracing]),
    )

    assert result.status == RunnableStatus.SUCCESS

    out = str(result.output.get(manager.id, {}).get("output", {}).get("content", ""))
    assert len(out.strip()) > 0

    received = 0
    sources = set()
    for event in streaming:
        received += 1
        src = getattr(event, "source", None)
        if src and getattr(src, "name", None):
            sources.add(src.name)
    assert received > 0

    if not any(name in sources for name in ("Researcher", "Writer", "Manager")):
        node_runs = [run for run in tracing.runs.values() if getattr(run, "metadata", None)]
        node_names = [getattr(run, "name", "") for run in node_runs]
        assert any(n in node_names for n in ("Researcher", "Writer", "Manager"))


@pytest.mark.integration
def test_agents_as_tools_with_map_parallel_streaming_tracing_memory():
    """End-to-end: agents-as-tools under function calling
    with Map parallelism, streaming, tracing, memory continuity."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")

    memory = Memory(backend=InMemory())
    llm = _openai_llm("gpt-4o-mini")

    researcher = _child_researcher(llm)
    writer = _child_writer(llm)
    researcher.memory = memory
    writer.memory = memory

    manager_role = (
        "You are a Manager agent coordinating sub-agents.\n"
        "- When invoking an agent tool, ALWAYS pass input as {'input': '<task>'}.\n"
        "- Do not include extraneous keys like 'query' or 'metadata' in tool inputs.\n"
        "- Prefer parallel calls when multiple subtasks are obvious.\n"
    )
    manager = Agent(
        name="Manager",
        description=("Manager that delegates to Researcher and Writer. Tools expect {'input': '<topic>'}."),
        role=manager_role,
        llm=llm,
        tools=[researcher, writer],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        parallel_tool_calls_enabled=True,
        memory=memory,
        max_loops=10,
    )

    stream_cfg = StreamingConfig(enabled=True, mode=StreamingMode.ALL)
    for a in (manager, researcher, writer):
        a.streaming = stream_cfg

    map_node = Map(node=manager, max_workers=2)
    wf = Workflow(flow=Flow(nodes=[map_node]))

    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())

    inputs = {
        "input": [
            {"input": "find facts about cats and write a brief"},
            {"input": "find facts about dogs and write a brief"},
        ]
    }

    streaming = StreamingIteratorCallbackHandler()
    tracing = TracingCallbackHandler()

    result = wf.run(input_data=inputs, config=RunnableConfig(callbacks=[streaming, tracing]))
    assert result.status == RunnableStatus.SUCCESS

    node_runs = [run for run in tracing.runs.values() if getattr(run, "metadata", None)]
    node_meta = [getattr(run, "metadata", {}).get("node", {}) for run in node_runs]
    names = [m.get("name") for m in node_meta if isinstance(m, dict)]
    assert "Map" in names
    assert "Manager" in names

    assert any(n in ("Researcher", "Writer") for n in names)

    received = 0
    for _ in streaming:
        received += 1
    assert received > 0

    followup_inputs = {
        "input": [
            {"input": "continue with more details on cats", "user_id": user_id, "session_id": session_id},
            {"input": "continue with more details on dogs", "user_id": user_id, "session_id": session_id},
        ]
    }

    followup_result = wf.run(input_data=followup_inputs, config=RunnableConfig(callbacks=[streaming, tracing]))
    assert followup_result.status == RunnableStatus.SUCCESS

    manager_runs_with_ids = [
        run
        for run in tracing.runs.values()
        if getattr(run, "metadata", {}).get("node", {}).get("name") == "Manager"
        and isinstance(getattr(run, "input", None), dict)
        and run.input.get("user_id") in (None, user_id)
        and run.input.get("session_id") in (None, session_id)
    ]
    assert len(manager_runs_with_ids) >= 1
