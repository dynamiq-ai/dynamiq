"""Integration tests for parallel tool calling (FUNCTION_CALLING native and XML run_parallel)."""

import os

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from tests.integration_with_creds.agents.streaming_assertions import assert_streaming_events, collect_streaming_events


def _make_tool(name: str, code: str) -> Python:
    return Python(
        name=name,
        description=f"Returns a fact about {name.lower()}.",
        code=code,
        is_parallel_execution_allowed=True,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "inference_mode", [InferenceMode.XML, InferenceMode.STRUCTURED_OUTPUT, InferenceMode.FUNCTION_CALLING]
)
def test_parallel_tool_calling(inference_mode: InferenceMode):
    """Agent with parallel_tool_calls_enabled calls two tools in parallel for both inference modes."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    llm = OpenAI(model="gpt-5.4-mini", connection=connections.OpenAI())

    tool_a = _make_tool("CatFacts", 'def run(input_data):\n    return "Cats sleep 12-16 hours per day."')
    tool_b = _make_tool(
        "DogFacts", 'def run(input_data):\n    return "Dogs have a sense of smell 40x better than humans."'
    )

    agent = Agent(
        name="parallel_tools_agent",
        role="Researcher agent.",
        llm=llm,
        tools=[tool_a, tool_b],
        parallel_tool_calls_enabled=True,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
        inference_mode=inference_mode,
    )

    tracing = TracingCallbackHandler()
    streaming_handler = StreamingIteratorCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    result = wf.run(
        input_data={"input": "Call a cat tool and dog tool in parallel"},
        config=RunnableConfig(callbacks=[tracing, streaming_handler]),
    )

    assert result.status == RunnableStatus.SUCCESS

    content = str(result.output.get(agent.id, {}).get("output", {}).get("content", ""))
    assert len(content.strip()) > 0

    llm_runs = [
        run for run in tracing.runs.values() if getattr(run, "metadata", {}).get("node", {}).get("name") == "OpenAI"
    ]
    assert len(llm_runs) <= 2, f"Expected at most 2 LLM loops (parallel tools + final answer), got {len(llm_runs)}"

    ordered_events = collect_streaming_events(streaming_handler, agent.id)
    assert_streaming_events(ordered_events, inference_mode)
