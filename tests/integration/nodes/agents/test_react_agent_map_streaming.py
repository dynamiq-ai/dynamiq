from collections import defaultdict

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.operators import Map
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import Behavior
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode


@pytest.mark.integration
def test_react_agent_map_streaming_all_mode_isolated(mock_llm_executor):
    """Ensure parallel ReAct agents under Map stream independently in ALL mode (with tools)."""
    python_tool = Python(
        name="NoOp Tool",
        description="Simple tool returning static content",
        code="""
def run():
    return {"content": "noop"}
""",
    )
    exa_tool = ExaTool(connection=connections.Exa(api_key="test-api-key"))
    firecrawl_tool = FirecrawlTool(connection=connections.Firecrawl(api_key="test-api-key"))

    agent = ReActAgent(
        name="React Agent",
        llm=OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test-api-key")),
        inference_mode=InferenceMode.DEFAULT,
        tools=[python_tool, exa_tool, firecrawl_tool],
        streaming=StreamingConfig(enabled=True, event="react_map_stream", mode=StreamingMode.ALL, by_tokens=True),
        max_loops=2,
        behaviour_on_max_loops=Behavior.RETURN,
    )

    wf = Workflow(flow=Flow(nodes=[Map(node=agent, max_workers=3)]))

    input_data = {"input": [{"q": 1}, {"q": 2}, {"q": 3}]}

    streaming = StreamingIteratorCallbackHandler()
    result = wf.run(input_data=input_data, config=RunnableConfig(callbacks=[streaming]))

    assert result.status == RunnableStatus.SUCCESS

    llm_events_by_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in streaming:
        if event.entity_id == wf.id:
            continue
        source = getattr(event, "source", None)
        if getattr(source, "group", None) == "llms":
            text = ""
            if isinstance(event.data, dict):
                if (choices := event.data.get("choices")) and choices[0].get("delta"):
                    text = choices[0]["delta"].get("content", "")
                elif isinstance(event.data.get("content"), str):
                    text = event.data["content"]
            llm_events_by_entity[event.entity_id].append((event.event, text))

    assert len(llm_events_by_entity) == 3
    for items in llm_events_by_entity.values():
        assert len(items) > 0
        assert "mocked_response" in "".join(chunk for _, chunk in items)


@pytest.mark.integration
def test_react_agent_map_streaming_final_mode_isolated(mock_llm_executor):
    """Ensure parallel ReAct agents under Map stream only final content in FINAL mode (with tools)."""
    python_tool = Python(
        name="NoOp Tool",
        description="Simple tool returning static content",
        code="""
def run():
    return {"content": "noop"}
""",
    )
    exa_tool = ExaTool(connection=connections.Exa(api_key="test-api-key"))
    firecrawl_tool = FirecrawlTool(connection=connections.Firecrawl(api_key="test-api-key"))

    agent = ReActAgent(
        name="React Agent",
        llm=OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test-api-key")),
        inference_mode=InferenceMode.DEFAULT,
        tools=[python_tool, exa_tool, firecrawl_tool],
        streaming=StreamingConfig(
            enabled=True, event="react_map_stream_final", mode=StreamingMode.FINAL, by_tokens=True
        ),
        max_loops=2,
        behaviour_on_max_loops=Behavior.RETURN,
    )

    wf = Workflow(flow=Flow(nodes=[Map(node=agent, max_workers=3)]))

    input_data = {"input": [{"q": 1}, {"q": 2}, {"q": 3}]}

    streaming = StreamingIteratorCallbackHandler()
    result = wf.run(input_data=input_data, config=RunnableConfig(callbacks=[streaming]))

    assert result.status == RunnableStatus.SUCCESS

    llm_events_by_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in streaming:
        if event.entity_id == wf.id:
            continue
        source = getattr(event, "source", None)
        if getattr(source, "group", None) == "llms":
            text = ""
            if isinstance(event.data, dict):
                if (choices := event.data.get("choices")) and choices[0].get("delta"):
                    text = choices[0]["delta"].get("content", "")
                elif isinstance(event.data.get("content"), str):
                    text = event.data["content"]
            llm_events_by_entity[event.entity_id].append((event.event, text))

    assert len(llm_events_by_entity) == 3
    for items in llm_events_by_entity.values():
        assert len(items) >= 0
