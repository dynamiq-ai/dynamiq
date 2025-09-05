from collections import defaultdict

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.callbacks.tracing import RunType
from dynamiq.flows import Flow
from dynamiq.nodes import Behavior
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.operators import Map
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode


@pytest.fixture
def mock_llm_response_text():
    return (
        "Thought: We'll try a noop, then query and scrape.\n"
        "Action: NoOp Tool\n"
        "Action Input: {}\n"
        "Action: Exa Search Tool\n"
        'Action Input: {"query": "test", "limit": 1}\n'
        "Action: Firecrawl Tool\n"
        'Action Input: {"url": "https://example.com"}'
    )


@pytest.fixture
def mock_tools_http(mocker):
    class DummyResponse:
        def __init__(self, data: dict):
            self._data = data

        def raise_for_status(self):
            return None

        def json(self):
            return self._data

    def fake_request(method, url, json=None, headers=None):
        if isinstance(url, str) and url.endswith("search"):
            return DummyResponse(
                {
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "publishedDate": "2024-01-01",
                            "author": "bot",
                            "score": 1.0,
                        }
                    ]
                }
            )
        if isinstance(url, str) and url.endswith("scrape"):
            return DummyResponse({"success": True, "data": {"content": "scraped"}})
        return DummyResponse({})

    m = mocker.patch("requests.request", side_effect=fake_request)
    return m


def test_map_react_agent_parallel_streams_isolated(mock_llm_executor, mock_tools_http):
    python_tool = Python(
        name="NoOp Tool",
        description="Simple tool returning static content",
        code="""
def run(input_data):
    return {"content": "noop"}
""",
    )

    exa_tool = ExaTool(connection=connections.Exa(api_key="test-api-key"))
    firecrawl_tool = FirecrawlTool(connection=connections.Firecrawl(api_key="test-api-key"))

    agent = ReActAgent(
        name="React Agent",
        llm=OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test-api-key")),
        inference_mode=InferenceMode.DEFAULT,
        parallel_tool_calls_enabled=True,
        tools=[python_tool, exa_tool, firecrawl_tool],
        streaming=StreamingConfig(enabled=True, event="map_react_stream", mode=StreamingMode.ALL, by_tokens=True),
        max_loops=20,
        behaviour_on_max_loops=Behavior.RETURN,
    )

    map_node = Map(node=agent, max_workers=3)

    wf = Workflow(flow=Flow(nodes=[map_node]))

    inputs = {"input": [{"q": 1}, {"q": 2}, {"q": 3}]}

    streaming = StreamingIteratorCallbackHandler()
    tracing = TracingCallbackHandler()
    result = wf.run(input_data=inputs, config=RunnableConfig(callbacks=[streaming, tracing]))

    assert result.status == RunnableStatus.SUCCESS

    node_runs = [run for run in tracing.runs.values() if run.type == RunType.NODE]
    assert len(node_runs) > 1

    llm_events_by_entity: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for event in streaming:
        if event.entity_id == wf.id:
            continue
        source = getattr(event, "source", None)
        if getattr(source, "group", None) == "agents":
            text = ""
            if isinstance(event.data, dict):
                if (choices := event.data.get("choices")) and choices[0].get("delta"):
                    text = choices[0]["delta"].get("content", "")
                elif isinstance(event.data.get("content"), str):
                    text = event.data["content"]
            llm_events_by_entity[event.entity_id].append((event.event, text))

    assert len(llm_events_by_entity.keys()) == 3

    for entity_id, items in llm_events_by_entity.items():
        assert len(items) > 0
        joined = "".join(str(chunk) for _, chunk in items)
        assert ("Action:" in joined) or ("Thought:" in joined) or ("mocked_response" in joined)

    # Verify each tool produced at least one trace
    def is_tool_group(g):
        return g == "tools" or getattr(g, "value", None) == "tools"

    def count_tool(name: str) -> int:
        return sum(
            1
            for run in tracing.runs.values()
            if run.type == RunType.NODE
            and run.name == name
            and is_tool_group(run.metadata.get("node", {}).get("group"))
        )

    assert count_tool("NoOp Tool") >= 1
    assert count_tool("Exa Search Tool") >= 1
    assert count_tool("Firecrawl Tool") >= 1
