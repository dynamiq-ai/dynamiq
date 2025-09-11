from collections import defaultdict

import pytest

from dynamiq import Workflow
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import STREAMING_EVENT, StreamingConfig, StreamingMode


@pytest.fixture(scope="module")
def openai_connection():
    return OpenAIConnection()


@pytest.fixture(scope="module")
def openai_llm(openai_connection):
    return OpenAI(
        connection=openai_connection,
        model="gpt-4o",
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def agent_role():
    return "helpful assistant that provides clear and concise information"


@pytest.fixture(scope="module")
def streaming_event():
    return "agent_streaming_test"


@pytest.fixture(scope="module")
def react_agent_with_all_streaming(openai_llm, agent_role, streaming_event):
    agent = ReActAgent(
        name="AllStreamingTestAgent",
        id="all_streaming_test_agent",
        llm=openai_llm,
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        verbose=True,
        streaming=StreamingConfig(
            enabled=True,
            event=streaming_event,
            mode=StreamingMode.ALL,
        ),
    )
    return agent


@pytest.fixture(scope="module")
def react_agent_with_final_streaming(openai_llm, agent_role, streaming_event):
    agent = ReActAgent(
        name="FinalStreamingTestAgent",
        id="final_streaming_test_agent",
        llm=openai_llm,
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        verbose=True,
        streaming=StreamingConfig(
            enabled=True,
            event=streaming_event,
            mode=StreamingMode.FINAL,
        ),
    )
    return agent


@pytest.fixture(scope="module")
def simple_agent_with_streaming(openai_llm, agent_role, streaming_event):
    agent = SimpleAgent(
        name="SimpleStreamingAgent",
        id="simple_streaming_agent",
        llm=openai_llm,
        role=agent_role,
        streaming=StreamingConfig(
            enabled=True,
            event=streaming_event,
            mode=StreamingMode.FINAL,
        ),
    )
    return agent


def collect_streaming_events(streaming_iterator, workflow_id):
    """
    Helper function to collect streaming events from the iterator.
    Returns a tuple of (events_by_entity, final_workflow_event)
    """
    events_output = defaultdict(list)
    final_wf_event_output = None

    for event in streaming_iterator:
        if event.entity_id != workflow_id:
            if "choices" in event.data and "delta" in event.data["choices"][0]:
                content = event.data["choices"][0]["delta"].get("content", "")
                events_output[event.entity_id].append((event.event, content))
            elif "content" in event.data:
                events_output[event.entity_id].append((event.event, event.data["content"]))
            else:
                events_output[event.entity_id].append((event.event, str(event.data)))
        else:
            final_wf_event_output = event.event, event.data

    return events_output, final_wf_event_output


@pytest.mark.integration
def test_react_agent_all_streaming(react_agent_with_all_streaming, streaming_event):
    """Test streaming functionality with ReActAgent in ALL mode."""
    with get_connection_manager():
        streaming = StreamingIteratorCallbackHandler()

        input_data = {"input": "What is the capital of France?"}

        wf = Workflow(flow=Flow(nodes=[react_agent_with_all_streaming]))

        response = wf.run(
            input_data=input_data,
            config=RunnableConfig(callbacks=[streaming]),
        )

        events_output, final_wf_event_output = collect_streaming_events(streaming, wf.id)

        assert response.status == RunnableStatus.SUCCESS
        assert final_wf_event_output[0] == STREAMING_EVENT

        agent_output = events_output[react_agent_with_all_streaming.id]
        assert len(agent_output) > 0
        assert all(event == streaming_event for event, _ in agent_output)

        full_content = " ".join(str(content) for _, content in agent_output)
        assert "Paris" in full_content


@pytest.mark.integration
def test_react_agent_final_streaming(react_agent_with_final_streaming, streaming_event):
    """Test streaming functionality with ReActAgent in FINAL mode."""
    with get_connection_manager():
        streaming = StreamingIteratorCallbackHandler()

        input_data = {"input": "What is the capital of Germany?"}

        wf = Workflow(flow=Flow(nodes=[react_agent_with_final_streaming]))

        response = wf.run(
            input_data=input_data,
            config=RunnableConfig(callbacks=[streaming]),
        )

        events_output, final_wf_event_output = collect_streaming_events(streaming, wf.id)

        assert response.status == RunnableStatus.SUCCESS
        assert final_wf_event_output[0] == STREAMING_EVENT

        agent_output = events_output[react_agent_with_final_streaming.id]
        assert len(agent_output) > 0
        assert all(event == streaming_event for event, _ in agent_output)

        full_content = " ".join(str(content) for _, content in agent_output)
        assert "Berlin" in full_content

        reasoning_indicators = ["I need to", "Let me think", "First,", "Step 1:"]
        found_reasoning = any(indicator in full_content for indicator in reasoning_indicators)
        assert not found_reasoning, "Should not find reasoning steps in FINAL streaming mode"

        assert "Berlin" in str(response.output)


@pytest.mark.integration
def test_simple_agent_streaming(simple_agent_with_streaming, streaming_event):
    """Test streaming functionality with SimpleAgent."""
    with get_connection_manager():
        streaming = StreamingIteratorCallbackHandler()

        input_data = {"input": "What is the tallest mountain in the world?"}

        wf = Workflow(flow=Flow(nodes=[simple_agent_with_streaming]))

        response = wf.run(
            input_data=input_data,
            config=RunnableConfig(callbacks=[streaming]),
        )

        events_output, final_wf_event_output = collect_streaming_events(streaming, wf.id)

        assert response.status == RunnableStatus.SUCCESS
        assert final_wf_event_output[0] == STREAMING_EVENT

        agent_output = events_output[simple_agent_with_streaming.id]
        assert len(agent_output) > 0
        assert all(event == streaming_event for event, _ in agent_output)

        full_content = " ".join(str(content) for _, content in agent_output)
        assert "Everest" in full_content
