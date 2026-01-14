"""Tests for agent history summarization functionality."""

import uuid

import pytest

from dynamiq import connections
from dynamiq.nodes.agents import Agent, SummarizationMode
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole, Prompt


@pytest.fixture
def openai_connection():
    return connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api-key",
    )


@pytest.fixture
def openai_node(openai_connection):
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=openai_connection,
        prompt=Prompt(
            messages=[
                Message(
                    role="user",
                    content="{{input}}",
                ),
            ],
        ),
    )


def test_agent_summarization_replace_mode(openai_node, mock_llm_executor):
    """Test that agent triggers summarization in replace mode (replaces entire history)."""

    # Create agent with replace mode (default) to replace history
    agent = Agent(
        name="Test Agent",
        llm=openai_node,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=100,  # Very low limit to trigger summarization
            max_attempts=2,
            mode=SummarizationMode.REPLACE,  # Use replace mode (default)
        ),
    )
    # Add messages with "Observation:" to trigger tool output summarization
    agent._prompt.messages.append(Message(content="User message", role=MessageRole.USER))
    # Add many messages to exceed token limit
    for i in range(20):
        agent._prompt.messages.append(
            Message(content=f"User message {i} with some content to increase tokens", role=MessageRole.USER)
        )
        agent._prompt.messages.append(
            Message(content=f"Assistant response {i} with detailed information", role=MessageRole.ASSISTANT)
        )

    initial_message_count = len(agent._prompt.messages)
    assert initial_message_count == 41  # 1 user + 20 assistant + 20 observation

    # Check that token limit is exceeded
    assert agent._history_manager.is_token_limit_exceeded()

    # Trigger summarization
    input_message = Message(content="Test query", role=MessageRole.USER)
    summary_offset = 1

    new_offset = agent._history_manager.summarize_history(
        input_message=input_message,
        summary_offset=summary_offset,
        config=None,
    )

    # After summarization, message count should be dramatically reduced
    final_message_count = len(agent._prompt.messages)

    # Should have very few messages now (system messages + summary message)
    assert final_message_count < initial_message_count
    assert final_message_count < 10  # Should be much smaller

    # Verify that the last message contains "Conversation Summary" or "summary"
    last_message = agent._prompt.messages[-1]
    assert "summary" in last_message.content.lower() or "Summary" in last_message.content

    # New offset should point to the end of the message list
    assert new_offset == final_message_count


def test_agent_summarization_preserve_mode(openai_node, mock_llm_executor):
    """Test that agent triggers summarization in preserve mode (keeps message structure)."""

    # Create agent with preserve mode to keep message structure
    agent = Agent(
        name="Test Agent",
        llm=openai_node,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=100,  # Very low limit to trigger summarization
            max_attempts=2,
            mode=SummarizationMode.PRESERVE,  # Use preserve mode
        ),
    )

    # Add messages with "Observation:" to trigger tool output summarization
    agent._prompt.messages.append(Message(content="User message", role=MessageRole.USER))
    for i in range(10):
        agent._prompt.messages.append(Message(content=f"Assistant reasoning {i}", role=MessageRole.ASSISTANT))
        agent._prompt.messages.append(
            Message(content=f"Observation: Tool output for action {i} with detailed results", role=MessageRole.USER)
        )

    initial_message_count = len(agent._prompt.messages)
    assert initial_message_count == 21  # 1 user + 10 assistant + 10 observation

    # Check that token limit is exceeded
    assert agent._history_manager.is_token_limit_exceeded()

    # Trigger summarization
    input_message = Message(content="Test query", role=MessageRole.USER)
    summary_offset = 1

    _ = agent._history_manager.summarize_history(
        input_message=input_message,
        summary_offset=summary_offset,
        config=None,
    )

    # In preserve mode, message count should stay similar (messages shortened, not replaced)
    final_message_count = len(agent._prompt.messages)

    # Message count should be similar (structure preserved)
    assert final_message_count == initial_message_count

    # Some messages should contain "Observation (shortened):" indicating they were summarized
    observation_messages = [
        msg for msg in agent._prompt.messages if msg.role == MessageRole.USER and "Observation" in msg.content
    ]
    assert len(observation_messages) > 0


def test_agent_summarization_mode_switching(openai_node, mock_llm_executor):
    """Test switching between summarization modes."""

    # Test replace mode
    agent_replace = Agent(
        name="Test Agent Replace",
        llm=openai_node,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=100,
            mode=SummarizationMode.REPLACE,
        ),
    )

    assert agent_replace.summarization_config.mode == SummarizationMode.REPLACE

    # Test preserve mode
    agent_preserve = Agent(
        name="Test Agent Preserve",
        llm=openai_node,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=100,
            mode=SummarizationMode.PRESERVE,
        ),
    )

    assert agent_preserve.summarization_config.mode == SummarizationMode.PRESERVE
