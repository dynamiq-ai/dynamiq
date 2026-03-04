"""
Integration tests for agent memory snapshot behavior.

Verifies that _save_history_to_memory replaces only the current user/session
slice and writes the current prompt state (all non-system messages) after
each agent run.
"""

import uuid

import pytest

from dynamiq import connections
from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole

USER_ID = "test-user"
SESSION_ID = "test-session"


@pytest.fixture
def openai_connection():
    return connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key")


@pytest.fixture
def llm(openai_connection):
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=openai_connection)


@pytest.fixture
def memory():
    return Memory(backend=InMemory())


@pytest.fixture
def agent(llm, memory, mock_llm_executor):
    return Agent(
        name="MemorySnapshotAgent",
        llm=llm,
        tools=[],
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )


def test_memory_snapshot_saves_all_non_system_messages(agent, memory):
    """After a single run, memory should contain all non-system messages from the prompt."""
    agent.run(input_data={"input": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    prompt_non_system = [m for m in agent._prompt.messages if m.role != MessageRole.SYSTEM]
    stored = memory.backend.messages

    assert len(stored) > 0, "Memory should not be empty after a run"
    assert len(stored) == len(prompt_non_system), (
        f"Memory should contain exactly the non-system prompt messages: "
        f"expected {len(prompt_non_system)}, got {len(stored)}"
    )

    for stored_msg, prompt_msg in zip(stored, prompt_non_system):
        assert stored_msg.role == prompt_msg.role
        expected_content = prompt_msg.content if isinstance(prompt_msg, Message) else "Image input"
        assert stored_msg.content == expected_content


def test_memory_snapshot_replaces_on_second_run(agent, memory):
    """On a second run, memory.clear() should wipe previous messages and write the new snapshot."""
    agent.run(input_data={"input": "First message", "user_id": USER_ID, "session_id": SESSION_ID})
    first_run_count = len(memory.backend.messages)
    assert first_run_count > 0

    agent.run(input_data={"input": "Second message", "user_id": USER_ID, "session_id": SESSION_ID})
    second_run_messages = memory.backend.messages

    contents = [m.content for m in second_run_messages]
    assert "First message" in contents, "History from first run should be included in snapshot"
    assert "Second message" in contents, "New user input should be in snapshot"

    prompt_non_system = [m for m in agent._prompt.messages if m.role != MessageRole.SYSTEM]
    assert len(second_run_messages) == len(
        prompt_non_system
    ), f"Memory should mirror the prompt: expected {len(prompt_non_system)}, got {len(second_run_messages)}"


def test_memory_snapshot_no_system_messages_stored(agent, memory):
    """System messages (agent prompt, history header) must never be persisted to memory."""
    agent.run(input_data={"input": "Test input", "user_id": USER_ID, "session_id": SESSION_ID})

    for msg in memory.backend.messages:
        assert msg.role != MessageRole.SYSTEM, f"System message leaked into memory: {msg.content[:60]}"


def test_memory_not_used_without_user_or_session(llm, mock_llm_executor):
    """When user_id and session_id are both None, memory should not be touched."""
    mem = Memory(backend=InMemory())
    agent = Agent(
        name="NoMemoryAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=mem,
    )
    agent.run(input_data={"input": "Hi"})

    assert mem.backend.messages == [], "Memory should remain empty when no user_id/session_id provided"


def test_memory_metadata_contains_user_and_session(agent, memory):
    """Each stored message should carry user_id and session_id in its metadata."""
    agent.run(input_data={"input": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    for msg in memory.backend.messages:
        assert msg.metadata.get("user_id") == USER_ID, f"Missing user_id in metadata: {msg.metadata}"
        assert msg.metadata.get("session_id") == SESSION_ID, f"Missing session_id in metadata: {msg.metadata}"


def test_memory_snapshot_replaces_only_current_user_scope(llm, memory, mock_llm_executor):
    """Replacing one user's snapshot must preserve other users' messages."""
    agent = Agent(
        name="ScopedMemorySnapshotAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )

    user_a = {"user_id": "user-a", "session_id": "session-a"}
    user_b = {"user_id": "user-b", "session_id": "session-b"}

    # Seed memory for user A and user B.
    agent.run(input_data={"input": "A1", **user_a})
    agent.run(input_data={"input": "B1", **user_b})

    before = memory.get_agent_conversation(filters=user_b)
    assert len(before) > 0, "Expected user B history to exist before user A update"
    before_contents = [m.content for m in before]

    # Update only user A snapshot.
    agent.run(input_data={"input": "A2", **user_a})

    after_b = memory.get_agent_conversation(filters=user_b)
    after_b_contents = [m.content for m in after_b]
    assert after_b_contents == before_contents, "User B history should remain unchanged"

    after_a = memory.get_agent_conversation(filters=user_a)
    assert any("A2" in m.content for m in after_a), "User A snapshot should contain the latest input"
