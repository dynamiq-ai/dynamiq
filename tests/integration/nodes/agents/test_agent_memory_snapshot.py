"""
Integration tests for agent memory snapshot behavior.

Verifies that _save_history_to_memory replaces only the current user/session
slice and writes the current prompt state (all non-system messages) after
each agent run.
"""

import uuid

import pytest

from dynamiq import connections
from dynamiq.memory import Memory, MemorySaveMode
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableStatus

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


def test_memory_on_failure_appends_only_user_input(llm, memory, mocker):
    """When the agent explicitly fails, only the user input should be appended to memory."""
    mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        side_effect=Exception("LLM failure"),
    )
    agent = Agent(
        name="FailingAgent",
        llm=llm,
        tools=[],
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )
    result = agent.run(input_data={"input": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    assert result.status == RunnableStatus.FAILURE

    stored = memory.backend.messages
    assert len(stored) == 1, f"Only user input should be stored on failure, got {len(stored)}"
    assert stored[0].role == MessageRole.USER
    assert stored[0].content == "Hello"
    assert stored[0].metadata.get("user_id") == USER_ID
    assert stored[0].metadata.get("session_id") == SESSION_ID


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


def test_memory_retrieved_messages_are_static(memory):
    """Messages from memory must have static=True to skip Jinja2 rendering."""
    memory.add(role=MessageRole.USER, content="hello", metadata={"user_id": USER_ID, "session_id": SESSION_ID})
    memory.add(role=MessageRole.ASSISTANT, content="hi", metadata={"user_id": USER_ID, "session_id": SESSION_ID})

    filters = {"user_id": USER_ID, "session_id": SESSION_ID}
    for msg in memory.get_all():
        assert msg.static is True
    for msg in memory.search(filters=filters):
        assert msg.static is True
    for msg in memory.get_agent_conversation(filters=filters):
        assert msg.static is True


def _seed_react_prompt(agent, user_input: str, final_answer: str) -> None:
    """Seed the agent with a pinned input and a simulated ReAct trace (thought,
    observation, final answer) on ``_prompt.messages`` so save-mode behavior
    can be tested without running a real tool loop.
    """
    agent._pinned_input = Message(role=MessageRole.USER, content=user_input)
    agent._prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content=user_input),
            Message(role=MessageRole.ASSISTANT, content="Thought: I should use the lookup tool."),
            Message(role=MessageRole.USER, content="Observation: tool returned 'Alex works at TechCorp.'"),
            Message(role=MessageRole.ASSISTANT, content=final_answer),
        ]
    )


def test_save_mode_full_persists_intermediate_messages(llm):
    """Default FULL mode stores every non-system message, including the trace."""
    memory = Memory(backend=InMemory(), save_mode=MemorySaveMode.FULL)
    agent = Agent(
        name="FullModeAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )
    assert agent.memory.save_mode == MemorySaveMode.FULL

    _seed_react_prompt(agent, user_input="Where does Alex work?", final_answer="Alex works at TechCorp.")
    agent._save_history_to_memory(
        {"user_id": USER_ID, "session_id": SESSION_ID}, final_output="Alex works at TechCorp."
    )

    stored = memory.backend.messages
    assert len(stored) == 4, f"Expected full trace (4 msgs) in memory, got {len(stored)}"
    assert [m.role for m in stored] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert any("Thought" in m.content for m in stored)
    assert any("Observation" in m.content for m in stored)


def test_save_mode_input_output_persists_only_input_and_final(llm):
    """INPUT_OUTPUT mode drops intermediate thoughts/observations."""
    memory = Memory(backend=InMemory())
    agent = Agent(
        name="IOModeAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=Memory(backend=memory.backend, save_mode=MemorySaveMode.INPUT_OUTPUT),
    )

    _seed_react_prompt(agent, user_input="Where does Alex work?", final_answer="Alex works at TechCorp.")
    agent._save_history_to_memory(
        {"user_id": USER_ID, "session_id": SESSION_ID}, final_output="Alex works at TechCorp."
    )

    stored = memory.backend.messages
    assert len(stored) == 2, f"Expected only input+final (2 msgs), got {len(stored)}"
    assert stored[0].role == MessageRole.USER
    assert stored[0].content == "Where does Alex work?"
    assert stored[1].role == MessageRole.ASSISTANT
    assert stored[1].content == "Alex works at TechCorp."
    assert not any("Thought" in m.content for m in stored)
    assert not any("Observation" in m.content for m in stored)


def test_save_mode_input_output_replaces_prior_full_snapshot(llm):
    """Switching to INPUT_OUTPUT on an existing scoped history replaces, not appends."""
    memory = Memory(backend=InMemory(), save_mode=MemorySaveMode.FULL)
    full_agent = Agent(name="FirstAgent", llm=llm, tools=[], memory=memory)
    _seed_react_prompt(full_agent, user_input="Q1", final_answer="A1")
    full_agent._save_history_to_memory({"user_id": USER_ID, "session_id": SESSION_ID})
    assert len(memory.backend.messages) == 4

    io_agent = Agent(
        name="SecondAgent",
        llm=llm,
        tools=[],
        memory=Memory(backend=memory.backend, save_mode=MemorySaveMode.INPUT_OUTPUT),
    )
    _seed_react_prompt(io_agent, user_input="Q2", final_answer="A2")
    io_agent._save_history_to_memory({"user_id": USER_ID, "session_id": SESSION_ID}, final_output="A2")

    stored = memory.backend.messages
    assert len(stored) == 2
    assert stored[0].content == "Q2"
    assert stored[1].content == "A2"


def test_save_mode_input_output_preserves_prior_turns_while_dropping_trace(llm):
    """INPUT_OUTPUT should preserve accumulated chat turns, not just the latest pair."""
    memory = Memory(backend=InMemory())
    agent = Agent(
        name="IOMultiTurnAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=Memory(backend=memory.backend, save_mode=MemorySaveMode.INPUT_OUTPUT),
    )

    agent._pinned_input = Message(role=MessageRole.USER, content="Q2")
    agent._prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Q1"),
            Message(role=MessageRole.ASSISTANT, content="A1"),
            Message(role=MessageRole.USER, content="Q2"),
            Message(role=MessageRole.ASSISTANT, content="Thought: I should use the lookup tool."),
            Message(role=MessageRole.USER, content="Observation: tool returned 'A2'."),
            Message(role=MessageRole.ASSISTANT, content="A2"),
        ]
    )

    agent._save_history_to_memory({"user_id": USER_ID, "session_id": SESSION_ID}, final_output="A2")

    stored = memory.get_agent_conversation(filters={"user_id": USER_ID, "session_id": SESSION_ID})
    assert [m.role for m in stored] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [m.content for m in stored] == ["Q1", "A1", "Q2", "A2"]
    assert not any("Thought:" in m.content for m in stored)
    assert not any("Observation:" in m.content for m in stored)


def test_save_mode_input_output_uses_final_output_for_current_assistant_turn(llm):
    """INPUT_OUTPUT should use the resolved final output for the current assistant turn."""
    memory = Memory(backend=InMemory())
    agent = Agent(
        name="IOAnswerExtractionAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=Memory(backend=memory.backend, save_mode=MemorySaveMode.INPUT_OUTPUT),
    )

    agent._pinned_input = Message(role=MessageRole.USER, content="How are you?")
    agent._prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello, I'm Oleks"),
            Message(role=MessageRole.ASSISTANT, content="Hello, Oleks!"),
            Message(role=MessageRole.USER, content="How are you?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Thought: The user is asking how I am doing.\nAnswer: Placeholder raw LLM output.",
            ),
        ]
    )

    agent._save_history_to_memory(
        {"user_id": USER_ID, "session_id": SESSION_ID},
        final_output="I'm here and ready to help. How can I assist you today?",
    )

    stored = memory.get_agent_conversation(filters={"user_id": USER_ID, "session_id": SESSION_ID})
    assert [m.role for m in stored] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [m.content for m in stored] == [
        "Hello, I'm Oleks",
        "Hello, Oleks!",
        "How are you?",
        "I'm here and ready to help. How can I assist you today?",
    ]


@pytest.mark.parametrize(
    ("inference_mode", "action_message", "final_message"),
    [
        (
            InferenceMode.FUNCTION_CALLING,
            'Function call: lookup({"thought": "search", "action_input": {"query": "Alex"}})',
            'Function call: provide_final_answer({"thought": "done", "answer": "A2"})',
        ),
        (
            InferenceMode.STRUCTURED_OUTPUT,
            '{"thought": "search", "action": "lookup", "action_input": {"query": "Alex"}}',
            '{"thought": "done", "action": "finish", "action_input": "A2"}',
        ),
        (
            InferenceMode.XML,
            "<output><thought>search</thought><action>lookup</action><action_input>{}</action_input></output>",
            "<output><thought>done</thought><answer>A2</answer></output>",
        ),
    ],
)
def test_save_mode_input_output_filters_non_default_trace_messages(
    llm,
    inference_mode,
    action_message,
    final_message,
):
    """INPUT_OUTPUT mode should drop tool traces for every inference mode."""
    memory = Memory(backend=InMemory())
    agent = Agent(
        name="IONonDefaultModeAgent",
        llm=llm,
        tools=[],
        inference_mode=inference_mode,
        memory=Memory(backend=memory.backend, save_mode=MemorySaveMode.INPUT_OUTPUT),
    )

    agent._pinned_input = Message(role=MessageRole.USER, content="Q2")
    agent._prompt = Prompt(
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Q1"),
            Message(role=MessageRole.ASSISTANT, content="A1"),
            Message(role=MessageRole.USER, content="Q2"),
            Message(role=MessageRole.ASSISTANT, content=action_message),
            Message(role=MessageRole.USER, content="Observation: tool returned A2."),
            Message(role=MessageRole.ASSISTANT, content=final_message),
        ]
    )

    agent._save_history_to_memory({"user_id": USER_ID, "session_id": SESSION_ID}, final_output="A2")

    stored = memory.get_agent_conversation(filters={"user_id": USER_ID, "session_id": SESSION_ID})
    assert [m.role for m in stored] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [m.content for m in stored] == ["Q1", "A1", "Q2", "A2"]
    assert not any("Function call:" in m.content for m in stored)
    assert not any("Observation:" in m.content for m in stored)
    assert not any("<thought>" in m.content for m in stored)


def test_save_mode_input_output_single_scope_fallback_does_not_prefilter_snapshot(llm, mocker):
    """Append-only fallback should build the current pair without pre-filtering the full snapshot."""
    memory = Memory(backend=InMemory(), save_mode=MemorySaveMode.INPUT_OUTPUT)
    agent = Agent(
        name="IOSingleScopeFallbackAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )
    _seed_react_prompt(agent, user_input="Where does Alex work?", final_answer="Alex works at TechCorp.")
    input_output_history_spy = mocker.spy(agent, "_input_output_history")

    agent._save_history_to_memory({"user_id": USER_ID}, final_output="Alex works at TechCorp.")

    input_output_history_spy.assert_not_called()
    stored = memory.get_agent_conversation(filters={"user_id": USER_ID})
    assert [m.role for m in stored] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert [m.content for m in stored] == [
        "Where does Alex work?",
        "Alex works at TechCorp.",
    ]


def test_fallback_append_preserves_react_assistant_response_in_full_mode(llm):
    """Append-only fallback must not drop ReAct assistant turns that start with Thought:."""
    memory = Memory(backend=InMemory(), save_mode=MemorySaveMode.FULL)
    agent = Agent(
        name="FallbackFullModeAgent",
        llm=llm,
        tools=[],
        inference_mode=InferenceMode.DEFAULT,
        memory=memory,
    )

    _seed_react_prompt(
        agent,
        user_input="Where does Alex work?",
        final_answer="Thought: I now know the answer.\nAnswer: Alex works at TechCorp.",
    )

    agent._save_history_to_memory({"user_id": USER_ID}, final_output="Alex works at TechCorp.")

    stored = memory.get_agent_conversation(filters={"user_id": USER_ID})
    assert [m.role for m in stored] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert [m.content for m in stored] == [
        "Where does Alex work?",
        "Alex works at TechCorp.",
    ]
