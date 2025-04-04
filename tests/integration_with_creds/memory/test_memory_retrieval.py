from datetime import datetime

import pytest

from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.prompts import Message, MessageRole


@pytest.fixture
def memory_instance():
    """Create a Memory instance with InMemory backend."""
    return Memory(backend=InMemory())


def test_empty_messages(memory_instance):
    """Test that empty input returns empty output."""
    result = memory_instance._extract_valid_conversation([], 10)
    assert result == []


def test_single_user_message(memory_instance):
    """Test with a single USER message."""
    msg = Message(role=MessageRole.USER, content="Hello", metadata={"timestamp": 1.0})
    result = memory_instance._extract_valid_conversation([msg], 10)
    assert len(result) == 1
    assert result[0].role == MessageRole.USER


def test_single_assistant_message(memory_instance):
    """Test with a single ASSISTANT message."""
    msg = Message(role=MessageRole.ASSISTANT, content="Hello", metadata={"timestamp": 1.0})
    result = memory_instance._extract_valid_conversation([msg], 10)
    assert result == []  # Should return empty list as no USER message


def test_basic_conversation_order(memory_instance):
    """Test that messages are properly ordered by timestamp."""
    messages = [
        Message(role=MessageRole.USER, content="U2", metadata={"timestamp": 3.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result) == 3
    assert [msg.content for msg in result] == ["U1", "A1", "U2"]


def test_timestamp_ties_user_priority(memory_instance):
    """Test that USER messages come before ASSISTANT when timestamps tie."""
    messages = [
        Message(role=MessageRole.ASSISTANT, content="A", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.USER, content="U", metadata={"timestamp": 1.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result) == 2
    assert result[0].content == "U"
    assert result[1].content == "A"


def test_limit_most_recent(memory_instance):
    """Test limiting to most recent messages."""
    messages = [
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.USER, content="U2", metadata={"timestamp": 3.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 4.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 2)
    assert len(result) == 2
    assert [msg.content for msg in result] == ["U2", "A2"]


def test_start_with_user_message(memory_instance):
    """Test that returned conversation always starts with USER message."""
    messages = [
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 3.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert [msg.content for msg in result] == ["U1", "A2"]


def test_limit_start_with_user(memory_instance):
    """Test that limited results still start with USER message."""
    messages = [
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 3.0}),
        Message(role=MessageRole.USER, content="U2", metadata={"timestamp": 4.0}),
        Message(role=MessageRole.ASSISTANT, content="A3", metadata={"timestamp": 5.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 3)
    assert len(result) == 2
    assert result[0].role == MessageRole.USER
    assert [msg.content for msg in result] == ["U2", "A3"]


def test_no_user_in_limit(memory_instance):
    """Test when limited set contains no USER messages."""
    messages = [
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 3.0}),
    ]

    # Limit to 2 most recent (both ASSISTANT)
    result = memory_instance._extract_valid_conversation(messages, 2)
    assert result == []


def test_missing_timestamps(memory_instance):
    """Test handling of missing timestamps (should sort to end)."""
    messages = [
        Message(role=MessageRole.USER, content="With timestamp", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.USER, content="No timestamp", metadata={}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result) == 2
    assert result[0].content == "With timestamp"
    assert result[1].content == "No timestamp"


def test_actual_timestamps_from_datetime(memory_instance):
    """Test with real timestamps from datetime."""
    now = datetime.now().timestamp()

    messages = [
        Message(role=MessageRole.USER, content="Earlier", metadata={"timestamp": now - 60}),
        Message(role=MessageRole.ASSISTANT, content="Response", metadata={"timestamp": now - 30}),
        Message(role=MessageRole.USER, content="Latest", metadata={"timestamp": now}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result) == 3
    assert result[0].content == "Earlier"
    assert result[2].content == "Latest"


def test_real_world_example(memory_instance):
    """Test example similar to the verify_memory scenario in the integration tests."""
    messages = [
        Message(
            role=MessageRole.USER,
            content="Hi, my name is Alex",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 1.0},
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Hello Alex",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 2.0},
        ),
        Message(
            role=MessageRole.USER,
            content="I work at TechCorp",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 3.0},
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="TechCorp sounds interesting",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 4.0},
        ),
        Message(
            role=MessageRole.USER,
            content="What's my name and where do I work?",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 5.0},
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Your name is Alex and you work at TechCorp",
            metadata={"user_id": "123", "session_id": "abc", "timestamp": 6.0},
        ),
    ]

    result_all = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result_all) == 6

    result_limited = memory_instance._extract_valid_conversation(messages, 4)
    assert len(result_limited) == 4
    assert result_limited[0].content == "I work at TechCorp"

    result_min = memory_instance._extract_valid_conversation(messages, 2)
    assert len(result_min) == 2
    assert "What's my name" in result_min[0].content


def test_large_conversation_with_different_limits(memory_instance):
    """Test with a large conversation and various limit values."""
    messages = []
    for i in range(1, 11):
        messages.append(Message(role=MessageRole.USER, content=f"U{i}", metadata={"timestamp": float(i * 2 - 1)}))
        messages.append(Message(role=MessageRole.ASSISTANT, content=f"A{i}", metadata={"timestamp": float(i * 2)}))

    result_all = memory_instance._extract_valid_conversation(messages, 30)
    assert len(result_all) == 20

    result_10 = memory_instance._extract_valid_conversation(messages, 10)
    assert len(result_10) == 10
    assert result_10[0].content == "U6"
    result_5 = memory_instance._extract_valid_conversation(messages, 5)
    assert len(result_5) == 4
    assert result_5[0].content == "U9"

    result_3 = memory_instance._extract_valid_conversation(messages, 3)
    assert len(result_3) == 2
    assert result_3[0].content == "U10"
    assert result_3[1].content == "A10"


def test_extract_valid_conversation_step_by_step(memory_instance):
    """Step-by-step test to demonstrate exactly how the method works."""

    messages = [
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.USER, content="U2", metadata={"timestamp": 3.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 4.0}),
        Message(role=MessageRole.USER, content="U3", metadata={"timestamp": 5.0}),
        Message(role=MessageRole.ASSISTANT, content="A3", metadata={"timestamp": 6.0}),
    ]

    result_3 = memory_instance._extract_valid_conversation(messages, 3)
    assert len(result_3) == 2
    assert result_3[0].content == "U3"
    assert result_3[1].content == "A3"

    result_4 = memory_instance._extract_valid_conversation(messages, 4)
    assert len(result_4) == 4
    assert result_4[0].content == "U2"

    result_2 = memory_instance._extract_valid_conversation(messages, 2)
    assert len(result_2) == 2
    assert result_2[0].content == "U3"

    result_1 = memory_instance._extract_valid_conversation(messages, 1)
    assert result_1 == []


def test_understanding_with_odd_ordering(memory_instance):
    """Test with messages that aren't in strict USER/ASSISTANT alternating pattern."""
    messages = [
        Message(role=MessageRole.USER, content="U1", metadata={"timestamp": 1.0}),
        Message(role=MessageRole.ASSISTANT, content="A1", metadata={"timestamp": 2.0}),
        Message(role=MessageRole.ASSISTANT, content="A2", metadata={"timestamp": 3.0}),
        Message(role=MessageRole.ASSISTANT, content="A3", metadata={"timestamp": 4.0}),
        Message(role=MessageRole.USER, content="U2", metadata={"timestamp": 5.0}),
        Message(role=MessageRole.ASSISTANT, content="A4", metadata={"timestamp": 6.0}),
    ]

    result = memory_instance._extract_valid_conversation(messages, 4)
    assert len(result) == 2
    assert result[0].content == "U2"
    assert result[1].content == "A4"
