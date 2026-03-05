"""Unit tests for HistoryManagerMixin — token-bounded _split_history and _compact_history."""

import uuid
from unittest.mock import patch

from dynamiq.nodes.agents.components.history_manager import HistoryManagerMixin
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.prompts import Message, MessageRole
from dynamiq.prompts.prompts import Prompt

MODEL = "gpt-4o-mini"


class FakeLLM:
    """Minimal LLM stub with model name and token limit."""

    def __init__(self, model: str = MODEL, token_limit: int = 200_000):
        self.model = model
        self._token_limit = token_limit

    def get_token_limit(self) -> int:
        return self._token_limit


class FakeAgent(HistoryManagerMixin):
    """Minimal object that satisfies HistoryManagerMixin's protocol."""

    def __init__(
        self,
        messages: list[Message],
        history_offset: int = 1,
        max_preserved_tokens: int = 10_000,
        max_token_context_length: int | None = None,
        context_usage_ratio: float = 0.8,
    ):
        self.llm = FakeLLM()
        self.summarization_config = SummarizationConfig(
            enabled=True,
            max_preserved_tokens=max_preserved_tokens,
            max_token_context_length=max_token_context_length,
            context_usage_ratio=context_usage_ratio,
        )
        self._prompt = Prompt(messages=messages)
        self._history_offset = history_offset
        self.name = "test-agent"
        self.id = str(uuid.uuid4())


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content, static=True)


def _system(content: str = "You are a helpful assistant.") -> Message:
    return _msg(MessageRole.SYSTEM, content)


def _user(content: str) -> Message:
    return _msg(MessageRole.USER, content)


def _assistant(content: str) -> Message:
    return _msg(MessageRole.ASSISTANT, content)


class TestSplitHistory:
    """Tests for _split_history token-bounded splitting."""

    def test_empty_history(self):
        agent = FakeAgent(messages=[_system()])
        to_summarize, to_preserve = agent._split_history()
        assert to_summarize == []
        assert to_preserve == []

    def test_small_messages_all_preserved(self):
        """When all messages fit within max_preserved_tokens, all go to to_preserve."""
        msgs = [_system(), _user("Hello"), _assistant("Hi there"), _user("How are you?")]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=10_000)

        to_summarize, to_preserve = agent._split_history()

        assert to_summarize == []
        assert len(to_preserve) == 3

    def test_large_message_exceeds_budget(self):
        """A single message exceeding the budget results in empty to_preserve."""
        huge_content = "word " * 20_000  # ~20k tokens
        msgs = [_system(), _user(huge_content)]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=100)

        to_summarize, to_preserve = agent._split_history()

        assert len(to_summarize) == 1
        assert to_preserve == []

    def test_newest_messages_preserved_first(self):
        """Preservation fills from newest to oldest."""
        msgs = [
            _system(),
            _user("msg1 " * 500),
            _assistant("msg2 " * 500),
            _user("msg3 " * 500),
            _assistant("msg4 " * 500),
            _user("newest message"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=100)

        to_summarize, to_preserve = agent._split_history()

        assert len(to_preserve) >= 1
        assert to_preserve[-1].content == "newest message"
        assert len(to_summarize) + len(to_preserve) == 5

    def test_split_covers_all_messages(self):
        """to_summarize + to_preserve should account for all conversation messages."""
        msgs = [
            _system(),
            _user("first"),
            _assistant("second"),
            _user("third"),
            _assistant("fourth"),
            _user("fifth"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=500)

        to_summarize, to_preserve = agent._split_history()

        total = len(to_summarize) + len(to_preserve)
        assert total == 5  # excludes system message (history_offset=1)

    def test_preserved_messages_are_copies(self):
        """Preserved messages should be copies, not references to originals."""
        msgs = [_system(), _user("hello"), _assistant("world")]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=10_000)

        _, to_preserve = agent._split_history()

        for preserved_msg in to_preserve:
            assert preserved_msg is not msgs[1] and preserved_msg is not msgs[2]

    def test_zero_budget_preserves_nothing(self):
        """max_preserved_tokens=0 means nothing is preserved."""
        msgs = [_system(), _user("hello"), _assistant("world")]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=0)

        to_summarize, to_preserve = agent._split_history()

        assert len(to_summarize) == 2
        assert to_preserve == []

    def test_history_offset_respected(self):
        """Messages before history_offset are not included in the split."""
        msgs = [_system(), _user("initial instruction"), _user("conv1"), _assistant("conv2")]
        agent = FakeAgent(messages=msgs, history_offset=2, max_preserved_tokens=10_000)

        to_summarize, to_preserve = agent._split_history()

        total = len(to_summarize) + len(to_preserve)
        assert total == 2  # only conv1, conv2

    def test_token_limit_exceeded_forces_summarize_all(self):
        """When conversation fits in max_preserved_tokens but the total prompt
        exceeds the token limit, all messages must go to to_summarize so that
        compaction makes progress instead of looping as a no-op."""
        msgs = [_system(), _user("Hello"), _assistant("Hi there"), _user("How are you?")]
        agent = FakeAgent(
            messages=msgs,
            max_preserved_tokens=10_000,
            max_token_context_length=1,
        )

        assert agent.is_token_limit_exceeded()

        to_summarize, to_preserve = agent._split_history()

        assert len(to_summarize) == 3
        assert to_preserve == []


class TestCompactHistory:
    """Tests for _compact_history."""

    def test_compact_with_summary(self):
        """After compaction with a summary, prompt has prefix + summary + preserved."""
        msgs = [
            _system(),
            _user("old message " * 1000),
            _assistant("old response " * 1000),
            _user("recent question"),
            _assistant("recent answer"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=500)

        agent._compact_history(summary="This is a summary of the conversation.")

        assert agent._prompt.messages[0].content == msgs[0].content
        summary_msg = agent._prompt.messages[1]
        assert "summary" in summary_msg.content.lower()
        assert summary_msg.role == MessageRole.USER

    def test_compact_without_summary(self):
        """Compaction without summary keeps only prefix + preserved."""
        msgs = [
            _system(),
            _user("old message " * 1000),
            _assistant("old response " * 1000),
            _user("recent"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=500)

        agent._compact_history(summary=None)

        assert agent._prompt.messages[0].content == msgs[0].content
        assert len(agent._prompt.messages) >= 1
        for msg in agent._prompt.messages[1:]:
            assert "Observation" not in msg.content

    def test_compact_reduces_message_count(self):
        """Compaction should reduce the total number of messages."""
        msgs = [_system()] + [
            _user(f"message {i} " * 200) if i % 2 == 0 else _assistant(f"response {i} " * 200) for i in range(20)
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=2000)
        original_count = len(agent._prompt.messages)

        agent._compact_history(summary="Summary of 20 messages.")

        assert len(agent._prompt.messages) < original_count

    def test_precomputed_preserved_skips_split_history(self):
        """When preserved is passed explicitly, _split_history must not be called."""
        msgs = [
            _system(),
            _user("old message " * 1000),
            _assistant("old response " * 1000),
            _user("recent question"),
            _assistant("recent answer"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=500)
        preserved = [_user("recent question"), _assistant("recent answer")]

        with patch.object(agent, "_split_history", wraps=agent._split_history) as spy:
            agent._compact_history(summary="A summary.", preserved=preserved)
            spy.assert_not_called()

        assert agent._prompt.messages[0].content == msgs[0].content
        assert "summary" in agent._prompt.messages[1].content.lower()
        assert agent._prompt.messages[2].content == "recent question"
        assert agent._prompt.messages[3].content == "recent answer"

    def test_no_preserved_arg_falls_back_to_split(self):
        """When preserved is not supplied, _split_history is still called."""
        msgs = [
            _system(),
            _user("old message " * 1000),
            _assistant("old response " * 1000),
            _user("recent"),
        ]
        agent = FakeAgent(messages=msgs, max_preserved_tokens=500)

        with patch.object(agent, "_split_history", wraps=agent._split_history) as spy:
            agent._compact_history(summary="A summary.")
            spy.assert_called_once()
