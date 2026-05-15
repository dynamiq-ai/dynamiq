from unittest.mock import MagicMock

import pytest

from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.prompts import Message, MessageRole
from dynamiq.runnables import RunnableStatus
from dynamiq.types.streaming import StreamingConfig


def _mock_tool(outputs):
    llm = MagicMock(spec=BaseLLM)
    llm.id = "mock-llm"
    llm.name = "mock-llm"
    llm.type = "mock-llm"
    llm.model = "gpt-4o-mini"
    llm.get_token_limit.return_value = 128_000
    llm.is_postponed_component_init = False
    llm.to_dict.return_value = {}
    llm.streaming = StreamingConfig()
    llm.error_handling = ErrorHandling()
    llm.run.side_effect = [MagicMock(status=RunnableStatus.SUCCESS, output=out) for out in outputs]
    return ContextManagerTool(llm=llm, max_retries=3), llm


def test_retry_returns_summary_after_empty_attempts():
    tool, llm = _mock_tool([{"content": ""}, {"content": "  "}, {"content": "ok"}])
    assert tool._call_llm_for_summary([Message(role=MessageRole.USER, content="x")]) == "ok"
    assert llm.run.call_count == 3


def test_retry_raises_when_all_attempts_empty():
    tool, llm = _mock_tool([{"content": ""}] * 3)
    with pytest.raises(ValueError, match="failed to generate summary after 3 attempts"):
        tool._call_llm_for_summary([Message(role=MessageRole.USER, content="x")])
    assert llm.run.call_count == 3


USER_PROMPT = "Compare the weather in Paris and Tokyo today."
TOOL_NAME = "weather_api"
TOOL_CALL_A_ID = "call_A"
TOOL_CALL_B_ID = "call_B"
TOOL_CALL_C_ID = "call_C"
TOOL_CALL_A_ARGS = '{"thought": "Need Paris first", "city": "Paris"}'
TOOL_CALL_B_ARGS = '{"thought": "Now Tokyo", "city": "Tokyo"}'
PARIS_OBSERVATION = '{"temp": 18, "conditions": "cloudy"}'
TOKYO_OBSERVATION = '{"temp": 24, "conditions": "sunny"}'
FINAL_ANSWER_TEXT = "Paris is 18C cloudy, Tokyo is 24C sunny."
FINAL_ANSWER_ARGS = '{"thought": "Both readings in hand", ' f'"answer": "{FINAL_ANSWER_TEXT}", ' '"output_files": ""}'


def _build_fc_history() -> list[Message]:
    """Build a realistic FC conversation history mirroring `_append_assistant_message`."""
    return [
        Message(role=MessageRole.USER, content=USER_PROMPT),
        Message(
            role=MessageRole.ASSISTANT,
            content=f"Calling: {TOOL_NAME}, {TOOL_NAME}",
            tool_calls=[
                {
                    "id": TOOL_CALL_A_ID,
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": TOOL_CALL_A_ARGS},
                },
                {
                    "id": TOOL_CALL_B_ID,
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": TOOL_CALL_B_ARGS},
                },
            ],
        ),
        Message(
            role=MessageRole.TOOL,
            content=PARIS_OBSERVATION,
            tool_call_id=TOOL_CALL_A_ID,
            name=TOOL_NAME,
        ),
        Message(
            role=MessageRole.TOOL,
            content=TOKYO_OBSERVATION,
            tool_call_id=TOOL_CALL_B_ID,
            name=TOOL_NAME,
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: provide_final_answer",
            tool_calls=[
                {
                    "id": TOOL_CALL_C_ID,
                    "type": "function",
                    "function": {"name": "provide_final_answer", "arguments": FINAL_ANSWER_ARGS},
                },
            ],
        ),
        Message(
            role=MessageRole.TOOL,
            content="Acknowledged.",
            tool_call_id=TOOL_CALL_C_ID,
            name="provide_final_answer",
        ),
    ]


def test_fc_summarizer_input_preserves_all_substantive_info():
    """When the whole FC conversation is summarized, every substantive piece of
    information must reach the summarizer LLM via `_flatten_messages_to_single`.

    Substantive info covers:
        - the user prompt
        - tool names the agent decided to call
        - tool arguments (what was asked of each tool)
        - chain-of-thought the agent emitted alongside each tool call
        - tool outputs (the actual returned data)
        - final answer text the agent produced
    """
    tool, _ = _mock_tool([{"content": "unused"}])

    body = tool._flatten_messages_to_single(_build_fc_history()).content

    # User prompt
    assert USER_PROMPT in body, "user prompt missing from summarizer input"

    # Tool names (preserved via the assistant content label).
    assert TOOL_NAME in body, "tool name missing from summarizer input"
    assert "provide_final_answer" in body, "final-answer function name missing from summarizer input"

    # Tool arguments — currently live in `tool_calls.function.arguments` which
    # `_flatten_messages_to_single` does NOT inspect. This assertion FAILS on
    # current code and surfaces the gap.
    assert '"city": "Paris"' in body, "Paris tool call arguments missing from summarizer input"
    assert '"city": "Tokyo"' in body, "Tokyo tool call arguments missing from summarizer input"

    # Chain-of-thought emitted alongside each tool call — same root cause as
    # tool arguments: lives in `tool_calls.function.arguments`.
    assert "Need Paris first" in body, "tool-call thought missing from summarizer input"
    assert "Both readings in hand" in body, "final-answer thought missing from summarizer input"

    # Tool outputs (this part already works).
    assert PARIS_OBSERVATION in body, "Paris tool output missing from summarizer input"
    assert TOKYO_OBSERVATION in body, "Tokyo tool output missing from summarizer input"

    # Final answer text — lives inside provide_final_answer.arguments.answer.
    assert FINAL_ANSWER_TEXT in body, "final answer text missing from summarizer input"


def test_fc_split_history_preserves_fc_protocol(monkeypatch):
    """`_split_history` plus the orphan-cleanup loop must produce a `to_preserve`
    tail that satisfies the function-calling protocol invariants:

        1. preserve does not start with a `role=TOOL` message
        2. every `assistant.tool_calls[*].id` in preserve has a matching
           `role=TOOL` message with `tool_call_id == id` later in preserve
        3. every `role=TOOL` message in preserve has a matching parent
           `assistant.tool_calls[*].id` earlier in preserve

    Stresses the boundary by forcing a tiny preserve budget so the orphan
    cleanup has to actively move tool replies into the summarize bucket.
    """
    from dynamiq.nodes.agents.components.history_manager import HistoryManagerMixin
    from dynamiq.nodes.agents.utils import SummarizationConfig
    from dynamiq.prompts import Prompt

    class _Agent(HistoryManagerMixin):
        def __init__(self, messages: list[Message]):
            self._prompt = Prompt(messages=[Message(role=MessageRole.SYSTEM, content="sys")] + messages)
            self._history_offset = 1
            self.llm = MagicMock()
            self.llm.model = "gpt-4o-mini"
            self.llm.get_token_limit = MagicMock(return_value=128_000)
            self.name = "test-agent"
            self.id = "test-id"
            self.summarization_config = SummarizationConfig(
                enabled=True,
                max_preserved_tokens=40,  # tight; forces a mid-block split
                max_token_context_length=None,
                context_usage_ratio=0.7,
            )

    history = _build_fc_history()
    agent = _Agent(history)

    to_summarize, to_preserve = agent._split_history()

    # Invariant 1: preserve never starts with a tool message
    assert (
        not to_preserve or to_preserve[0].role != MessageRole.TOOL
    ), f"preserve starts with role=TOOL (orphan): {to_preserve[0]}"

    # Collect tool_call_ids that appear in preserve under either an
    # assistant.tool_calls payload or a TOOL message.
    expected_tool_call_ids: set[str] = set()
    seen_tool_replies: set[str] = set()
    for msg in to_preserve:
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                expected_tool_call_ids.add(tc["id"])
        elif msg.role == MessageRole.TOOL and msg.tool_call_id is not None:
            seen_tool_replies.add(msg.tool_call_id)

    # Invariant 2: every assistant.tool_calls id is matched by a tool reply in preserve
    unmatched_calls = expected_tool_call_ids - seen_tool_replies
    assert (
        not unmatched_calls
    ), f"preserve contains assistant tool_calls without matching tool replies: {unmatched_calls}"

    # Invariant 3: every TOOL message in preserve has a parent assistant tool_call in preserve
    unmatched_replies = seen_tool_replies - expected_tool_call_ids
    assert (
        not unmatched_replies
    ), f"preserve contains TOOL messages without parent assistant tool_calls: {unmatched_replies}"


def _sequential_fc_history() -> list[Message]:
    """Single-tool FC chain: user → assistant(call A) → tool(A) → assistant(call B) → tool(B) → final."""
    return [
        Message(role=MessageRole.USER, content="What is 2+2 and then 3+3?"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: calc",
            tool_calls=[
                {"id": "s_A", "type": "function", "function": {"name": "calc", "arguments": '{"x": "2+2"}'}},
            ],
        ),
        Message(role=MessageRole.TOOL, content="4", tool_call_id="s_A", name="calc"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: calc",
            tool_calls=[
                {"id": "s_B", "type": "function", "function": {"name": "calc", "arguments": '{"x": "3+3"}'}},
            ],
        ),
        Message(role=MessageRole.TOOL, content="6", tool_call_id="s_B", name="calc"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: provide_final_answer",
            tool_calls=[
                {
                    "id": "s_FA",
                    "type": "function",
                    "function": {"name": "provide_final_answer", "arguments": '{"answer": "4 and 6"}'},
                },
            ],
        ),
        Message(role=MessageRole.TOOL, content="Acknowledged.", tool_call_id="s_FA", name="provide_final_answer"),
    ]


def _mixed_fa_with_real_tool_history() -> list[Message]:
    """One assistant turn with BOTH a real tool call AND provide_final_answer in parallel.

    Mirrors `_append_assistant_message`: payload contains both, the FA stub
    TOOL is appended immediately, the real tool reply follows after execution.
    """
    return [
        Message(role=MessageRole.USER, content="Search and finish."),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: web_search, provide_final_answer",
            tool_calls=[
                {"id": "m_real", "type": "function", "function": {"name": "web_search", "arguments": '{"q": "x"}'}},
                {
                    "id": "m_fa",
                    "type": "function",
                    "function": {"name": "provide_final_answer", "arguments": '{"answer": "done"}'},
                },
            ],
        ),
        Message(role=MessageRole.TOOL, content="Acknowledged.", tool_call_id="m_fa", name="provide_final_answer"),
        Message(role=MessageRole.TOOL, content='[{"title": "..."}]', tool_call_id="m_real", name="web_search"),
    ]


def _multi_block_fc_history() -> list[Message]:
    """Multiple FC blocks: parallel + single + final, to stress boundary-between-blocks cases."""
    return [
        Message(role=MessageRole.USER, content="Multi-step."),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: t1, t2",
            tool_calls=[
                {"id": "b1_a", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
                {"id": "b1_b", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
            ],
        ),
        Message(role=MessageRole.TOOL, content="r1_a", tool_call_id="b1_a", name="t1"),
        Message(role=MessageRole.TOOL, content="r1_b", tool_call_id="b1_b", name="t2"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: t3",
            tool_calls=[
                {"id": "b2_a", "type": "function", "function": {"name": "t3", "arguments": "{}"}},
            ],
        ),
        Message(role=MessageRole.TOOL, content="r2", tool_call_id="b2_a", name="t3"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Calling: provide_final_answer",
            tool_calls=[
                {
                    "id": "b3_fa",
                    "type": "function",
                    "function": {"name": "provide_final_answer", "arguments": '{"answer": "ok"}'},
                },
            ],
        ),
        Message(role=MessageRole.TOOL, content="Acknowledged.", tool_call_id="b3_fa", name="provide_final_answer"),
    ]


def _assert_protocol_valid(messages: list[Message]) -> None:
    """End-to-end FC protocol check: every `role=TOOL` has a preceding
    `assistant.tool_calls[*].id == tool_call_id`, and that mapping is unique."""
    seen_call_ids: set[str] = set()
    pending: dict[str, int] = {}

    for idx, msg in enumerate(messages):
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc["id"]
                assert tc_id not in seen_call_ids, f"duplicate tool_call_id {tc_id!r} at index {idx}"
                seen_call_ids.add(tc_id)
                pending[tc_id] = idx
        elif msg.role == MessageRole.TOOL:
            tc_id = msg.tool_call_id
            assert tc_id is not None, f"tool message at index {idx} missing tool_call_id"
            assert tc_id in pending, (
                f"tool message at index {idx} with tool_call_id {tc_id!r} has no "
                f"preceding assistant.tool_calls (orphan reply)"
            )
            del pending[tc_id]


def _make_history_manager_agent(messages: list[Message], max_preserved_tokens: int):
    """Build a minimal stub that satisfies HistoryManagerMixin's expectations."""
    from dynamiq.nodes.agents.components.history_manager import HistoryManagerMixin
    from dynamiq.nodes.agents.utils import SummarizationConfig
    from dynamiq.prompts import Prompt

    class _Agent(HistoryManagerMixin):
        def __init__(self):
            self._prompt = Prompt(messages=[Message(role=MessageRole.SYSTEM, content="sys")] + messages)
            self._history_offset = 1
            self.llm = MagicMock()
            self.llm.model = "gpt-4o-mini"
            self.llm.get_token_limit = MagicMock(return_value=128_000)
            self.name = "test-agent"
            self.id = "test-id"
            self.summarization_config = SummarizationConfig(
                enabled=True,
                max_preserved_tokens=max_preserved_tokens,
                max_token_context_length=None,
                context_usage_ratio=0.7,
            )

    return _Agent()


@pytest.mark.parametrize(
    "history_factory, max_preserved_tokens",
    [
        pytest.param(_build_fc_history, 20, id="parallel_tight"),
        pytest.param(_build_fc_history, 40, id="parallel_mid"),
        pytest.param(_build_fc_history, 200, id="parallel_loose"),
        pytest.param(_sequential_fc_history, 20, id="sequential_tight"),
        pytest.param(_sequential_fc_history, 40, id="sequential_mid"),
        pytest.param(_sequential_fc_history, 200, id="sequential_loose"),
        pytest.param(_mixed_fa_with_real_tool_history, 20, id="mixed_fa_tight"),
        pytest.param(_mixed_fa_with_real_tool_history, 200, id="mixed_fa_loose"),
        pytest.param(_multi_block_fc_history, 15, id="multi_block_tight"),
        pytest.param(_multi_block_fc_history, 60, id="multi_block_mid"),
        pytest.param(_multi_block_fc_history, 100, id="multi_block_loose"),
        pytest.param(_multi_block_fc_history, 1000, id="multi_block_fits"),
    ],
)
def test_fc_summarization_preserves_protocol_across_scenarios(history_factory, max_preserved_tokens):
    """Sweep FC patterns × budgets and assert the compacted history is protocol-valid."""
    agent = _make_history_manager_agent(history_factory(), max_preserved_tokens)

    _, to_preserve = agent._split_history()
    agent._compact_history(summary="<summary>", preserved=to_preserve)

    final_messages = agent._prompt.messages

    assert final_messages and final_messages[0].role == MessageRole.SYSTEM, "system prefix lost"

    _assert_protocol_valid(final_messages[1:])

    if len(final_messages) > 1:
        assert (
            final_messages[1].role != MessageRole.TOOL
        ), "message immediately after system prefix is role=TOOL (orphan)"
