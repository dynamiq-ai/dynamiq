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
