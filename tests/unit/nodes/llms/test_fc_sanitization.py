"""Tests for BaseLLM._sanitize_fc_messages."""
from unittest.mock import patch

import litellm
import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig


class TestSanitizeFCMessages:
    def test_orphan_tool_call_is_repaired(self):
        """Orphan tool_call gets a synthetic reply inserted."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_orphan",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "Did you check?"},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        assert len(out) == 4
        tool_reply = out[2]
        assert tool_reply["role"] == "tool"
        assert tool_reply["tool_call_id"] == "call_orphan"
        assert tool_reply["content"]

    def test_orphan_tool_reply_is_dropped(self):
        """Tool reply with unknown id is dropped."""
        messages = [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "tool_call_id": "unknown_id", "content": "sunny"},
            {"role": "user", "content": "Thanks"},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        assert len(out) == 2
        assert all(m["role"] != "tool" for m in out)

    def test_empty_content_is_filled_with_placeholder(self):
        """Empty assistant content is replaced with a placeholder."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Are you there?"},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        assert len(out) == 3
        assert out[1]["role"] == "assistant"
        assert out[1]["content"] != ""

    def test_well_formed_fc_pair_passes_through(self):
        """Well-formed FC pair is not structurally altered."""
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Calling: get_weather",
                "tool_calls": [
                    {
                        "id": "call_ok",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_ok", "content": "sunny"},
            {"role": "assistant", "content": "It's sunny."},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        assert len(out) == 4
        assert out[1]["tool_calls"][0]["id"] == "call_ok"
        assert out[2]["tool_call_id"] == "call_ok"

    def test_empty_messages_returns_empty(self):
        assert BaseLLM._sanitize_fc_messages([]) == []

    def test_mismatched_ids(self):
        """Mismatched ids: A gets a synthetic reply, B is dropped."""
        messages = [
            {"role": "user", "content": "?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "A", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "B", "content": "wrong id"},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        tool_replies = [m for m in out if m.get("role") == "tool"]
        assert [m["tool_call_id"] for m in tool_replies] == ["A"]

    def test_works_even_when_global_flag_is_false(self):
        """Helper must not depend on litellm.modify_params being True."""
        starting = litellm.modify_params
        try:
            litellm.modify_params = False
            messages = [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                    ],
                },
                {"role": "user", "content": "?"},
            ]
            out = BaseLLM._sanitize_fc_messages(messages)
            assert len(out) == 4
            assert any(m.get("tool_call_id") == "call_x" for m in out)
        finally:
            litellm.modify_params = starting

    def test_non_tool_message_between_assistant_and_replies_does_not_duplicate(self):
        """Regression: a non-tool message between the assistant and its tool
        replies must not cause already-consumed replies to be re-emitted as
        duplicates that escape `_dedupe_tool_results`.
        """
        messages = [
            {"role": "user", "content": "do A, B, C"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": "c", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "interjection"},
            {"role": "tool", "tool_call_id": "call_1", "content": "A done"},
            {"role": "user", "content": "another interjection"},
            {"role": "tool", "tool_call_id": "call_2", "content": "B done"},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
        assert len(ids) == len(set(ids)), f"duplicate tool_call_ids in output: {ids}"
        assert set(ids) == {"call_1", "call_2", "call_3"}

    def test_duplicate_tool_results_in_same_block_are_deduped_to_latest(self):
        """Case D: Anthropic rejects two tool_results for the same tool_call_id.

        Repros a session-resume scenario where history replay produces two tool
        replies for the same tool_call_id within one block. Without dedup the
        provider returns 400 ("each tool_use must have a single result"). With
        dedup only the latest reply survives.
        """
        messages = [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "Calling: f",
                "tool_calls": [{"id": "call_dup", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_dup", "content": "stale result"},
            {"role": "tool", "tool_call_id": "call_dup", "content": "latest result"},
            {"role": "assistant", "content": "done."},
        ]

        out = BaseLLM._sanitize_fc_messages(messages)

        tool_replies = [m for m in out if m.get("role") == "tool" and m.get("tool_call_id") == "call_dup"]
        assert len(tool_replies) == 1, f"expected one tool reply after dedup, got {len(tool_replies)}"
        assert tool_replies[0]["content"] == "latest result", "dedup should keep the last occurrence, not the first"


@pytest.fixture
def llm():
    return OpenAI(
        model="gpt-4o-mini",
        connection=OpenAIConnection(api_key="test-key"),
    )


def _orphan_payload():
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_orphan",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "?"},
    ]


class TestSanitizationWiring:
    def test_does_not_sanitize_when_no_tools(self, llm):
        """Plain completions bypass sanitization."""
        config = RunnableConfig()

        original = _orphan_payload()
        params = llm._build_completion_params(
            messages=original,
            config=config,
            prompt=Prompt(messages=[]),
            tools=None,
        )

        sent = params["messages"]
        assert sent == original

    def test_sanitizer_invoked_via_spy(self, llm):
        """Spy confirms the helper is invoked."""
        config = RunnableConfig()
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "x", "parameters": {"type": "object", "properties": {}}},
            }
        ]

        with patch.object(BaseLLM, "_sanitize_fc_messages", wraps=BaseLLM._sanitize_fc_messages) as spy:
            llm._build_completion_params(
                messages=[{"role": "user", "content": "hi"}],
                config=config,
                prompt=Prompt(messages=[]),
                tools=tools,
            )
            assert spy.called

    def test_sanitizer_not_invoked_when_no_tools(self, llm):
        config = RunnableConfig()

        with patch.object(BaseLLM, "_sanitize_fc_messages") as spy:
            llm._build_completion_params(
                messages=[{"role": "user", "content": "hi"}],
                config=config,
                prompt=Prompt(messages=[]),
                tools=None,
            )
            assert not spy.called
