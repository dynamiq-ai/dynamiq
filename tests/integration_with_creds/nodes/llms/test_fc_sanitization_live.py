"""Live FC-sanitization test against a real OpenAI model.

Sends a payload with an orphan assistant tool_call (no matching tool reply) and
verifies:
  1. the sanitizer rewrites the outbound payload into a valid FC pair
  2. the request succeeds end-to-end (model returns content)
"""
import os
from unittest.mock import patch

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus


pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def test_orphan_tool_call_is_repaired_live(capsys):
    prompt = Prompt(
        messages=[
            Message(role=MessageRole.USER, content="What's the weather in Paris?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    {
                        "id": "call_orphan_live",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            ),
            Message(
                role=MessageRole.USER,
                content="Never mind. Just say 'hello world' and stop.",
            ),
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    llm = OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0,
        prompt=prompt,
    )

    sanitized_holder: list[list[dict]] = []

    original = BaseLLM._sanitize_fc_messages

    def capturing_sanitize(messages):
        out = original(messages)
        sanitized_holder.append(out)
        return out

    with patch.object(BaseLLM, "_sanitize_fc_messages", side_effect=capturing_sanitize):
        result = llm.run(input_data={}, config=RunnableConfig())

    # --- 1. Sanitizer was invoked and produced a valid FC pair ---
    assert sanitized_holder, "Sanitizer was never called"
    sanitized = sanitized_holder[-1]
    print("\n--- sanitized outbound payload ---")
    for i, m in enumerate(sanitized):
        role = m.get("role")
        tc = [t["id"] for t in m.get("tool_calls") or []]
        tcid = m.get("tool_call_id")
        content = (m.get("content") or "")[:80].replace("\n", " ")
        print(f"  [{i}] {role:9s} tool_calls={tc} tool_call_id={tcid} content={content!r}")

    tool_replies = [m for m in sanitized if m.get("role") == "tool"]
    assert any(
        m.get("tool_call_id") == "call_orphan_live" for m in tool_replies
    ), "Sanitizer did not insert a synthetic tool reply for the orphan call"

    # --- 2. End-to-end call succeeded ---
    assert result.status == RunnableStatus.SUCCESS, f"Expected SUCCESS, got: {result.output}"
    content = result.output.get("content") or ""
    assert content, "Expected non-empty response content"
