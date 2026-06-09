"""Live test for ``JSONInnerThoughtsExtractor`` against a real OpenAI model.

Drives a real FC stream through ``AgentStreamingParserCallback`` — the same
``on_node_execute_stream`` path the production pipeline uses — and asserts:

  * REASONING events accumulate to the model's thought text (non-empty).
  * TOOL_INPUT events accumulate into JSON args with the deterministic
    ``message`` we asked for, with ``thought`` correctly stripped.

This exercises the full inline-thought split inside the streaming layer end
to end, not just the extractor in isolation.
"""

import json
import os
from unittest.mock import MagicMock

import litellm
import pytest

from dynamiq.callbacks.streaming import AgentStreamingParserCallback, StreamingState
from dynamiq.nodes.types import InferenceMode
from dynamiq.types.streaming import StreamingMode

pytestmark = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")


EXPECTED_MESSAGE = "hello world from inner-thoughts test"
LLM_ID = "llm-1"


def _make_fc_callback() -> AgentStreamingParserCallback:
    """Build the same callback the production pipeline wires up, but with a
    mock agent so we can inspect emissions directly via ``call_args_list``."""
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.stream_tool_input = None
    agent.streaming.min_chunk_chars = 0
    agent.inference_mode.name = InferenceMode.FUNCTION_CALLING.value
    agent.name = "live-extractor-agent"
    agent._streaming_tool_run_id = None
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    agent.llm = MagicMock()
    agent.llm.id = LLM_ID
    return AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)


def _collect_reasoning(cb) -> list[str]:
    """REASONING chunks are wrapped as ``{"thought": "<fragment>"}``."""
    out: list[str] = []
    for c in cb.agent.stream_content.call_args_list:
        if c.kwargs.get("step") != StreamingState.REASONING:
            continue
        content = c.kwargs.get("content")
        if isinstance(content, dict) and isinstance(content.get("thought"), str):
            out.append(content["thought"])
    return out


def _collect_tool_input(cb) -> list[str]:
    """TOOL_INPUT chunks are dicts with ``action_input`` string fragments."""
    out: list[str] = []
    for c in cb.agent.stream_content.call_args_list:
        if c.kwargs.get("step") != StreamingState.TOOL_INPUT:
            continue
        content = c.kwargs.get("content")
        if isinstance(content, dict) and isinstance(content.get("action_input"), str):
            out.append(content["action_input"])
    return out


def test_extractor_emits_correct_reasoning_and_tool_input_via_callback():
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the given message verbatim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning about using this tool.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to echo back.",
                    },
                },
                "required": ["thought", "message"],
                "additionalProperties": False,
            },
        },
    }

    stream = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Call the `echo` tool. The `message` field MUST be exactly "
                    f"'{EXPECTED_MESSAGE}' (verbatim, no extra words). Use the "
                    f"`thought` field to briefly explain why you're calling it."
                ),
            }
        ],
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "echo"}},
        stream=True,
        temperature=0,
        max_tokens=200,
    )

    cb = _make_fc_callback()
    serialized = {"group": "llms", "id": LLM_ID}

    for chunk in stream:
        cb.on_node_execute_stream(serialized, chunk.model_dump())
    cb.on_node_execute_end(serialized, output_data={})

    reasoning_chunks = _collect_reasoning(cb)
    tool_input_chunks = _collect_tool_input(cb)

    assert reasoning_chunks, "no REASONING events were emitted"
    assert tool_input_chunks, "no TOOL_INPUT events were emitted"

    accumulated_thought = "".join(reasoning_chunks).strip()
    assert accumulated_thought, "REASONING channel reassembled to empty string"

    accumulated_tool_input = "".join(tool_input_chunks)
    parsed = json.loads(accumulated_tool_input)
    assert "thought" not in parsed, f"thought leaked into TOOL_INPUT: {parsed}"
    assert parsed.get("message") == EXPECTED_MESSAGE, (
        f"message mismatch — expected {EXPECTED_MESSAGE!r}, " f"got {parsed.get('message')!r} (full payload: {parsed})"
    )
