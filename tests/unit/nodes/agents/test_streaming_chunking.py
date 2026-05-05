"""Unit tests for ``StreamingConfig.min_chunk_chars`` chunking behaviour
in ``AgentStreamingParserCallback``.

Mocked-LLM, drip-fed-character-by-character tests covering every inference
mode and three chunk sizes (0/16/64). No real LLM, no creds.

Three scenarios are exercised:
  * Final answer path  — REASONING + ANSWER (all 4 inference modes)
  * Tool-call path     — REASONING + TOOL_INPUT (XML, SO, FC)
  * Parallel tool calls — FC only (multiple tool_calls indices)
"""

from unittest.mock import MagicMock

import pytest

from dynamiq.callbacks.streaming import AgentStreamingParserCallback, InferenceMode, StreamingState
from dynamiq.types.streaming import StreamingMode

SAMPLE_THOUGHT = "I will respond directly with a friendly greeting that demonstrates the chunking buffer in action."
SAMPLE_ANSWER = "Hello there! This is a sufficiently long answer to exercise the chunking buffer end-to-end."
SAMPLE_TOOL_NAME = "search_tool"
SAMPLE_TOOL_INPUT = "Find information about the chunking buffer behavior end-to-end."


# ---------------------------------------------------------------------------
# Callback factory + assertion helpers
# ---------------------------------------------------------------------------


def _make_callback_for_mode(mode: InferenceMode, min_chunk_chars: int = 0):
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.stream_tool_input = None
    agent.streaming.min_chunk_chars = min_chunk_chars
    agent.inference_mode.name = mode.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = None
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    agent.llm = MagicMock()
    agent.llm.id = "llm-1"
    return AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)


def _emitted_by_step(cb, step) -> list[str]:
    contents = []
    for c in cb.agent.stream_content.call_args_list:
        if c.kwargs.get("step") != step:
            continue
        content = c.kwargs.get("content")
        if step == StreamingState.REASONING and isinstance(content, dict):
            contents.append(content.get("thought", ""))
        elif step == StreamingState.ANSWER and isinstance(content, str):
            contents.append(content)
    return contents


def _emitted_tool_input_chunks(cb) -> list[str]:
    """action_input fragment from every TOOL_INPUT delta event."""
    contents = []
    for c in cb.agent.stream_content.call_args_list:
        if c.kwargs.get("step") != StreamingState.TOOL_INPUT:
            continue
        content = c.kwargs.get("content")
        if isinstance(content, dict):
            fragment = content.get("action_input")
            if isinstance(fragment, str):
                contents.append(fragment)
    return contents


def _has_tool_input_start(cb) -> bool:
    return any(c.kwargs.get("step") == StreamingState.TOOL_INPUT_START for c in cb.agent.stream_content.call_args_list)


def _tool_input_events_by_run_id(cb) -> dict:
    """{tool_run_id: [action_input_fragment, ...]} preserving order."""
    grouped: dict = {}
    for c in cb.agent.stream_content.call_args_list:
        if c.kwargs.get("step") != StreamingState.TOOL_INPUT:
            continue
        content = c.kwargs.get("content")
        if not isinstance(content, dict):
            continue
        tid = content.get("tool_run_id")
        fragment = content.get("action_input")
        if isinstance(tid, str) and isinstance(fragment, str):
            grouped.setdefault(tid, []).append(fragment)
    return grouped


def _assert_chunking(reasoning: list[str], answers: list[str], min_chunk_chars: int) -> None:
    """Per-step: every chunk except the final flush must be >= min_chunk_chars."""
    if min_chunk_chars <= 0:
        return
    for chunk in reasoning[:-1]:
        assert len(chunk) >= min_chunk_chars, f"REASONING chunk {len(chunk)} < {min_chunk_chars}"
    for chunk in answers[:-1]:
        assert len(chunk) >= min_chunk_chars, f"ANSWER chunk {len(chunk)} < {min_chunk_chars}"


# ---------------------------------------------------------------------------
# Mocked-LLM payload builders + chunk feeders
# ---------------------------------------------------------------------------


def _payload_for_mode(mode: InferenceMode, thought: str, answer: str) -> str | None:
    if mode == InferenceMode.STRUCTURED_OUTPUT:
        return '{"thought": "' + thought + '", "action": "finish", "action_input": "' + answer + '"}'
    if mode == InferenceMode.XML:
        return f"<thought>{thought}</thought><answer>{answer}</answer>"
    if mode == InferenceMode.DEFAULT:
        return f"Thought: {thought}\nAnswer: {answer}"
    return None


def _tool_payload_for_mode(mode: InferenceMode, thought: str, tool_name: str, tool_input: str) -> str | None:
    if mode == InferenceMode.STRUCTURED_OUTPUT:
        return '{"thought": "' + thought + '", "action": "' + tool_name + '", "action_input": "' + tool_input + '"}'
    if mode == InferenceMode.XML:
        return (
            f"<thought>{thought}</thought>" f"<action>{tool_name}</action>" f"<action_input>{tool_input}</action_input>"
        )
    return None


def _feed_text_chunks(cb, payload: str) -> None:
    """Drip-feed a delta.content payload character-by-character."""
    serialized = {"group": "llms", "id": "llm-1"}
    for ch in payload:
        cb.on_node_execute_stream(serialized, {"choices": [{"delta": {"content": ch}}]})


def _feed_fc_chunks(cb, thought: str, answer: str) -> None:
    """FC final answer: function_name=provide_final_answer + arguments JSON."""
    serialized = {"group": "llms", "id": "llm-1"}
    cb.on_node_execute_stream(
        serialized,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "type": "function", "function": {"name": "provide_final_answer"}}]
                    }
                }
            ]
        },
    )
    args = '{"thought": "' + thought + '", "answer": "' + answer + '"}'
    for ch in args:
        cb.on_node_execute_stream(
            serialized,
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": ch}}]}}]},
        )


def _feed_fc_tool_chunks(cb, thought: str, tool_name: str, tool_input: str) -> None:
    """FC single tool call: function name = tool, arguments contain thought + action_input."""
    serialized = {"group": "llms", "id": "llm-1"}
    cb.on_node_execute_stream(
        serialized,
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"name": tool_name}}]}}]},
    )
    args = '{"thought": "' + thought + '", "action_input": "' + tool_input + '"}'
    for ch in args:
        cb.on_node_execute_stream(
            serialized,
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": ch}}]}}]},
        )


def _feed_fc_parallel_tool_chunks(cb, calls: list) -> None:
    """Multiple FC tool calls. Each ``calls`` entry is (tool_name, thought, action_input).
    The tc_index increments per tool, which triggers the parser's _reset_tool_call_state."""
    serialized = {"group": "llms", "id": "llm-1"}
    for index, (tool_name, thought, action_input) in enumerate(calls):
        cb.on_node_execute_stream(
            serialized,
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": index, "type": "function", "function": {"name": tool_name}}]}}
                ]
            },
        )
        args = '{"thought": "' + thought + '", "action_input": "' + action_input + '"}'
        for ch in args:
            cb.on_node_execute_stream(
                serialized,
                {
                    "choices": [
                        {"delta": {"tool_calls": [{"index": index, "type": "function", "function": {"arguments": ch}}]}}
                    ]
                },
            )


# ---------------------------------------------------------------------------
# Final-answer path: REASONING + ANSWER, all 4 inference modes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_chunk_chars", [0, 16, 64], ids=lambda n: f"chunk{n}")
@pytest.mark.parametrize(
    "mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
        InferenceMode.STRUCTURED_OUTPUT,
        InferenceMode.FUNCTION_CALLING,
    ],
    ids=lambda m: m.value.lower(),
)
def test_streaming_chunking_for_all_modes(mode, min_chunk_chars):
    cb = _make_callback_for_mode(mode, min_chunk_chars=min_chunk_chars)

    if mode == InferenceMode.FUNCTION_CALLING:
        _feed_fc_chunks(cb, SAMPLE_THOUGHT, SAMPLE_ANSWER)
    else:
        _feed_text_chunks(cb, _payload_for_mode(mode, SAMPLE_THOUGHT, SAMPLE_ANSWER))
    cb.on_node_execute_end({"group": "llms"}, output_data={})

    reasoning = _emitted_by_step(cb, StreamingState.REASONING)
    answers = _emitted_by_step(cb, StreamingState.ANSWER)

    assert reasoning, f"{mode.value}/chunk{min_chunk_chars}: no REASONING events"
    assert answers, f"{mode.value}/chunk{min_chunk_chars}: no ANSWER events"

    _assert_chunking(reasoning, answers, min_chunk_chars)

    # Content must round-trip (modulo whitespace from mode-specific delimiters).
    assert "".join(reasoning).strip() == SAMPLE_THOUGHT.strip()
    assert "".join(answers).strip() == SAMPLE_ANSWER.strip()


# ---------------------------------------------------------------------------
# Tool-call path: REASONING + TOOL_INPUT.
# DEFAULT mode parser does not stream TOOL_INPUT, so it's excluded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_chunk_chars", [0, 16, 64], ids=lambda n: f"chunk{n}")
@pytest.mark.parametrize(
    "mode",
    [
        InferenceMode.XML,
        InferenceMode.STRUCTURED_OUTPUT,
        InferenceMode.FUNCTION_CALLING,
    ],
    ids=lambda m: m.value.lower(),
)
def test_streaming_chunking_tool_call_path(mode, min_chunk_chars):
    cb = _make_callback_for_mode(mode, min_chunk_chars=min_chunk_chars)

    if mode == InferenceMode.FUNCTION_CALLING:
        _feed_fc_tool_chunks(cb, SAMPLE_THOUGHT, SAMPLE_TOOL_NAME, SAMPLE_TOOL_INPUT)
    else:
        _feed_text_chunks(cb, _tool_payload_for_mode(mode, SAMPLE_THOUGHT, SAMPLE_TOOL_NAME, SAMPLE_TOOL_INPUT))
    cb.on_node_execute_end({"group": "llms"}, output_data={})

    reasoning = _emitted_by_step(cb, StreamingState.REASONING)
    tool_inputs = _emitted_tool_input_chunks(cb)

    assert reasoning, f"{mode.value}/chunk{min_chunk_chars}: no REASONING events"
    assert tool_inputs, f"{mode.value}/chunk{min_chunk_chars}: no TOOL_INPUT events"
    assert _has_tool_input_start(cb), f"{mode.value}/chunk{min_chunk_chars}: missing tool_input_start event"

    if min_chunk_chars > 0:
        for chunk in reasoning[:-1]:
            assert len(chunk) >= min_chunk_chars, f"REASONING chunk {len(chunk)} < {min_chunk_chars}"
        for chunk in tool_inputs[:-1]:
            assert len(chunk) >= min_chunk_chars, f"TOOL_INPUT chunk {len(chunk)} < {min_chunk_chars}"

    assert "".join(reasoning).strip() == SAMPLE_THOUGHT.strip()
    assert "".join(tool_inputs).strip() == SAMPLE_TOOL_INPUT.strip()

    # No final-answer events on this path.
    assert not _emitted_by_step(
        cb, StreamingState.ANSWER
    ), f"{mode.value}/chunk{min_chunk_chars}: ANSWER emitted on tool-call path"


# ---------------------------------------------------------------------------
# FC parallel tool calls: each tool gets its own _streaming_tool_run_id and
# its own per-tool flush at the boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_chunk_chars", [0, 16, 64], ids=lambda n: f"chunk{n}")
def test_streaming_chunking_parallel_tool_calls(min_chunk_chars):
    cb = _make_callback_for_mode(InferenceMode.FUNCTION_CALLING, min_chunk_chars=min_chunk_chars)

    thought_a = "I should look up cat facts to answer the question correctly and thoroughly."
    input_a = "Cat facts about sleeping habits, behavior, and lifestyle of common breeds."
    thought_b = "I will also need information about dogs to provide a balanced comparative answer."
    input_b = "Dog facts about sense of smell, training, and breed differences."

    _feed_fc_parallel_tool_chunks(
        cb,
        [
            ("CatFacts", thought_a, input_a),
            ("DogFacts", thought_b, input_b),
        ],
    )
    cb.on_node_execute_end({"group": "llms"}, output_data={})

    starts = [
        c for c in cb.agent.stream_content.call_args_list if c.kwargs.get("step") == StreamingState.TOOL_INPUT_START
    ]
    assert len(starts) == 2, f"expected 2 tool_input_start events, got {len(starts)}"

    by_run_id = _tool_input_events_by_run_id(cb)
    assert len(by_run_id) == 2, f"expected 2 distinct tool_run_ids, got {len(by_run_id)}"

    fragments_per_tool = list(by_run_id.values())
    assert "".join(fragments_per_tool[0]) == input_a
    assert "".join(fragments_per_tool[1]) == input_b

    if min_chunk_chars > 0:
        for tid, frags in by_run_id.items():
            for chunk in frags[:-1]:
                assert len(chunk) >= min_chunk_chars, f"tool_run_id={tid} chunk {len(chunk)} < {min_chunk_chars}"

    reasoning = _emitted_by_step(cb, StreamingState.REASONING)
    assert reasoning, "no REASONING events for parallel tool calls"
    assert "".join(reasoning).strip() == (thought_a + thought_b).strip()
