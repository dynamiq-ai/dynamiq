"""Escape handling in ``AgentStreamingParserCallback``.

One rule, asserted in both directions: text the client **renders** must arrive decoded,
and text the client **re-parses** must arrive byte-identical.

The UI renders ``thought`` and the answer directly (``thinkingDelta += content.thought``)
but ``JSON.parse``s accumulated ``action_input``. So a thought carrying ``\\"`` must reach
it as ``"``, while the same bytes inside tool arguments must stay escaped or the client's
parse fails.

Payloads are built with ``json.dumps`` — byte-for-byte what a provider puts on the wire —
and fed one character per chunk, which splits every escape across a chunk boundary.
"""

import json
from unittest.mock import MagicMock

import pytest

from dynamiq.callbacks.streaming import AgentStreamingParserCallback, InferenceMode, StreamingState
from dynamiq.types.streaming import StreamingMode

SERIALIZED = {"group": "llms", "id": "llm-1"}

# A quote, a newline and a literal backslash — the cases from the bug report.
THOUGHT = 'The request "make work." is too vague.\nNo deliverable was specified.'
ANSWER = 'Done: I wrote "report.md".\nIt uses a \\ separator.'
QUERY = 'a "quoted" value with a \\ backslash'


def _make_callback(mode: InferenceMode, min_chunk_chars: int = 0):
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.stream_tool_input = None
    agent.streaming.min_chunk_chars = min_chunk_chars
    agent.streaming.fc_wait_for_first_key = True  # the StreamingConfig default
    agent.inference_mode.name = mode.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = None
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    agent.llm = MagicMock()
    agent.llm.id = "llm-1"
    return AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)


def _streamed(cb, step) -> str:
    """Concatenate every event for ``step`` — what the client would accumulate."""
    parts = []
    for call in cb.agent.stream_content.call_args_list:
        if call.kwargs.get("step") != step:
            continue
        content = call.kwargs.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, dict):
            fragment = content.get("thought") if step == StreamingState.REASONING else content.get("action_input")
            if isinstance(fragment, str):
                parts.append(fragment)
    return "".join(parts)


def _feed_text(cb, payload: str) -> None:
    for char in payload:
        cb.on_node_execute_stream(SERIALIZED, {"choices": [{"delta": {"content": char}}]})
    cb.on_node_execute_end({"group": "llms"}, output_data={})


def _feed_fc(cb, tool_name: str, args: str, fragments: list[str] | None = None) -> None:
    """A tool call: the name arrives first, then the argument fragments."""
    cb.on_node_execute_stream(
        SERIALIZED,
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"name": tool_name}}]}}]},
    )
    for fragment in fragments if fragments is not None else args:
        cb.on_node_execute_stream(
            SERIALIZED,
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "type": "function", "function": {"arguments": fragment}}]}}
                ]
            },
        )
    cb.on_node_execute_end({"group": "llms"}, output_data={})


# ---------------------------------------------------------------------------
# Rendered text must be decoded.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_chunk_chars", [0, 16, 64], ids=lambda n: f"chunk{n}")
def test_fc_tool_call_thought_is_decoded(min_chunk_chars):
    """The reported bug. Parametrized because ``min_chunk_chars > 0`` re-enters ``_emit``
    via ``_flush_chunk_buffer``, which would double-feed a decoder placed inside it."""
    cb = _make_callback(InferenceMode.FUNCTION_CALLING, min_chunk_chars=min_chunk_chars)
    _feed_fc(cb, "search", json.dumps({"thought": THOUGHT, "query": QUERY}))

    assert _streamed(cb, StreamingState.REASONING) == THOUGHT


def test_fc_final_answer_is_decoded():
    """The answer path runs ``_emit_json_field_content`` with ``_trim_buffer`` active — a
    combination STRUCTURED_OUTPUT never reaches, since it skips trimming. The answer is
    long enough to force a trim, which rebases the slice indices mid-field."""
    long_answer = ANSWER + " " + "padding to force a buffer trim. " * 20
    cb = _make_callback(InferenceMode.FUNCTION_CALLING)
    _feed_fc(cb, "provide_final_answer", json.dumps({"thought": THOUGHT, "answer": long_answer}))

    assert _streamed(cb, StreamingState.REASONING) == THOUGHT
    assert _streamed(cb, StreamingState.ANSWER) == long_answer


def test_structured_output_thought_and_answer_are_decoded():
    cb = _make_callback(InferenceMode.STRUCTURED_OUTPUT)
    _feed_text(cb, json.dumps({"thought": THOUGHT, "action": "finish", "action_input": ANSWER}))

    assert _streamed(cb, StreamingState.REASONING) == THOUGHT
    assert _streamed(cb, StreamingState.ANSWER) == ANSWER


# ---------------------------------------------------------------------------
# Re-parsed text must survive byte-for-byte.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("min_chunk_chars", [0, 16, 64], ids=lambda n: f"chunk{n}")
def test_fc_tool_arguments_stay_valid_json(min_chunk_chars):
    """Decoding tool args would unescape their interior quotes and break the client's parse."""
    cb = _make_callback(InferenceMode.FUNCTION_CALLING, min_chunk_chars=min_chunk_chars)
    _feed_fc(cb, "search", json.dumps({"thought": THOUGHT, "query": QUERY}))

    assert json.loads(_streamed(cb, StreamingState.TOOL_INPUT)) == {"query": QUERY}


def test_structured_output_tool_input_is_unwrapped_to_parseable_json():
    """SO types ``action_input`` as a string, so the tool's arguments are JSON-encoded a
    second time to fit. That wrapper is not part of the payload and the client cannot strip
    it — the top-level object never reaches it — so it comes off here, leaving the JSON the
    client parses with its own escapes intact."""
    args = {"brief": "notes.md", "content": 'a "quoted" line'}
    cb = _make_callback(InferenceMode.STRUCTURED_OUTPUT)
    _feed_text(cb, json.dumps({"thought": "t", "action": "file_write", "action_input": json.dumps(args)}))

    streamed = _streamed(cb, StreamingState.TOOL_INPUT)
    assert json.loads(streamed) == args
    assert '\\"' in streamed, "the payload's own escaping must survive"


def test_structured_output_plain_text_tool_input_is_decoded():
    """Not every ``action_input`` is JSON — a bare query is rendered, so it needs the
    wrapper off too."""
    cb = _make_callback(InferenceMode.STRUCTURED_OUTPUT)
    _feed_text(cb, json.dumps({"thought": "t", "action": "search", "action_input": 'find "weather"'}))

    assert _streamed(cb, StreamingState.TOOL_INPUT) == 'find "weather"'


def test_tool_input_parses_identically_in_both_modes():
    """The invariant that pins the whole rule: the same logical tool call must reach the
    client as the same arguments, whichever mode produced it. FC nests the arguments
    structurally and SO nests them textually, but that is transport, not payload."""
    args = {"brief": "notes.md", "content": 'a "quoted" line'}

    fc = _make_callback(InferenceMode.FUNCTION_CALLING)
    _feed_fc(fc, "file_write", json.dumps({"thought": THOUGHT, **args}))

    so = _make_callback(InferenceMode.STRUCTURED_OUTPUT)
    _feed_text(so, json.dumps({"thought": THOUGHT, "action": "file_write", "action_input": json.dumps(args)}))

    fc_input = json.loads(_streamed(fc, StreamingState.TOOL_INPUT))
    so_input = json.loads(_streamed(so, StreamingState.TOOL_INPUT))
    assert fc_input == so_input == args
    assert _streamed(fc, StreamingState.REASONING) == _streamed(so, StreamingState.REASONING) == THOUGHT


def test_fc_object_answer_is_not_decoded():
    """A brace-delimited answer is a JSON object the client parses, not prose."""
    cb = _make_callback(InferenceMode.FUNCTION_CALLING)
    cb._answer_started = True
    cb._buffer = '{"answer": {"foo": "a \\"b\\""}}'
    cb._process_json_mode(final_answer_only=False)

    assert cb._fc_object_answer is True
    assert _streamed(cb, StreamingState.ANSWER) == '{"foo": "a \\"b\\""}'


@pytest.mark.parametrize("mode", [InferenceMode.XML, InferenceMode.DEFAULT], ids=lambda m: m.value.lower())
def test_plain_text_modes_keep_real_backslashes(mode):
    """XML and DEFAULT never carry JSON source, so decoding would eat genuine backslashes."""
    answer = r"Run C:\temp\new then \\server\share"
    payload = (
        f"<thought>thinking</thought><answer>{answer}</answer>"
        if mode == InferenceMode.XML
        else f"Thought: thinking\nAnswer: {answer}"
    )
    cb = _make_callback(mode)
    _feed_text(cb, payload)

    assert _streamed(cb, StreamingState.ANSWER).strip() == answer


# ---------------------------------------------------------------------------
# Decoder state across chunk boundaries — why a streaming decoder is needed.
# ---------------------------------------------------------------------------


def test_escape_split_across_chunk_boundary():
    """A ``\\`` in one chunk and the character it escapes in the next must still resolve."""
    args = json.dumps({"thought": 'say "hi"\nbye', "query": "q"})
    fragments, index = [], 0
    while index < len(args):
        if args[index] == "\\":
            fragments.append(args[index])  # the backslash alone ends this chunk
            fragments.append(args[index + 1])
            index += 2
        else:
            fragments.append(args[index])
            index += 1

    cb = _make_callback(InferenceMode.FUNCTION_CALLING)
    _feed_fc(cb, "search", args, fragments=fragments)

    assert _streamed(cb, StreamingState.REASONING) == 'say "hi"\nbye'


def test_surrogate_pair_split_across_chunk_boundary():
    """The halves of a ``\\uD83D\\uDE00`` pair can arrive in different chunks."""
    head, tail = '{"thought": "hi \\ud8', '3d\\ude00 there", "query": "q"}'

    cb = _make_callback(InferenceMode.FUNCTION_CALLING)
    _feed_fc(cb, "search", head + tail, fragments=[head, tail])

    thought = _streamed(cb, StreamingState.REASONING)
    assert thought == "hi \U0001f600 there"
    thought.encode("utf-8")  # a lone surrogate would raise here


def test_lone_surrogate_degrades_to_replacement_char():
    """A truncated pair must not produce a string that fails to UTF-8 encode."""
    cb = _make_callback(InferenceMode.FUNCTION_CALLING)
    _feed_fc(cb, "search", '{"thought": "oops \\ud83d", "query": "q"}')

    thought = _streamed(cb, StreamingState.REASONING)
    assert thought == "oops \ufffd"
    thought.encode("utf-8")
