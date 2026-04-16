import json
from enum import Enum, auto

from dynamiq.nodes.types import InferenceMode
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger


def collect_streaming_events(streaming_iterator, agent_id):
    """Collect streaming events in chronological order.

    Returns:
        list[tuple[str, Any]]: [(step, content), ...] in the order received.
    """
    ordered_events = []
    raw_events = []

    for event in streaming_iterator:
        raw_events.append(event)

        if event.entity_id != agent_id:
            continue
        data = event.data
        if not isinstance(data, dict):
            continue

        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        step = delta.get("step")
        content = delta.get("content")
        if step is not None:
            ordered_events.append((step, content))

    logger.info(f"Collected {len(ordered_events)} streaming events from {len(raw_events)} raw events")

    return ordered_events


# ---------------------------------------------------------------------------
# FSM states and event classification
# ---------------------------------------------------------------------------


class State(Enum):
    INIT = auto()
    REASONING = auto()
    TOOL_INPUT = auto()
    POST_PARSE = auto()
    TOOL_RESULT = auto()
    ERROR = auto()
    ANSWER = auto()


def _classify_event(step, content):
    """Map a raw (step, content) pair to an FSM event name."""
    if step == "reasoning":
        if isinstance(content, dict) and "tool_run_id" in content:
            return "post_parse_reasoning"
        return "reasoning"
    if step == "tool_input_start":
        return "tool_input_start"
    if step == "tool_input":
        return "tool_input"
    if step == "tool":
        return "tool_result"
    if step == "tool_input_error":
        return "tool_input_error"
    if step == "answer":
        return "answer"
    return None


# ---------------------------------------------------------------------------
# Transition tables
# ---------------------------------------------------------------------------

_TRANSITIONS_WITH_TOOL_INPUT = {
    State.INIT: {
        "reasoning": State.REASONING,
    },
    State.REASONING: {
        "reasoning": State.REASONING,
        "tool_input_start": State.TOOL_INPUT,
        "answer": State.ANSWER,
    },
    State.TOOL_INPUT: {
        "tool_input_start": State.TOOL_INPUT,
        "tool_input": State.TOOL_INPUT,
        "post_parse_reasoning": State.POST_PARSE,
        "tool_input_error": State.ERROR,
    },
    State.POST_PARSE: {
        "post_parse_reasoning": State.POST_PARSE,
        "tool_result": State.TOOL_RESULT,
    },
    State.TOOL_RESULT: {
        "tool_result": State.TOOL_RESULT,
        "reasoning": State.REASONING,
        "answer": State.ANSWER,
    },
    State.ERROR: {
        "tool_input_error": State.ERROR,
        "reasoning": State.REASONING,
        "answer": State.ANSWER,
    },
    State.ANSWER: {
        "answer": State.ANSWER,
    },
}

# FC mode: allows reasoning after tool_input for parallel tool calls.
_TRANSITIONS_FC = {
    **_TRANSITIONS_WITH_TOOL_INPUT,
    State.TOOL_INPUT: {
        **_TRANSITIONS_WITH_TOOL_INPUT[State.TOOL_INPUT],
        "reasoning": State.REASONING,
    },
}

_TRANSITIONS_DEFAULT = {
    State.INIT: {
        "reasoning": State.REASONING,
        "answer": State.ANSWER,
    },
    State.REASONING: {
        "reasoning": State.REASONING,
        "post_parse_reasoning": State.POST_PARSE,
        "answer": State.ANSWER,
    },
    State.POST_PARSE: {
        "post_parse_reasoning": State.POST_PARSE,
        "tool_result": State.TOOL_RESULT,
    },
    State.TOOL_RESULT: {
        "tool_result": State.TOOL_RESULT,
        "reasoning": State.REASONING,
        "answer": State.ANSWER,
    },
    State.ANSWER: {
        "answer": State.ANSWER,
    },
}

_TRANSITIONS_FINAL = {
    State.INIT: {
        "answer": State.ANSWER,
    },
    State.ANSWER: {
        "answer": State.ANSWER,
    },
}


# ---------------------------------------------------------------------------
# Structural validators (called inline during FSM walk)
# ---------------------------------------------------------------------------


def _validate_post_parse_reasoning(content, idx):
    assert isinstance(content, dict), f"Event {idx}: post-parse reasoning should be dict, got {type(content)}"
    for key in ("thought", "action", "tool", "action_input", "loop_num"):
        assert key in content, f"Event {idx}: post-parse reasoning missing '{key}': {content}"
    tool = content["tool"]
    assert "name" in tool and "type" in tool, f"Event {idx}: post-parse reasoning tool missing name/type: {tool}"


def _validate_tool_result(content, idx):
    assert isinstance(content, dict), f"Event {idx}: tool result should be dict, got {type(content)}"
    for key in ("tool_run_id", "name", "result", "status"):
        assert key in content, f"Event {idx}: tool result missing '{key}': {content}"


def _validate_tool_input_start(content, idx):
    assert isinstance(content, dict), f"Event {idx}: tool_input_start should be dict, got {type(content)}"
    for key in ("tool_run_id", "action", "tool"):
        assert key in content, f"Event {idx}: tool_input_start missing '{key}': {content}"


def _validate_tool_input(content, idx):
    assert isinstance(content, dict), f"Event {idx}: tool_input should be dict, got {type(content)}"
    for key in ("tool_run_id", "action_input"):
        assert key in content, f"Event {idx}: tool_input missing '{key}': {content}"


_VALIDATORS = {
    "post_parse_reasoning": _validate_post_parse_reasoning,
    "tool_result": _validate_tool_result,
    "tool_input_start": _validate_tool_input_start,
    "tool_input": _validate_tool_input,
}


# ---------------------------------------------------------------------------
# Shared FSM helpers (reusable across all modes)
# ---------------------------------------------------------------------------


def _fsm_step_transition(event_name, state, transitions, idx, step, content):
    """Validate and return the next state for a single FSM step.

    Asserts that event_name is allowed from the current state, runs structural
    validators, and returns next_state.
    """
    assert event_name is not None, f"Event {idx}: unknown step '{step}'"

    allowed = transitions.get(state, {})
    assert event_name in allowed, (
        f"Event {idx}: unexpected '{event_name}' in state {state.name}. "
        f"Allowed: {list(allowed.keys())}. "
        f"Raw: step={step}, content_type={'dict' if isinstance(content, dict) else 'str'}"
    )

    validator = _VALIDATORS.get(event_name)
    if validator:
        validator(content, idx)

    return allowed[event_name]


def _track_reasoning(event_name, state, next_state, content, reasoning_blocks):
    """Track reasoning block lifecycle. Call on every FSM step."""
    if next_state == State.REASONING and state != State.REASONING:
        reasoning_blocks.append("")
    if event_name == "reasoning" and reasoning_blocks:
        if isinstance(content, str):
            reasoning_blocks[-1] += content
        elif isinstance(content, dict) and "thought" in content:
            reasoning_blocks[-1] += content["thought"]


def _track_tool_input(event_name, content, tool_blocks):
    """Track tool_input_start and tool_input chunk accumulation. Returns updated max delta."""
    if event_name == "tool_input_start" and isinstance(content, dict):
        tid = content.get("tool_run_id")
        if tid:
            tool_blocks[tid] = {
                "name": content.get("action"),
                "action_input_chunks": [],
            }
    elif event_name == "tool_input" and isinstance(content, dict):
        tid = content.get("tool_run_id")
        if tid and tid in tool_blocks:
            tool_blocks[tid]["action_input_chunks"].append(content.get("action_input", ""))


def _handle_tool_result(event_name, content, tool_blocks, reasoning_blocks, run_parallel_count):
    """Handle tool_result / tool_input_error events. Returns updated run_parallel_count."""
    if event_name not in ("tool_result", "tool_input_error") or not isinstance(content, dict):
        return run_parallel_count

    tid = content.get("tool_run_id")
    result_name = content.get("name", "")

    if result_name == "run-parallel":
        run_parallel_count -= 1
        assert run_parallel_count == 0, (
            f"run-parallel tool_result without matching post_parse_reasoning, "
            f"run_parallel_count={run_parallel_count}"
        )
        if tid and tid in tool_blocks:
            tool_blocks.pop(tid)
    else:
        if tid and tid in tool_blocks:
            tool_blocks.pop(tid)
        if reasoning_blocks:
            reasoning_blocks.pop(0)

    return run_parallel_count


def _handle_answer(event_name, reasoning_blocks):
    """Pop reasoning block on answer event."""
    if event_name == "answer" and reasoning_blocks:
        reasoning_blocks.pop(0)


def _track_parallel_individual_post_parse(content, idx, parallel_post_parse_tids):
    """Record an individual post_parse tid seen during a parallel batch.

    Asserts the tid is present and not a duplicate within the current batch.
    """
    tid = content.get("tool_run_id")
    assert tid, f"Event {idx}: individual post_parse during parallel batch missing tool_run_id"
    assert (
        tid not in parallel_post_parse_tids
    ), f"Event {idx}: duplicate individual post_parse for tool_run_id {tid} in parallel batch"
    parallel_post_parse_tids.add(tid)


def _verify_parallel_tool_result(event_name, content, idx, parallel_post_parse_tids, run_parallel_count):
    """Enforce 1:1 pairing between individual post_parse and per-tool tool_result.

    - Per-tool tool_result during a parallel batch: tid must be in the set; remove it.
    - run-parallel tool_result: the set must be empty (all per-tool results paired).
    """
    if event_name != "tool_result" or not isinstance(content, dict):
        return
    name = content.get("name")
    tid = content.get("tool_run_id")
    if name == "run-parallel":
        assert not parallel_post_parse_tids, (
            f"Event {idx}: run-parallel tool_result but {len(parallel_post_parse_tids)} "
            f"individual post_parse event(s) have no matching tool_result: "
            f"{parallel_post_parse_tids}"
        )
    elif run_parallel_count > 0:
        assert tid in parallel_post_parse_tids, (
            f"Event {idx}: tool_result for tool_run_id {tid} during parallel batch "
            f"without matching individual post_parse. Seen: {parallel_post_parse_tids}"
        )
        parallel_post_parse_tids.discard(tid)


def _match_action_input(accumulated: str, expected) -> bool:
    """Check if accumulated tool_input string matches expected action_input.

    Handles:
      1. Direct string match.
      2. Wrapped in {"input": ...} dict — compare against inner value.
      3. Accumulated is JSON-escaped (streamed as raw JSON) — decode and compare.
      4. Both sides JSON-decoded for structural comparison.
    """
    attempts = []

    # 1. Direct match
    if accumulated == expected:
        return True
    attempts.append(f"direct: {accumulated!r} == {expected!r} -> False")

    # 2. Wrapped dict match
    if isinstance(expected, dict) and "input" in expected:
        inner = expected["input"]
        if accumulated == inner:
            return True
        attempts.append(f"wrapped: {accumulated!r} == {inner!r} -> False")

    # 3. Decode accumulated (JSON-escaped streaming) and compare.
    #    Structured output streams action_input as a JSON string field, so the
    #    accumulated text is the raw string body with escape sequences (e.g.
    #    {\"key\":\"val\"}).  Wrap in quotes to form a valid JSON string literal
    #    before decoding.
    decoded = None
    try:
        decoded = json.loads(accumulated)
    except (json.JSONDecodeError, TypeError):
        try:
            decoded = json.loads(f'"{accumulated}"')
        except (json.JSONDecodeError, TypeError):
            pass
    if decoded is None:
        attempts.append("json.loads(accumulated) -> FAILED")

    # If decoded is itself a JSON string (structured output double-encoding),
    # unwrap one more level.
    if isinstance(decoded, str):
        try:
            decoded = json.loads(decoded)
        except (json.JSONDecodeError, TypeError):
            pass

    if decoded is not None:
        if decoded == expected:
            return True
        attempts.append(f"decoded==expected: {decoded!r} == {expected!r} -> False")

        if isinstance(expected, dict) and "input" in expected:
            inner = expected["input"]
            if decoded == inner:
                return True
            attempts.append(f"decoded==inner: {decoded!r} == {inner!r} -> False")

            # 4. Both sides JSON-decoded (inner may also be a JSON string)
            if isinstance(inner, str):
                try:
                    inner_decoded = json.loads(inner)
                    if decoded == inner_decoded:
                        return True
                    attempts.append(f"decoded==inner_decoded: {decoded!r} == {inner_decoded!r} -> False")
                except (json.JSONDecodeError, TypeError):
                    attempts.append("json.loads(inner) -> FAILED")

    logger.debug("[_match_action_input] all attempts failed:\n" + "\n".join(f"  {a}" for a in attempts))
    return False


def _validate_single_post_parse(content, tool_blocks, reasoning_blocks):
    """Validate an individual (non-run-parallel) post_parse_reasoning event.

    Checks:
      - Must not appear when multiple tools are in-flight.
      - Accumulated tool_input matches action_input (direct or {"input": ...} wrapped).
      - Accumulated reasoning matches thought.
    """
    tid = content.get("tool_run_id")
    tool_name = content.get("action", "<unknown>")

    assert len(tool_blocks) <= 1, (
        f"Individual post_parse_reasoning for '{tool_name}' appeared with "
        f"{len(tool_blocks)} tools in-flight. Individual post_parse events "
        f"should not be emitted during parallel execution: "
        f"{[b['name'] for b in tool_blocks.values()]}"
    )

    if tid and tid in tool_blocks:
        block = tool_blocks[tid]
        accumulated_input = "".join(block["action_input_chunks"])
        expected_input = content.get("action_input")
        if accumulated_input and expected_input is not None:
            assert _match_action_input(accumulated_input, expected_input), (
                f"tool_run_id {tid} ({tool_name}): accumulated tool_input "
                f"does not match post_parse action_input. "
                f"Accumulated: {accumulated_input!r}\n"
                f"Expected:    {expected_input!r}"
            )

    expected_thought = content.get("thought")
    if reasoning_blocks and expected_thought:
        accumulated_thought = reasoning_blocks[0]
        # Streaming emits raw JSON string content (e.g. \n as two chars),
        # while post_parse decodes via json.loads (real newline).
        # Encode expected to raw JSON form for a fair comparison.
        expected_thought_raw = json.dumps(expected_thought)[1:-1]
        assert accumulated_thought == expected_thought or accumulated_thought == expected_thought_raw, (
            f"tool_run_id {tid} ({tool_name}): accumulated reasoning "
            f"({len(accumulated_thought)} chars) does not match post_parse thought. "
            f"Accumulated: {accumulated_thought!r}\n"
            f"Expected:    {expected_thought!r}"
        )
        logger.debug(f"[reasoning_block] {tool_name} ({tid}): thought matched " f"({len(accumulated_thought)} chars)")


def _assert_fsm_end(tool_blocks, reasoning_blocks, run_parallel_count, parallel_post_parse_tids=None):
    """Assert clean end state: no in-flight tool blocks, reasoning, or parallel batches."""
    assert len(tool_blocks) == 0, (
        f"Unresolved tool blocks at end of stream: " f"{[{tid: b['name']} for tid, b in tool_blocks.items()]}"
    )
    assert len(reasoning_blocks) == 0, (
        f"Unresolved reasoning blocks at end of stream: {len(reasoning_blocks)} remaining. "
        f"Previews: {[b[:80] + '...' if len(b) > 80 else b for b in reasoning_blocks]}"
    )
    assert run_parallel_count == 0, (
        f"Unresolved run-parallel events: run_parallel_count={run_parallel_count} "
        f"(post_parse_reasoning without matching tool_result or vice versa)"
    )
    if parallel_post_parse_tids:
        raise AssertionError(
            f"Unresolved individual post_parse events at end of stream: "
            f"{parallel_post_parse_tids} (no matching per-tool tool_result)"
        )


def _log_event(idx, step, event_name, state, content):
    """Debug-log a single FSM event."""
    content_preview = content
    if isinstance(content, str):
        content_preview = repr(content[:80]) if len(content) > 80 else repr(content)
    logger.debug(
        f"[FSM] Event {idx}: step={step}, event={event_name}, " f"state={state.name}, content={content_preview}"
    )


# ---------------------------------------------------------------------------
# Per-mode FSM runners
# ---------------------------------------------------------------------------


def _run_fsm_fc(ordered_events, streaming_mode):
    """FSM for FUNCTION_CALLING mode.

    FC streams per-tool (reasoning → tool_input_start → tool_input) blocks,
    then a single run-parallel post_parse_reasoning. Individual per-tool
    post_parse_reasoning events emitted during a parallel batch are tracked
    (tid-only) and paired 1:1 with per-tool tool_result events. Validates
    per-tool action_input inside run-parallel.
    """
    transitions = _TRANSITIONS_FINAL if streaming_mode == StreamingMode.FINAL else _TRANSITIONS_FC
    state = State.INIT
    visited = {state}
    reasoning_blocks: list[str] = []
    tool_blocks: dict[str, dict] = {}
    run_parallel_count = 0
    parallel_post_parse_tids: set[str] = set()

    for idx, (step, content) in enumerate(ordered_events):
        event_name = _classify_event(step, content)
        _log_event(idx, step, event_name, state, content)

        if (
            event_name == "post_parse_reasoning"
            and isinstance(content, dict)
            and content.get("action") != "run-parallel"
            and run_parallel_count > 0
        ):
            _track_parallel_individual_post_parse(content, idx, parallel_post_parse_tids)
            continue

        _verify_parallel_tool_result(event_name, content, idx, parallel_post_parse_tids, run_parallel_count)

        next_state = _fsm_step_transition(event_name, state, transitions, idx, step, content)

        _track_reasoning(event_name, state, next_state, content, reasoning_blocks)
        _track_tool_input(event_name, content, tool_blocks)

        if event_name == "post_parse_reasoning" and isinstance(content, dict):
            tool_name = content.get("action", "<unknown>")
            is_run_parallel = tool_name == "run-parallel"

            if is_run_parallel:
                run_parallel_count += 1
                # FC parallel: multiple per-tool tool_blocks must be in-flight
                assert len(tool_blocks) > 1, (
                    f"run-parallel post_parse_reasoning but only "
                    f"{len(tool_blocks)} tool(s) in-flight: "
                    f"{[b['name'] for b in tool_blocks.values()]}"
                )
                assert run_parallel_count == 1, (
                    f"Expected exactly 1 run-parallel event per parallel batch, " f"got {run_parallel_count}"
                )

                # Validate per-tool entries inside action_input
                action_input = content.get("action_input")
                if isinstance(action_input, list):
                    tool_entries = [e for e in action_input if isinstance(e, dict)]

                    # Each tool must have its own reasoning block, validated by index
                    assert len(reasoning_blocks) >= len(tool_entries), (
                        f"run-parallel: {len(tool_entries)} tool entries but only "
                        f"{len(reasoning_blocks)} reasoning blocks accumulated"
                    )

                    for i, entry in enumerate(tool_entries):
                        entry_tid = entry.get("tool_run_id")
                        entry_input = entry.get("action_input")
                        entry_thought = entry.get("thought", "")
                        tool_label = entry.get("action", "?")

                        # Validate action_input match
                        if entry_tid and entry_tid in tool_blocks:
                            block = tool_blocks[entry_tid]
                            accumulated = "".join(block["action_input_chunks"])
                            if accumulated and entry_input is not None:
                                assert _match_action_input(accumulated, entry_input), (
                                    f"run-parallel tool[{i}] {entry_tid} "
                                    f"({tool_label}): accumulated tool_input "
                                    f"does not match action_input. "
                                    f"Accumulated: {accumulated!r}\n"
                                    f"Expected:    {entry_input!r}"
                                )

                        # Validate thought against reasoning_blocks[i].
                        # Streaming may emit raw JSON escapes (e.g. \n as two chars),
                        # so also compare against the JSON-encoded form.
                        accumulated_thought = reasoning_blocks[i]
                        expected_thought_raw = json.dumps(entry_thought)[1:-1]
                        assert accumulated_thought == entry_thought or accumulated_thought == expected_thought_raw, (
                            f"run-parallel tool[{i}] {entry_tid} "
                            f"({tool_label}): reasoning_blocks[{i}] "
                            f"does not match per-tool thought. "
                            f"Accumulated: {accumulated_thought!r}\n"
                            f"Expected:    {entry_thought!r}"
                        )
            else:
                _validate_single_post_parse(content, tool_blocks, reasoning_blocks)

        run_parallel_count = _handle_tool_result(event_name, content, tool_blocks, reasoning_blocks, run_parallel_count)
        _handle_answer(event_name, reasoning_blocks)

        state = next_state
        visited.add(state)

    _assert_fsm_end(tool_blocks, reasoning_blocks, run_parallel_count)
    return state, visited, reasoning_blocks


def _run_fsm_blob(ordered_events, streaming_mode):
    """FSM for STRUCTURED_OUTPUT and XML modes.

    SO/XML stream a single run-parallel blob:
      tool_input_start(action=run-parallel) → tool_input chunks → post_parse → tool results

    No per-tool tool_input_start events inside a parallel batch. Individual
    per-tool post_parse_reasoning events emitted during a parallel batch are
    tracked (tid-only) and paired 1:1 with per-tool tool_result events.
    """
    transitions = _TRANSITIONS_FINAL if streaming_mode == StreamingMode.FINAL else _TRANSITIONS_WITH_TOOL_INPUT
    state = State.INIT
    visited = {state}
    reasoning_blocks: list[str] = []
    tool_blocks: dict[str, dict] = {}
    run_parallel_count = 0
    parallel_post_parse_tids: set[str] = set()

    for idx, (step, content) in enumerate(ordered_events):
        event_name = _classify_event(step, content)
        _log_event(idx, step, event_name, state, content)

        if (
            event_name == "post_parse_reasoning"
            and isinstance(content, dict)
            and content.get("action") != "run-parallel"
            and run_parallel_count > 0
        ):
            _track_parallel_individual_post_parse(content, idx, parallel_post_parse_tids)
            continue

        _verify_parallel_tool_result(event_name, content, idx, parallel_post_parse_tids, run_parallel_count)

        next_state = _fsm_step_transition(event_name, state, transitions, idx, step, content)

        _track_reasoning(event_name, state, next_state, content, reasoning_blocks)
        _track_tool_input(event_name, content, tool_blocks)

        if event_name == "post_parse_reasoning" and isinstance(content, dict):
            tool_name = content.get("action", "<unknown>")
            is_run_parallel = tool_name == "run-parallel"

            if is_run_parallel:
                run_parallel_count += 1
                # SO/XML parallel: a single run-parallel blob in tool_blocks
                has_parallel_context = len(tool_blocks) > 1 or any(
                    b["name"] == "run-parallel" for b in tool_blocks.values()
                )
                assert has_parallel_context, (
                    f"run-parallel post_parse_reasoning but no parallel context: "
                    f"{len(tool_blocks)} tool(s) in-flight: "
                    f"{[b['name'] for b in tool_blocks.values()]}"
                )
                assert run_parallel_count == 1, (
                    f"Expected exactly 1 run-parallel event per parallel batch, " f"got {run_parallel_count}"
                )

                # Validate thought matches accumulated reasoning
                expected_thought = content.get("thought", "")
                if reasoning_blocks and expected_thought:
                    accumulated_thought = reasoning_blocks[0]
                    if accumulated_thought:
                        assert accumulated_thought == expected_thought, (
                            f"run-parallel: accumulated reasoning "
                            f"({len(accumulated_thought)} chars) does not match thought. "
                            f"Accumulated: {accumulated_thought!r}\n"
                            f"Expected:    {expected_thought!r}"
                        )
            else:
                _validate_single_post_parse(content, tool_blocks, reasoning_blocks)

        run_parallel_count = _handle_tool_result(event_name, content, tool_blocks, reasoning_blocks, run_parallel_count)
        _handle_answer(event_name, reasoning_blocks)

        state = next_state
        visited.add(state)

    _assert_fsm_end(tool_blocks, reasoning_blocks, run_parallel_count)
    return state, visited, reasoning_blocks


def _run_fsm_default(ordered_events, streaming_mode):
    """FSM for DEFAULT mode.

    DEFAULT mode has no tool_input streaming phase. Only validates
    transitions and reasoning/tool_result lifecycle.
    """
    transitions = _TRANSITIONS_FINAL if streaming_mode == StreamingMode.FINAL else _TRANSITIONS_DEFAULT
    state = State.INIT
    visited = {state}
    reasoning_blocks: list[str] = []
    run_parallel_count = 0
    tool_blocks: dict[str, dict] = {}

    for idx, (step, content) in enumerate(ordered_events):
        event_name = _classify_event(step, content)
        _log_event(idx, step, event_name, state, content)
        next_state = _fsm_step_transition(event_name, state, transitions, idx, step, content)

        _track_reasoning(event_name, state, next_state, content, reasoning_blocks)

        if event_name == "post_parse_reasoning" and isinstance(content, dict):
            tool_name = content.get("action", "<unknown>")
            if tool_name == "run-parallel":
                run_parallel_count += 1

        run_parallel_count = _handle_tool_result(event_name, content, tool_blocks, reasoning_blocks, run_parallel_count)
        _handle_answer(event_name, reasoning_blocks)

        state = next_state
        visited.add(state)

    _assert_fsm_end(tool_blocks, reasoning_blocks, run_parallel_count)
    return state, visited, reasoning_blocks


# ---------------------------------------------------------------------------
# High-level assertion
# ---------------------------------------------------------------------------


def assert_streaming_events(
    ordered_events: list,
    inference_mode: InferenceMode,
    streaming_mode: StreamingMode = StreamingMode.ALL,
):
    """Validate ordered streaming events against the FSM event policy.

    Args:
        ordered_events: List of (step, content) tuples from collect_streaming_events().
        inference_mode: The InferenceMode the agent was configured with.
        streaming_mode: The StreamingMode used during the run.
    """
    assert len(ordered_events) > 0, "No streaming events collected"

    steps = [s for s, _ in ordered_events]
    step_counts = {}
    for s in steps:
        step_counts[s] = step_counts.get(s, 0) + 1

    logger.info(
        f"Asserting streaming FSM {inference_mode.value}/{streaming_mode.value}: "
        f"{len(ordered_events)} events, counts = {step_counts}"
    )

    if inference_mode == InferenceMode.FUNCTION_CALLING:
        final_state, visited, reasoning_blocks = _run_fsm_fc(ordered_events, streaming_mode)
    elif inference_mode in (InferenceMode.STRUCTURED_OUTPUT, InferenceMode.XML):
        final_state, visited, reasoning_blocks = _run_fsm_blob(ordered_events, streaming_mode)
    else:
        final_state, visited, reasoning_blocks = _run_fsm_default(ordered_events, streaming_mode)

    logger.info(f"Reasoning blocks: {reasoning_blocks}")

    assert final_state == State.ANSWER, (
        f"FSM ended in {final_state.name}, expected ANSWER. " f"Last event: {ordered_events[-1]}"
    )

    if streaming_mode == StreamingMode.ALL:
        assert State.REASONING in visited, (
            f"{inference_mode.value}/ALL: never entered REASONING state. " f"Visited: {[s.name for s in visited]}"
        )
