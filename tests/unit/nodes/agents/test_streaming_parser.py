from unittest.mock import MagicMock

import pytest

from dynamiq.callbacks.streaming import (
    AgentStreamingParserCallback,
    InferenceMode,
    StreamingState,
)
from dynamiq.types.streaming import StreamingMode


def _make_callback():
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.min_chunk_chars = 0
    agent.inference_mode.name = InferenceMode.STRUCTURED_OUTPUT.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = None
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    return AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)


@pytest.mark.parametrize(
    "buf, expected_state, answer_started, tool_input_started, action_name",
    [
        pytest.param(
            '{"action": "finish", "action_input": "Hello',
            StreamingState.ANSWER,
            True,
            False,
            None,
            id="finish",
        ),
        pytest.param(
            '{"action": "search", "action_input": "query',
            StreamingState.TOOL_INPUT,
            False,
            True,
            "search",
            id="non_finish",
        ),
        pytest.param(
            '{"thought": "I should search',
            StreamingState.REASONING,
            False,
            False,
            None,
            id="thought_only",
        ),
    ],
)
def test_process_json_mode_structured_output(
    buf, expected_state, answer_started, tool_input_started, action_name
):
    cb = _make_callback()
    cb._buffer = buf
    cb._process_structured_output_mode(final_answer_only=False)

    assert cb._current_state == expected_state
    assert cb._answer_started is answer_started
    assert cb._tool_input_started is tool_input_started
    assert cb._current_action_name == action_name


def _make_fc_callback(tool_input_started=False, answer_started=False, action_name=None):
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.min_chunk_chars = 0
    agent.inference_mode.name = InferenceMode.FUNCTION_CALLING.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = "test-run-id"
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    cb = AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)
    cb._tool_input_started = tool_input_started
    cb._answer_started = answer_started
    cb._current_action_name = action_name
    return cb


@pytest.mark.parametrize(
    "buf, tool_input_started, answer_started, action_name, expected_state, expected_fc_object",
    [
        pytest.param(
            '{"action_input": {"query": "hello"',
            True,
            False,
            "exa_search",
            StreamingState.TOOL_INPUT,
            True,
            id="object_action_input",
        ),
        pytest.param(
            '{"action_input": "sub-query text',
            True,
            False,
            "sub_agent",
            StreamingState.TOOL_INPUT,
            False,
            id="string_action_input",
        ),
        pytest.param(
            '{"answer": "Here is the result',
            False,
            True,
            None,
            StreamingState.ANSWER,
            False,
            id="final_answer",
        ),
        pytest.param(
            '{"thought": "I need to search',
            False,
            False,
            None,
            StreamingState.REASONING,
            False,
            id="thought_only",
        ),
    ],
)
def test_process_json_mode_function_calling(
    buf, tool_input_started, answer_started, action_name, expected_state, expected_fc_object
):
    cb = _make_fc_callback(
        tool_input_started=tool_input_started,
        answer_started=answer_started,
        action_name=action_name,
    )
    cb._buffer = buf
    cb._process_json_mode(final_answer_only=False)

    assert cb._current_state == expected_state
    assert cb._fc_object_tool_input is expected_fc_object


def _make_fc_chunk(index=0, function_name=None, arguments=""):
    """Build a minimal LLM streaming chunk for function calling mode."""
    tc = {"index": index, "type": "function", "function": {}}
    if function_name:
        tc["function"]["name"] = function_name
    if arguments:
        tc["function"]["arguments"] = arguments
    return {"choices": [{"delta": {"tool_calls": [tc]}}]}


def test_parallel_tool_calls_get_unique_ids():
    """Each parallel tool call must receive a distinct tool_run_id during streaming."""
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.streaming.min_chunk_chars = 0
    agent.inference_mode.name = InferenceMode.FUNCTION_CALLING.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = None
    agent._streaming_tool_run_ids = []
    agent.tool_by_names = {}
    agent.sanitize_tool_name = lambda name: name
    agent.llm = MagicMock()
    agent.llm.id = "llm-1"

    cb = AgentStreamingParserCallback(agent=agent, config=None, loop_num=1)
    serialized = {"group": "llms", "id": "llm-1"}

    cb.on_node_execute_stream(serialized, _make_fc_chunk(index=0, function_name="exa_search"))
    cb.on_node_execute_stream(serialized, _make_fc_chunk(index=0, arguments='{"query":"hello"}'))
    id_tool_0 = agent._streaming_tool_run_id

    cb.on_node_execute_stream(serialized, _make_fc_chunk(index=1, function_name="tavily_search"))
    cb.on_node_execute_stream(serialized, _make_fc_chunk(index=1, arguments='{"query":"world"}'))
    id_tool_1 = agent._streaming_tool_run_id

    assert id_tool_0 is not None
    assert id_tool_1 is not None
    assert id_tool_0 != id_tool_1

    assert agent._streaming_tool_run_ids == [id_tool_0, id_tool_1]


@pytest.mark.parametrize(
    "stream_tool_input, action_name, should_stream",
    [
        pytest.param(None, "search", True, id="none_allows_all"),
        pytest.param(["search"], "search", True, id="in_allowlist"),
        pytest.param(["search"], "calculator", False, id="not_in_allowlist"),
        pytest.param([], "search", False, id="empty_list_blocks_all"),
    ],
)
def test_stream_tool_input_allowlist(stream_tool_input, action_name, should_stream):
    """stream_tool_input allowlist controls which tools get TOOL_INPUT events."""
    cb = _make_callback()
    cb.agent.streaming.stream_tool_input = stream_tool_input
    cb._current_action_name = action_name
    cb.agent._streaming_tool_run_id = "test-run-id"

    cb._emit("some input content", step=StreamingState.TOOL_INPUT)

    if should_stream:
        cb.agent.stream_content.assert_called_once()
    else:
        cb.agent.stream_content.assert_not_called()
