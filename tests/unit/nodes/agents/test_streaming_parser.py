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
    agent.inference_mode.name = InferenceMode.STRUCTURED_OUTPUT.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = None
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
    cb._process_json_mode(final_answer_only=False, is_function_calling=False)

    assert cb._current_state == expected_state
    assert cb._answer_started is answer_started
    assert cb._tool_input_started is tool_input_started
    assert cb._current_action_name == action_name


def _make_fc_callback(tool_input_started=False, answer_started=False, action_name=None):
    agent = MagicMock()
    agent.streaming.enabled = True
    agent.streaming.mode = StreamingMode.ALL
    agent.inference_mode.name = InferenceMode.FUNCTION_CALLING.value
    agent.name = "test-agent"
    agent._streaming_tool_run_id = "test-run-id"
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
    cb._process_json_mode(final_answer_only=False, is_function_calling=True)

    assert cb._current_state == expected_state
    assert cb._fc_object_tool_input is expected_fc_object
