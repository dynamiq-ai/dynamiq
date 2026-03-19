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
