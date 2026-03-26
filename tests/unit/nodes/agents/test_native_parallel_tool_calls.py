"""Tests for native parallel tool calling in FUNCTION_CALLING inference mode."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME


class TestNativeParallelToolCalling:
    """Core tests for native parallel tool calling."""

    def test_tool_calls_returned_as_list(self):
        from dynamiq.nodes.llms.base import BaseLLM

        tc1 = {"function": {"name": "search", "arguments": json.dumps({"thought": "t1", "action_input": {"q": "a"}})}}
        tc2 = {"function": {"name": "search", "arguments": json.dumps({"thought": "t2", "action_input": {"q": "b"}})}}

        tc_objects = []
        for tc in [tc1, tc2]:
            obj = MagicMock()
            obj.model_dump.return_value = tc
            tc_objects.append(obj)

        message = MagicMock()
        message.content = "content"
        message.tool_calls = tc_objects
        response = MagicMock()
        response.choices = [MagicMock(message=message)]

        with patch.object(BaseLLM, "get_usage_data") as mock_usage, patch.object(BaseLLM, "run_on_node_execute_run"):
            mock_usage.return_value = MagicMock(model_dump=lambda: {})
            result = BaseLLM._handle_completion_response(MagicMock(), response, config=MagicMock())

        assert isinstance(result["tool_calls"], list)
        assert len(result["tool_calls"]) == 2

    def test_multiple_tool_calls_routed_as_parallel_batch(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = MagicMock()
        agent.verbose = False
        agent.parallel_tool_calls_enabled = True
        agent.log_reasoning = MagicMock()
        agent.sanitize_tool_name = lambda name: name

        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"thought": "first", "action_input": {"q": "a"}}}},
                    {"function": {"name": "calc", "arguments": {"thought": "second", "action_input": {"expr": "1+1"}}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == PARALLEL_TOOL_NAME
        assert len(action_input["tools"]) == 2
        assert action_input["tools"][0]["name"] == "search"
        assert action_input["tools"][1]["name"] == "calc"

    def test_single_tool_call_unchanged(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = MagicMock()
        agent.verbose = False
        agent.parallel_tool_calls_enabled = True
        agent.log_reasoning = MagicMock()

        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"thought": "t", "action_input": {"q": "a"}}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "search"
        assert action_input == {"q": "a"}
