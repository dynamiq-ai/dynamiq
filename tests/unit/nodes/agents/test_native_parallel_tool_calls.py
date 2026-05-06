"""Tests for native parallel tool calling in FUNCTION_CALLING inference mode."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME


def _make_agent(**kwargs):
    agent = MagicMock()
    agent.verbose = False
    agent.parallel_tool_calls_enabled = kwargs.get("parallel_tool_calls_enabled", True)
    agent.log_reasoning = MagicMock()
    agent.log_final_output = MagicMock()
    agent.sanitize_tool_name = lambda name: name
    agent._parse_output_files_csv = lambda v: []
    return agent


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


class TestFunctionCallingEdgeCases:

    def test_no_tool_calls_raises(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(output={})

        with pytest.raises(ActionParsingException):
            Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

    def test_arguments_as_json_string(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"thought": "t", "action_input": {"q": "a"}}),
                        }
                    }
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "search"
        assert thought == "t"
        assert action_input == {"q": "a"}

    def test_missing_thought_defaults_to_empty(self):
        """LLMs sometimes omit `thought` despite the schema. Tolerate it
        (default to empty string) instead of forcing a recoverable retry."""
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"action_input": {"q": "a"}}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "search"
        assert thought == ""
        assert action_input == {"q": "a"}

    def test_missing_action_input_raises(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"thought": "t"}}},
                ]
            }
        )

        with pytest.raises(ActionParsingException):
            Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

    def test_final_answer(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {
                        "function": {
                            "name": "provide_final_answer",
                            "arguments": {"thought": "done", "answer": "42", "output_files": ""},
                        }
                    }
                ]
            }
        )

        thought, action, result = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "final_answer"
        assert result == "42"
        assert thought == "done"

    def test_final_answer_missing_answer_raises(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {
                        "function": {
                            "name": "provide_final_answer",
                            "arguments": {"thought": "done"},
                        }
                    }
                ]
            }
        )

        with pytest.raises(ActionParsingException):
            Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

    def test_invalid_tool_calls_structure_raises(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(output={"tool_calls": [{"bad": "structure"}]})

        with pytest.raises(ActionParsingException):
            Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)


class TestFunctionCallingProtocolEmission:
    """End-to-end tests for OpenAI function-calling protocol emission in agent.py.

    Together these verify both halves of the protocol round-trip:
    - assistant turn carries native tool_calls + FA stub (provide_final_answer
      gets a dummy tool result so OpenAI doesn't 400 on missing tool_call_id),
    - observations are emitted as role:'tool' messages with matching ids/names.
    """

    def test_append_assistant_message_emits_native_tool_calls_with_fa_stub(self):
        """LLM returns native tool_calls including provide_final_answer.
        We must emit a real assistant message with native tool_calls AND a stub
        role:'tool' acknowledgment for the FA call so the protocol stays valid.
        Real-tool ids must end up in the pending stash; FA id must NOT."""
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.nodes.types import InferenceMode
        from dynamiq.prompts.prompts import MessageRole

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = ["stale_id_from_previous_loop"]

        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"id": "call_a", "function": {"name": "CatFacts", "arguments": {"q": "sleep"}}},
                    {"id": "call_b", "function": {"name": "DogFacts", "arguments": {"q": "smell"}}},
                    {
                        "id": "call_fa",
                        "function": {"name": "provide_final_answer", "arguments": {"answer": "done"}},
                    },
                ]
            }
        )

        Agent._append_assistant_message(agent, llm_result, llm_generated_output="")

        # 1. Assistant message + 1 stub tool message for FA = 2 messages
        assert len(agent._prompt.messages) == 2

        assistant = agent._prompt.messages[0]
        assert assistant.role == MessageRole.ASSISTANT
        assert assistant.content == "Calling: CatFacts, DogFacts, provide_final_answer"
        # All three calls (including FA) appear in the native tool_calls payload
        assert [tc["function"]["name"] for tc in assistant.tool_calls] == [
            "CatFacts", "DogFacts", "provide_final_answer",
        ]
        # arguments are JSON-encoded strings (OpenAI native shape)
        assert json.loads(assistant.tool_calls[0]["function"]["arguments"]) == {"q": "sleep"}

        fa_stub = agent._prompt.messages[1]
        assert fa_stub.role == MessageRole.TOOL
        assert fa_stub.tool_call_id == "call_fa"
        assert fa_stub.name == "provide_final_answer"
        assert fa_stub.content == "Acknowledged."

        # Pending stash holds ONLY real-tool ids (FA is acknowledged inline, not pending).
        # Stale id from previous loop is gone.
        assert agent._pending_fc_tool_call_ids == ["call_a", "call_b"]

    def test_append_assistant_message_drops_extra_calls_when_parallel_disabled(self):
        """Regression: when parallel is disabled and the LLM still returns multiple
        non-final-answer tool calls, only the first one may be recorded in the assistant
        message."""
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.nodes.types import InferenceMode

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent.parallel_tool_calls_enabled = False
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = []

        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"id": "call_a", "function": {"name": "CatFacts", "arguments": {}}},
                    {"id": "call_b", "function": {"name": "DogFacts", "arguments": {}}},
                    {"id": "call_fa", "function": {"name": "provide_final_answer", "arguments": {"answer": "done"}}},
                ]
            }
        )

        Agent._append_assistant_message(agent, llm_result, llm_generated_output="")

        assistant = agent._prompt.messages[0]
        # Only call_a (first non-FA) and call_fa survive in the assistant payload —
        # call_b is dropped at source so no orphan tool_call_id is left behind.
        assert [tc["id"] for tc in assistant.tool_calls] == ["call_a", "call_fa"]
        # Pending ids match: only call_a will get a tool response from execution;
        # call_fa is acknowledged inline.
        assert agent._pending_fc_tool_call_ids == ["call_a"]

    def test_emit_tool_observations_parallel_pairs_ids_results_and_names(self):
        """In FC mode, parallel observations must produce one role:'tool' message per
        pending id, paired with the matching ordered_result (id ↔ result by index),
        carrying tool_call_id, content, and name. Stash must be cleared after."""
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.nodes.types import InferenceMode
        from dynamiq.prompts.prompts import MessageRole

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = ["call_a", "call_b"]

        ordered_results = [
            {"tool_name": "CatFacts", "result": "Cats sleep 12-16h", "success": True, "files": []},
            {"tool_name": "DogFacts", "result": "Dogs have 40x smell", "success": True, "files": []},
        ]

        Agent._emit_tool_observations(
            agent, tool_result="combined_string_unused_here", ordered_results=ordered_results
        )

        assert len(agent._prompt.messages) == 2
        for m in agent._prompt.messages:
            assert m.role == MessageRole.TOOL

        first, second = agent._prompt.messages
        assert (first.tool_call_id, first.name, first.content) == ("call_a", "CatFacts", "Cats sleep 12-16h")
        assert (second.tool_call_id, second.name, second.content) == (
            "call_b", "DogFacts", "Dogs have 40x smell",
        )
        # Stash must be empty so the next loop doesn't see stale ids.
        assert agent._pending_fc_tool_call_ids == []


class TestRollbackOrphanFCPayload:
    def test_rollback_removes_orphan_assistant_and_fa_stub(self):
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.nodes.types import InferenceMode
        from dynamiq.prompts import Message, MessageRole

        earlier = Message(role=MessageRole.USER, content="hi", static=True)
        orphan = Message(
            role=MessageRole.ASSISTANT,
            content="Calling: search, provide_final_answer",
            tool_calls=[
                {"id": "call_a", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {
                    "id": "call_fa",
                    "type": "function",
                    "function": {"name": "provide_final_answer", "arguments": '{"answer": "x"}'},
                },
            ],
            static=True,
        )
        fa_stub = Message(
            role=MessageRole.TOOL,
            content="Acknowledged.",
            tool_call_id="call_fa",
            name="provide_final_answer",
            static=True,
        )

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = [earlier, orphan, fa_stub]
        agent._pending_fc_tool_call_ids = ["call_a"]

        Agent._rollback_orphan_fc_payload(agent)

        assert agent._prompt.messages == [earlier]
        assert agent._pending_fc_tool_call_ids == []
