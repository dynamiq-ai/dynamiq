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

        tc1 = {"function": {"name": "search", "arguments": json.dumps({"thought": "t1", "q": "a"})}}
        tc2 = {"function": {"name": "search", "arguments": json.dumps({"thought": "t2", "q": "b"})}}

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
                    {"function": {"name": "search", "arguments": {"thought": "first", "q": "a"}}},
                    {"function": {"name": "calc", "arguments": {"thought": "second", "expr": "1+1"}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == PARALLEL_TOOL_NAME
        assert len(action_input["tools"]) == 2
        assert action_input["tools"][0]["name"] == "search"
        assert action_input["tools"][0]["input"] == {"q": "a"}
        assert action_input["tools"][1]["name"] == "calc"
        assert action_input["tools"][1]["input"] == {"expr": "1+1"}

    def test_single_tool_call_unchanged(self):
        from dynamiq.nodes.agents.agent import Agent

        agent = MagicMock()
        agent.verbose = False
        agent.parallel_tool_calls_enabled = True
        agent.log_reasoning = MagicMock()

        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"thought": "t", "q": "a"}}},
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
        """OpenAI's wire protocol: function.arguments arrives as a JSON-encoded string."""
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"thought": "t", "q": "a"}),
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
                    {"function": {"name": "search", "arguments": {"q": "a"}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "search"
        assert thought == ""
        assert action_input == {"q": "a"}

    def test_only_thought_yields_empty_action_input(self):
        """A tool call with only `thought` and no real params produces an empty dict.

        Some tools genuinely take no parameters (their schema has only `thought`),
        so this is not an error.
        """
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        llm_result = SimpleNamespace(
            output={
                "tool_calls": [
                    {"function": {"name": "search", "arguments": {"thought": "t"}}},
                ]
            }
        )

        thought, action, action_input = Agent._handle_function_calling_mode(agent, llm_result, loop_num=1)

        assert action == "search"
        assert thought == "t"
        assert action_input == {}

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
        # Batched with tool calls, so the answer predates their results.
        assert fa_stub.content.startswith("Not accepted:")

        # Pending stash holds ONLY real-tool ids (FA is acknowledged inline, not pending).
        # Stale id from previous loop is gone.
        assert agent._pending_fc_tool_call_ids == ["call_a", "call_b"]

    def test_append_assistant_message_records_every_call_when_parallel_disabled(self):
        """Regression: extra calls used to be erased here, so the loss was silent.

        ``parallel_tool_calls_enabled=False`` means "do not run these concurrently", not
        "discard them", so history must record every call the model asked for."""
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
        assert [tc["id"] for tc in assistant.tool_calls] == ["call_a", "call_b", "call_fa"]
        # Both real calls await a result; call_fa is acknowledged inline.
        assert agent._pending_fc_tool_call_ids == ["call_a", "call_b"]

    def test_emit_tool_observations_parallel_pairs_ids_results_and_names(self):
        """In FC mode, parallel observations must produce one role:'tool' message per
        pending id, paired with the result carrying that id, with tool_call_id, content,
        and name. Stash must be cleared after."""
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.nodes.types import InferenceMode
        from dynamiq.prompts.prompts import MessageRole

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = ["call_a", "call_b"]
        agent._current_tool_call_ids = lambda: agent._pending_fc_tool_call_ids

        ordered_results = [
            {"tool_call_id": "call_a", "tool_name": "CatFacts", "result": "Cats sleep 12-16h", "success": True},
            {"tool_call_id": "call_b", "tool_name": "DogFacts", "result": "Dogs have 40x smell", "success": True},
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


class TestEveryPendingIdIsAnswered:
    """A request carrying an unanswered tool_call_id is rejected by the provider, so any
    path that ends a batch early must still reply to the calls it did not run."""

    @staticmethod
    def _agent(pending_ids):
        from dynamiq.nodes.types import InferenceMode

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = list(pending_ids)
        agent._current_tool_call_ids = lambda: agent._pending_fc_tool_call_ids
        return agent

    def test_results_are_matched_by_id_not_position(self):
        """The pool returns in completion order, so a result's position in the list says
        nothing about which call it belongs to. Only the id it carries does."""
        from dynamiq.nodes.agents.agent import Agent

        agent = self._agent(["call_a", "call_b", "call_c"])
        # Deliberately shuffled: pairing by position here would answer every call wrongly.
        ordered_results = [
            {"tool_call_id": "call_c", "tool_name": "Fetch", "result": "c-result"},
            {"tool_call_id": "call_a", "tool_name": "Search", "result": "a-result"},
            {"tool_call_id": "call_b", "tool_name": "Dogs", "result": "b-result"},
        ]

        Agent._emit_tool_observations(agent, tool_result="unused", ordered_results=ordered_results)

        assert [(m.tool_call_id, m.name, m.content) for m in agent._prompt.messages] == [
            ("call_a", "Search", "a-result"),
            ("call_b", "Dogs", "b-result"),
            ("call_c", "Fetch", "c-result"),
        ]
        assert agent._pending_fc_tool_call_ids == []

    def test_ids_without_a_result_get_the_step_message(self):
        """Nothing ran — the batch was refused before execution — so every id is answered
        with the refusal, which applies to all of them equally."""
        from dynamiq.nodes.agents.agent import Agent
        from dynamiq.prompts.prompts import MessageRole

        agent = self._agent(["call_a", "call_b", "call_c"])

        Agent._emit_tool_observations(agent, tool_result="Sub-agent invocation limit exceeded.")

        assert [m.tool_call_id for m in agent._prompt.messages] == ["call_a", "call_b", "call_c"]
        assert all(m.role == MessageRole.TOOL for m in agent._prompt.messages)
        assert all(m.content == "Sub-agent invocation limit exceeded." for m in agent._prompt.messages)
        assert agent._pending_fc_tool_call_ids == []

    def test_partial_results_pair_and_the_rest_are_explained(self):
        """A batch that only partly reported still answers every id: the reported call
        gets its own result, the others the step's message."""
        from dynamiq.nodes.agents.agent import Agent

        agent = self._agent(["call_a", "call_b"])
        ordered_results = [{"tool_call_id": "call_b", "tool_name": "Dogs", "result": "dogs"}]

        Agent._emit_tool_observations(agent, tool_result="batch ended early", ordered_results=ordered_results)

        assert [(m.tool_call_id, m.content) for m in agent._prompt.messages] == [
            ("call_a", "batch ended early"),
            ("call_b", "dogs"),
        ]

    def test_a_result_without_an_id_is_never_misattributed(self):
        """An unstamped result costs that call its content. It must never be handed to a
        different call — losing information beats fabricating it."""
        from dynamiq.nodes.agents.agent import Agent

        agent = self._agent(["call_a", "call_b"])
        ordered_results = [
            {"tool_name": "Search", "result": "a-result"},
            {"tool_name": "Dogs", "result": "b-result"},
        ]

        Agent._emit_tool_observations(agent, tool_result="no result reported", ordered_results=ordered_results)

        assert [m.content for m in agent._prompt.messages] == ["no result reported"] * 2


class TestFinalAnswerMustStandAlone:
    """An answer batched with tool calls predates their results, so it is declined.

    It is found by name, not position: llama-3.3-70b puts it last, gpt-4o first."""

    @staticmethod
    def _call(name, args, call_id):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def _parse(self, calls):
        from dynamiq.nodes.agents.agent import Agent

        agent = _make_agent()
        agent.tool_by_names = {}
        return Agent._handle_function_calling_mode(agent, SimpleNamespace(output={"tool_calls": calls}), loop_num=1)

    def test_an_answer_on_its_own_is_honored(self):
        _, action, answer = self._parse([self._call("provide_final_answer", {"thought": "t", "answer": "done"}, "f")])
        assert (action, answer) == ("final_answer", "done")

    @pytest.mark.parametrize("position", ["first", "last", "middle"])
    def test_a_batched_answer_is_declined_wherever_it_sits(self, position):
        tools = [self._call("probe", {"thought": "t", "i": i}, f"t{i}") for i in (1, 2)]
        final = self._call("provide_final_answer", {"thought": "t", "answer": "done"}, "f")
        batch = {"first": [final, *tools], "last": [*tools, final], "middle": [tools[0], final, tools[1]]}[position]

        _, action, _ = self._parse(batch)

        assert action == PARALLEL_TOOL_NAME, "the tools must run rather than be discarded"

    def test_a_single_tool_alongside_the_answer_still_runs(self):
        _, action, _ = self._parse(
            [
                self._call("provide_final_answer", {"thought": "t", "answer": "done"}, "f"),
                self._call("probe", {"thought": "t", "i": 1}, "a"),
            ]
        )
        assert action == "probe"


class TestStructuredOutputSurvivesOddJson:
    """`null`, a bare list, a quoted string and a missing `action_input` used to raise
    unrecoverable TypeError/KeyError, ending the run instead of asking the model again."""

    def _parse(self, payload):
        from dynamiq.nodes.agents.agent import Agent

        return Agent._handle_structured_output_mode(_make_agent(), payload, loop_num=1)

    @pytest.mark.parametrize("payload", ["null", "[1, 2]", '"just a string"'])
    def test_valid_json_that_is_not_an_object_is_recoverable(self, payload):
        with pytest.raises(ActionParsingException) as exc:
            self._parse(payload)
        assert exc.value.recoverable

    def test_missing_fields_are_named(self):
        with pytest.raises(ActionParsingException) as exc:
            self._parse(json.dumps({"thought": "t"}))
        assert "action" in str(exc.value) and "action_input" in str(exc.value)
        assert exc.value.recoverable

    def test_a_complete_response_still_parses(self):
        thought, action, action_input = self._parse(
            json.dumps({"thought": "t", "action": "probe", "action_input": {"i": 1}})
        )
        assert (thought, action, action_input) == ("t", "probe", {"i": 1})


class TestFailedToolResultsAreMarked:
    """The status is computed either way, but FUNCTION_CALLING mode used to drop it, so the
    same failure read as a plain result here and as a failure in every other mode."""

    def test_failures_are_tagged_and_successes_are_untouched(self):
        from dynamiq.nodes.agents.agent import TOOL_ERROR_PREFIX, Agent

        agent = TestEveryPendingIdIsAnswered._agent(["call_ok", "call_bad"])
        ordered_results = [
            {"tool_call_id": "call_ok", "tool_name": "Search", "result": "3 results", "success": True},
            {"tool_call_id": "call_bad", "tool_name": "Fetch", "result": "ToolExecutionException: 500",
             "success": False},
        ]

        Agent._emit_tool_observations(agent, tool_result="unused", ordered_results=ordered_results)

        ok, bad = agent._prompt.messages
        assert ok.content == "3 results", "a successful result must not be modified"
        assert bad.content == f"{TOOL_ERROR_PREFIX} ToolExecutionException: 500"

    def test_unreported_status_is_left_alone(self):
        """``None`` means the caller reported no status. Guessing would tag successes."""
        from dynamiq.nodes.agents.agent import TOOL_ERROR_PREFIX, Agent

        agent = TestEveryPendingIdIsAnswered._agent(["call_a"])
        ordered_results = [{"tool_call_id": "call_a", "tool_name": "Search", "result": "partial"}]

        Agent._emit_tool_observations(agent, tool_result="unused", ordered_results=ordered_results)

        assert agent._prompt.messages[0].content == "partial"
        assert TOOL_ERROR_PREFIX not in agent._prompt.messages[0].content

    def test_ids_answered_by_the_step_message_use_its_status(self):
        """A refused batch is a failure, and every id it answers must say so."""
        from dynamiq.nodes.agents.agent import TOOL_ERROR_PREFIX, Agent

        agent = TestEveryPendingIdIsAnswered._agent(["call_a", "call_b"])

        Agent._emit_tool_observations(agent, tool_result="Sub-agent invocation limit exceeded.", success=False)

        assert all(m.content.startswith(TOOL_ERROR_PREFIX) for m in agent._prompt.messages)

    def test_marker_is_not_stacked(self):
        """Repeated passes over the same content must not accumulate prefixes."""
        from dynamiq.nodes.agents.agent import TOOL_ERROR_PREFIX, mark_tool_failure

        once = mark_tool_failure("boom", False)
        assert mark_tool_failure(once, False) == once
        assert once.count(TOOL_ERROR_PREFIX) == 1


class TestNoToolCallIsDropped:
    """``parallel_tool_calls_enabled`` controls *how* a step's tool calls run, never
    *whether* they run. Driven against a scripted LLM, so the whole path is covered."""

    @staticmethod
    def _agent_with_probe(probe, **kwargs):
        import threading
        import time
        from typing import ClassVar

        from pydantic import BaseModel

        from dynamiq import connections
        from dynamiq.nodes import llms
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.node import Node, NodeGroup
        from dynamiq.nodes.types import InferenceMode

        lock = threading.Lock()

        class _Input(BaseModel):
            i: int = 0

        class ProbeTool(Node):
            group: NodeGroup = NodeGroup.TOOLS
            name: str = "probe"
            description: str = "records execution"
            input_schema: ClassVar[type[BaseModel]] = _Input
            is_parallel_execution_allowed: bool = True

            def execute(self, input_data, config=None, **kw):
                with lock:
                    probe["inflight"] += 1
                    probe["peak"] = max(probe["peak"], probe["inflight"])
                time.sleep(0.01)
                with lock:
                    probe["inflight"] -= 1
                    probe["order"].append(input_data.i)
                return {"content": f"done-{input_data.i}"}

        return Agent(
            id="agent",
            name="agent",
            llm=llms.OpenAI(
                id="llm",
                model="gpt-4o-mini",
                connection=connections.OpenAI(api_key="not-used"),
                is_postponed_component_init=True,
            ),
            tools=[ProbeTool()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            parallel_tool_calls_enabled=False,
            max_loops=3,
            **kwargs,
        )

    @staticmethod
    def _run(agent, n_calls):
        from litellm import ModelResponse

        turns = [
            [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "probe", "arguments": json.dumps({"thought": f"t{i}", "i": i})},
                }
                for i in range(n_calls)
            ],
            [
                {
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "provide_final_answer",
                        "arguments": json.dumps({"thought": "done", "answer": "ok"}),
                    },
                }
            ],
        ]
        state = {"n": 0}

        def _completion(stream=False, *a, **kw):
            calls = turns[min(state["n"], len(turns) - 1)]
            state["n"] += 1
            return ModelResponse(choices=[{"message": {"content": None, "tool_calls": calls}}])

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_completion):
            return agent.run(input_data={"input": "go"})

    def test_every_call_runs_when_parallel_is_disabled(self):
        """The batch used to execute one call and discard the rest."""
        probe = {"inflight": 0, "peak": 0, "order": []}
        agent = self._agent_with_probe(probe)

        self._run(agent, 5)

        assert probe["order"] == [0, 1, 2, 3, 4], "every call must run, in the order requested"
        assert probe["peak"] == 1, "disabled must still mean sequential, not concurrent"

    def test_every_tool_call_id_receives_a_reply(self):
        """Recording all N calls is only safe if all N get answered — an unanswered
        tool_call_id makes the next provider request invalid."""
        from dynamiq.prompts.prompts import MessageRole

        probe = {"inflight": 0, "peak": 0, "order": []}
        agent = self._agent_with_probe(probe)

        self._run(agent, 4)

        requested, answered = [], set()
        for m in agent._prompt.messages:
            if m.role == MessageRole.ASSISTANT and m.tool_calls:
                requested.extend(tc["id"] for tc in m.tool_calls)
            if m.role == MessageRole.TOOL and m.tool_call_id:
                answered.add(m.tool_call_id)

        assert requested, "expected the assistant message to record its tool calls"
        assert [r for r in requested if r not in answered] == []

    def test_each_reply_carries_its_own_call_result(self):
        """Answers must reach the call that produced them, not merely exist."""
        from dynamiq.prompts.prompts import MessageRole

        probe = {"inflight": 0, "peak": 0, "order": []}
        agent = self._agent_with_probe(probe)

        self._run(agent, 3)

        by_id = {m.tool_call_id: m.content for m in agent._prompt.messages if m.role == MessageRole.TOOL}
        for i in range(3):
            assert by_id[f"call_{i}"] == f"done-{i}"


class TestParallelToolConcurrencyIsBounded:
    """``max_parallel_tool_calls`` bounds the thread pool, so the model decides how many
    tools to call but not how many threads the process spawns."""

    def test_pool_is_capped_and_no_call_is_dropped(self):
        import threading
        import time
        from typing import ClassVar

        from pydantic import BaseModel

        from dynamiq import connections
        from dynamiq.nodes import llms
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.node import Node, NodeGroup
        from dynamiq.nodes.types import InferenceMode
        from dynamiq.runnables import RunnableConfig

        lock = threading.Lock()
        stats = {"inflight": 0, "peak": 0, "completed": 0}

        class _ProbeInput(BaseModel):
            i: int = 0

        class ProbeTool(Node):
            group: NodeGroup = NodeGroup.TOOLS
            name: str = "probe"
            description: str = "records concurrency"
            input_schema: ClassVar[type[BaseModel]] = _ProbeInput
            is_parallel_execution_allowed: bool = True

            def execute(self, input_data, config=None, **kwargs):
                with lock:
                    stats["inflight"] += 1
                    stats["peak"] = max(stats["peak"], stats["inflight"])
                time.sleep(0.02)
                with lock:
                    stats["inflight"] -= 1
                    stats["completed"] += 1
                return {"content": f"done-{input_data.i}"}

        agent = Agent(
            id="agent",
            name="agent",
            llm=llms.OpenAI(
                id="llm",
                model="gpt-4o-mini",
                connection=connections.OpenAI(api_key="not-used"),
                is_postponed_component_init=True,
            ),
            tools=[ProbeTool()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            parallel_tool_calls_enabled=True,
            max_parallel_tool_calls=3,
        )

        tools_data = [{"name": "probe", "input": {"i": i}, "thought": f"t{i}"} for i in range(12)]
        agent._execute_tools(tools_data, "batch", 1, RunnableConfig())

        assert stats["completed"] == 12, "capping must queue calls, never drop them"
        assert stats["peak"] <= 3, "more tools ran at once than max_parallel_tool_calls allows"


class TestDelegateFinalMustStandAlone:
    """``delegate_final`` answers for the whole step, so it is only honoured when it is the
    step's only call. Batched with others it is refused before it runs, since the answer is
    streamed as it returns and the loop would then produce a second one."""

    @staticmethod
    def _parent_with_sub_agent(probe, **kwargs):
        from typing import ClassVar

        from pydantic import BaseModel

        from dynamiq import connections
        from dynamiq.nodes import llms
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.node import Node, NodeGroup
        from dynamiq.nodes.tools.agent_tool import SubAgentTool
        from dynamiq.nodes.types import InferenceMode

        class _Input(BaseModel):
            i: int = 0

        class ProbeTool(Node):
            group: NodeGroup = NodeGroup.TOOLS
            name: str = "probe"
            description: str = "records execution"
            input_schema: ClassVar[type[BaseModel]] = _Input

            def execute(self, input_data, config=None, **kw):
                probe["ran"].append(input_data.i)
                return {"content": f"done-{input_data.i}"}

        def _llm(node_id):
            return llms.OpenAI(
                id=node_id,
                model="gpt-4o-mini",
                connection=connections.OpenAI(api_key="not-used"),
                is_postponed_component_init=True,
            )

        child = Agent(id="child", name="child", llm=_llm("child-llm"), tools=[], max_loops=2)

        return Agent(
            id="agent",
            name="agent",
            llm=_llm("llm"),
            tools=[ProbeTool(), SubAgentTool(agent=child, name="sub_agent", description="delegates")],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            parallel_tool_calls_enabled=False,
            delegation_allowed=True,
            max_loops=3,
            **kwargs,
        )

    @staticmethod
    def _run(agent, first_turn):
        from litellm import ModelResponse

        turns = [
            first_turn,
            [
                {
                    "id": "call_final",
                    "type": "function",
                    "function": {
                        "name": "provide_final_answer",
                        "arguments": json.dumps({"thought": "done", "answer": "MODEL ANSWER"}),
                    },
                }
            ],
        ]
        state = {"n": 0}

        def _completion(stream=False, *a, **kw):
            calls = turns[min(state["n"], len(turns) - 1)]
            state["n"] += 1
            return ModelResponse(choices=[{"message": {"content": None, "tool_calls": calls}}])

        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_completion) as completion:
            result = agent.run(input_data={"input": "go"})
        return result, state, completion

    def test_a_batched_delegation_is_refused_and_never_runs(self):
        """The sub-agent must not be invoked at all: running it and then discarding the flag
        still streams its answer and bills the call."""
        from dynamiq.prompts.prompts import MessageRole

        probe = {"ran": []}
        agent = self._parent_with_sub_agent(probe)

        batch = [
            {
                "id": "call_delegate",
                "type": "function",
                "function": {
                    "name": "sub_agent",
                    "arguments": json.dumps({"thought": "t", "input": "do it", "delegate_final": True}),
                },
            },
            {
                "id": "call_probe",
                "type": "function",
                "function": {"name": "probe", "arguments": json.dumps({"thought": "t", "i": 7})},
            },
        ]
        result, state, completion = self._run(agent, batch)

        by_id = {m.tool_call_id: m.content for m in agent._prompt.messages if m.role == MessageRole.TOOL}
        assert "was not executed" in by_id["call_delegate"], "the refusal must reach the call that asked"
        assert probe["ran"] == [7], "the siblings of a refused delegation must still run"
        assert "done-7" in by_id["call_probe"]
        assert state["n"] == 2, "the sub-agent must never have been invoked"
        assert result.output["content"] == "MODEL ANSWER", "the run must end on the model's own single answer"

    def test_a_lone_delegation_still_ends_the_run_with_its_result(self):
        """The flag survives the batch path when nothing else was called — dropping it here
        is what let the loop run on and produce a second answer."""
        from dynamiq.runnables import RunnableConfig

        probe = {"ran": []}
        agent = self._parent_with_sub_agent(probe)

        with patch.object(type(agent), "_execute_single_tool", return_value=("SUB ANSWER", [], True, True, None)):
            answer = agent._execute_tools_and_update_prompt(
                PARALLEL_TOOL_NAME,
                {"tools": [{"name": "sub_agent", "input": {"input": "do it", "delegate_final": True}, "thought": "t"}]},
                "t",
                1,
                RunnableConfig(),
            )

        assert answer == "SUB ANSWER", "a delegated result must end the loop, not become an observation"


class TestAgentOwnedFieldsStayOutOfThePrompt:
    """``tool_call_id`` is agent bookkeeping. The modes shown ``run-parallel`` author its
    argument themselves and have no provider ids, so the field must not be described."""

    def test_tool_call_id_is_not_described_to_the_model(self):
        from dynamiq.nodes.agents.components.schema_generator import generate_input_formats
        from dynamiq.nodes.tools.parallel_tool_calls import ParallelToolCallsTool

        rendered = generate_input_formats([ParallelToolCallsTool()], lambda n: n)

        assert "tool_call_id" not in rendered, f"agent-owned field reached the prompt: {rendered}"
        for shown in ("name:", "input:", "thought:"):
            assert shown in rendered, f"{shown} must still be described, got {rendered}"

    def test_the_field_still_exists_for_the_agent_to_set(self):
        """Hiding it from the description must not stop FUNCTION_CALLING from carrying it;
        ``extra='forbid'`` would reject the id outright if the field were removed."""
        from dynamiq.nodes.tools.parallel_tool_calls import ToolCallItem

        item = ToolCallItem(name="probe", input={"i": 1}, thought="t", tool_call_id="call_1")

        assert item.model_dump()["tool_call_id"] == "call_1"

    def test_nested_visible_fields_are_still_rendered(self):
        """The filter is per-field, not a blanket skip of the nested branch."""
        from typing import Any, ClassVar

        from pydantic import BaseModel, Field

        from dynamiq.nodes.agents.components.schema_generator import generate_input_formats
        from dynamiq.nodes.node import Node, NodeGroup

        class _Item(BaseModel):
            shown: str = Field(..., description="visible one")
            hidden: str | None = Field(default=None, json_schema_extra={"is_accessible_to_agent": False})

        class _Schema(BaseModel):
            items: list[_Item] = Field(..., description="the items")

        class _Tool(Node):
            group: NodeGroup = NodeGroup.TOOLS
            name: str = "probe"
            description: str = "d"
            input_schema: ClassVar[type[BaseModel]] = _Schema

            def execute(self, input_data, config=None, **kw) -> Any:
                return {}

        rendered = generate_input_formats([_Tool()], lambda n: n)

        assert "shown: str - visible one" in rendered, rendered
        assert "hidden" not in rendered, rendered


class TestParallelToolInjectionSeesEveryTool:
    """``run-parallel`` is withheld only from an agent with nothing to batch, so the
    emptiness check must run after ``SkillsTool`` is appended."""

    @staticmethod
    def _agent(tools, **kwargs):
        from dynamiq import connections
        from dynamiq.nodes import llms
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.types import InferenceMode

        return Agent(
            id="agent",
            name="agent",
            llm=llms.OpenAI(
                id="llm",
                model="gpt-4o-mini",
                connection=connections.OpenAI(api_key="not-used"),
                is_postponed_component_init=True,
            ),
            tools=tools,
            inference_mode=InferenceMode.XML,
            parallel_tool_calls_enabled=True,
            **kwargs,
        )

    @staticmethod
    def _skills():
        from dynamiq.skills.config import SkillsConfig
        from dynamiq.skills.registries import FileSystem
        from dynamiq.skills.registries.filesystem import FileSystemSkillEntry

        return SkillsConfig(
            enabled=True,
            source=FileSystem(base_path="/nonexistent", skills=[FileSystemSkillEntry(name="a", description="A")]),
        )

    def test_a_skills_only_agent_can_still_batch(self):
        agent = self._agent([], skills=self._skills())

        names = [t.name for t in agent.tools]
        assert "skills-tool" in names, f"expected SkillsTool among {names}"
        assert PARALLEL_TOOL_NAME in names, f"a skills-only agent has a tool to batch, got {names}"

    def test_a_genuinely_tool_less_agent_is_left_alone(self):
        agent = self._agent([])

        assert agent.tools == [], "nothing to batch means no run-parallel and no tools prompt block"
