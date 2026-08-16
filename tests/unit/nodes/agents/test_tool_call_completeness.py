"""Every tool call the model asks for must run, and concurrency must stay bounded.

``parallel_tool_calls_enabled`` controls *how* a step's tool calls run, never *whether*
they run. These tests drive a real ``Agent`` against a scripted LLM so they exercise the
whole path — parsing, batching, execution, and the messages the model reads back.
"""

import json
import threading
import time
from typing import ClassVar
from unittest.mock import patch

import pytest
from litellm import ModelResponse
from pydantic import BaseModel, Field

from dynamiq import connections
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.agent import TOOL_ERROR_PREFIX
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableConfig, RunnableStatus


class _ToolInput(BaseModel):
    i: int = Field(default=0)


class ConcurrencyProbe:
    """Records the true peak number of tools executing at the same instant."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.completed = 0
        self.order: list[int] = []

    def enter(self, i: int):
        with self.lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            self.order.append(i)

    def leave(self):
        with self.lock:
            self.inflight -= 1
            self.completed += 1


@pytest.fixture
def probe():
    return ConcurrencyProbe()


def _make_tool(
    probe: ConcurrencyProbe,
    hold: float = 0.05,
    parallel_ok: bool = True,
    barrier: threading.Barrier | None = None,
):
    """Build a tool that records concurrency.

    With a ``barrier``, each call blocks until the expected number of calls are in
    flight together. That turns "did these really run at the same time?" into a
    deterministic question, instead of one that depends on how loaded the machine is —
    CI runs the suite under `pytest -n auto`, where a sleep-and-measure check would
    flake.
    """

    class ProbeTool(Node):
        group: NodeGroup = NodeGroup.TOOLS
        name: str = "probe"
        description: str = "records concurrency"
        input_schema: ClassVar[type[BaseModel]] = _ToolInput
        is_parallel_execution_allowed: bool = parallel_ok

        def execute(self, input_data, config: RunnableConfig = None, **kwargs):
            probe.enter(input_data.i)
            try:
                if barrier is not None:
                    barrier.wait(timeout=10)
                else:
                    time.sleep(hold)
            finally:
                probe.leave()
            return {"content": f"done-{input_data.i}"}

    return ProbeTool()


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _scripted_llm(turns: list[list[dict]]):
    """Return a ``_completion`` side effect that replays ``turns``, then repeats the last."""
    state = {"n": 0}

    def side_effect(stream=False, *args, **kwargs):
        calls = turns[min(state["n"], len(turns) - 1)]
        state["n"] += 1
        return ModelResponse(choices=[{"message": {"content": None, "tool_calls": calls}}])

    return side_effect


def _build_agent(tool: Node, **agent_kwargs) -> Agent:
    return Agent(
        id="agent",
        name="agent",
        llm=llms.OpenAI(
            id="llm",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="not-used"),
            is_postponed_component_init=True,
        ),
        tools=[tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=3,
        **agent_kwargs,
    )


def _build_agent_no_tools(mode: InferenceMode, **agent_kwargs) -> Agent:
    """An agent with no tools, so the tool list reflects only what the agent injects."""
    return Agent(
        id="agent",
        name="agent",
        llm=llms.OpenAI(
            id="llm",
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="not-used"),
            is_postponed_component_init=True,
        ),
        tools=[],
        inference_mode=mode,
        **agent_kwargs,
    )


def _run(agent: Agent, n_calls: int):
    turns = [
        [_tool_call("probe", {"thought": f"t{i}", "i": i}, f"call_{i}") for i in range(n_calls)],
        [_tool_call("provide_final_answer", {"thought": "done", "answer": "ok"}, "call_final")],
    ]
    with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_scripted_llm(turns)):
        return agent.run(input_data={"input": "go"})


class TestNoToolCallIsDropped:
    @pytest.mark.parametrize("n_calls", [2, 5, 12])
    def test_all_calls_execute_when_parallel_disabled(self, probe, n_calls):
        """Regression: the batch used to execute one call and discard the rest."""
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=False)
        result = _run(agent, n_calls)

        assert result.status == RunnableStatus.SUCCESS
        assert probe.completed == n_calls

    def test_disabled_means_sequential_not_dropped(self, probe):
        """The flag still suppresses concurrency — it just no longer loses work."""
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=False)
        _run(agent, 6)

        assert probe.completed == 6
        assert probe.peak == 1, "parallel_tool_calls_enabled=False must not run tools concurrently"

    def test_every_tool_call_id_receives_a_reply(self, probe):
        """An unanswered tool_call_id makes the next provider request invalid."""
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=False)
        _run(agent, 4)

        requested, answered = [], set()
        for msg in agent._prompt.messages:
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                requested.extend(tc["id"] for tc in msg.tool_calls)
            if msg.role == MessageRole.TOOL and msg.tool_call_id:
                answered.add(msg.tool_call_id)

        assert requested, "expected the assistant message to record its tool calls"
        assert [r for r in requested if r not in answered] == []

    def test_results_are_returned_in_the_requested_order(self, probe):
        """Sequential execution must preserve the model's ordering."""
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=False)
        _run(agent, 5)

        assert probe.order == [0, 1, 2, 3, 4]


class TestEveryPendingIdIsClosedOut:
    """A request carrying an unanswered tool_call_id is rejected by the provider, so any
    path that ends a batch early must still reply to the calls it did not run."""

    def _emit(self, pending_ids, ordered_results=None, tool_result="combined", success=None, not_executed_reason=None):
        from unittest.mock import MagicMock

        from dynamiq.nodes.agents.agent import Agent

        agent = MagicMock()
        agent.inference_mode = InferenceMode.FUNCTION_CALLING
        agent._prompt = MagicMock()
        agent._prompt.messages = []
        agent._pending_fc_tool_call_ids = list(pending_ids)

        Agent._emit_tool_observations(
            agent,
            tool_result=tool_result,
            ordered_results=ordered_results,
            tool_name="t",
            success=success,
            not_executed_reason=not_executed_reason,
        )
        return agent

    def test_unmatched_ids_receive_a_not_executed_reply(self):
        from dynamiq.nodes.agents.agent import TOOL_NOT_EXECUTED_PREFIX

        agent = self._emit(["a", "b", "c"])

        replies = {m.tool_call_id: m.content for m in agent._prompt.messages}
        assert set(replies) == {"a", "b", "c"}
        assert replies["b"].startswith(TOOL_NOT_EXECUTED_PREFIX)
        assert replies["c"].startswith(TOOL_NOT_EXECUTED_PREFIX)

    def test_the_reason_reaches_the_model(self):
        """A specific cause is actionable; "something happened" only invites a blind retry."""
        agent = self._emit(["a", "b"], not_executed_reason="the budget is exhausted")

        replies = {m.tool_call_id: m.content for m in agent._prompt.messages}
        assert "the budget is exhausted" in replies["b"]

    def test_matched_results_are_unaffected(self):
        ordered = [
            {"tool_name": "x", "result": "r1", "success": True},
            {"tool_name": "y", "result": "r2", "success": True},
        ]
        agent = self._emit(["a", "b"], ordered_results=ordered)

        replies = {m.tool_call_id: m.content for m in agent._prompt.messages}
        assert replies == {"a": "r1", "b": "r2"}

    def test_not_executed_notice_is_not_marked_as_an_error(self):
        """It is outstanding work, not a failure — a double marker would mislead."""
        from dynamiq.nodes.agents.agent import TOOL_NOT_EXECUTED_PREFIX

        agent = self._emit(["a", "b"])
        replies = {m.tool_call_id: m.content for m in agent._prompt.messages}

        assert not replies["b"].startswith(TOOL_ERROR_PREFIX)
        assert replies["b"].startswith(TOOL_NOT_EXECUTED_PREFIX)

    def test_stash_is_cleared(self):
        agent = self._emit(["a", "b"])
        assert agent._pending_fc_tool_call_ids == []


class TestConcurrencyIsBounded:
    def test_peak_concurrency_respects_the_cap(self, probe):
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=True, max_parallel_tool_calls=3)
        _run(agent, 12)

        assert probe.completed == 12, "capping must queue calls, never drop them"
        assert probe.peak <= 3

    def test_default_cap_is_eight(self, probe):
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=True)
        assert agent.max_parallel_tool_calls == 8

        _run(agent, 20)
        assert probe.completed == 20
        assert probe.peak <= 8

    def test_small_batches_are_unaffected_by_the_cap(self, probe):
        """A batch below the cap keeps running fully concurrently.

        The barrier only releases once all four calls are in flight together, so this
        fails loudly if the cap ever starts throttling batches it should not.
        """
        barrier = threading.Barrier(4)
        agent = _build_agent(
            _make_tool(probe, barrier=barrier), parallel_tool_calls_enabled=True, max_parallel_tool_calls=8
        )
        result = _run(agent, 4)

        assert result.status == RunnableStatus.SUCCESS
        assert probe.completed == 4
        assert probe.peak == 4

    def test_cap_must_be_at_least_one(self):
        with pytest.raises(ValueError):
            _build_agent(_make_tool(ConcurrencyProbe()), max_parallel_tool_calls=0)

    def test_tool_opting_out_still_runs_sequentially(self, probe):
        """Per-tool opt-out is independent of the agent-level flag."""
        agent = _build_agent(_make_tool(probe, parallel_ok=False), parallel_tool_calls_enabled=True)
        _run(agent, 6)

        assert probe.completed == 6
        assert probe.peak == 1


class TestFailedToolResultsAreMarked:
    def test_failure_is_marked_and_success_is_untouched(self):
        """A recoverable tool failure keeps the loop alive, so the model reads the
        result and must be able to see that it failed."""

        class FlakyInput(BaseModel):
            i: int = Field(default=0)

        class FlakyTool(Node):
            group: NodeGroup = NodeGroup.TOOLS
            name: str = "flaky"
            description: str = "fails on odd i"
            input_schema: ClassVar[type[BaseModel]] = FlakyInput
            is_parallel_execution_allowed: bool = True

            def execute(self, input_data, config: RunnableConfig = None, **kwargs):
                if input_data.i % 2:
                    raise ToolExecutionException("upstream returned 502", recoverable=True)
                return {"content": f"ok-{input_data.i}"}

        agent = _build_agent(FlakyTool(), parallel_tool_calls_enabled=True)
        turns = [
            [_tool_call("flaky", {"thought": "t", "i": i}, f"call_{i}") for i in range(2)],
            [_tool_call("provide_final_answer", {"thought": "d", "answer": "ok"}, "call_final")],
        ]
        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_scripted_llm(turns)):
            agent.run(input_data={"input": "go"})

        by_id = {m.tool_call_id: m.content for m in agent._prompt.messages if m.role == MessageRole.TOOL}

        assert by_id["call_0"] == "ok-0", "a successful result must not be rewritten"
        assert by_id["call_1"].startswith(TOOL_ERROR_PREFIX)
        assert "502" in by_id["call_1"], "the original error text must survive the marker"


class TestMarkToolFailureHelper:
    """Unit-level edge cases for the marker itself."""

    def test_marks_only_explicit_failure(self):
        from dynamiq.nodes.agents.agent import mark_tool_failure

        assert mark_tool_failure("boom", False) == f"{TOOL_ERROR_PREFIX} boom"
        assert mark_tool_failure("fine", True) == "fine"

    def test_unknown_status_is_left_alone(self):
        """``None`` means "not reported" and must not be guessed at as a failure."""
        from dynamiq.nodes.agents.agent import mark_tool_failure

        assert mark_tool_failure("no status", None) == "no status"

    def test_marker_is_not_applied_twice(self):
        from dynamiq.nodes.agents.agent import mark_tool_failure

        once = mark_tool_failure("boom", False)
        assert mark_tool_failure(once, False) == once

    def test_empty_content_still_marked(self):
        from dynamiq.nodes.agents.agent import mark_tool_failure

        assert mark_tool_failure("", False) == f"{TOOL_ERROR_PREFIX} "


class TestFinalAnswerBatchedWithToolCalls:
    """An answer batched with tool calls was written before those results existed, so it
    cannot have used them. Returning it hands back a guess and discards the work; the
    tools run instead and the model answers on the next loop.

    Position within the batch carries no meaning — llama-3.3-70b puts the answer last,
    gpt-4o puts it first — so it is found by name, never by index.
    """

    def _parse(self, calls):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from dynamiq.nodes.agents.agent import Agent

        agent = MagicMock()
        agent.verbose = False
        agent.sanitize_tool_name = lambda n: n
        agent._parse_output_files_csv = lambda v: []
        agent.tool_by_names = {}
        agent.parallel_tool_calls_enabled = True
        return Agent._handle_function_calling_mode(agent, SimpleNamespace(output={"tool_calls": calls}), loop_num=1)

    @staticmethod
    def _call(name, args, call_id):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def test_final_answer_alone_is_honored(self):
        _, action, answer = self._parse([self._call("provide_final_answer", {"thought": "t", "answer": "done"}, "f")])
        assert action == "final_answer"
        assert answer == "done"

    @pytest.mark.parametrize("position", ["first", "last", "middle"])
    def test_batched_answer_is_declined_wherever_it_sits(self, position):
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

    def test_batch_without_a_final_answer_still_batches(self):
        _, action, _ = self._parse(
            [
                self._call("probe", {"thought": "t", "i": 1}, "a"),
                self._call("probe", {"thought": "t", "i": 2}, "b"),
            ]
        )
        assert action == PARALLEL_TOOL_NAME

    def test_the_model_is_told_why_the_finish_was_declined(self, probe):
        """Silently dropping the answer leaves the model repeating the same batch until
        max_loops; the stub reply is what turns the next loop into an informed answer."""
        agent = _build_agent(_make_tool(probe), parallel_tool_calls_enabled=True)
        turns = [
            [
                _tool_call("probe", {"thought": "t", "i": 0}, "call_0"),
                _tool_call("provide_final_answer", {"thought": "guess", "answer": "guessed"}, "call_fa"),
            ],
            [_tool_call("provide_final_answer", {"thought": "informed", "answer": "real"}, "call_final")],
        ]
        with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=_scripted_llm(turns)):
            result = agent.run(input_data={"input": "go"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "real", "the guess must not be what the caller gets"
        assert probe.completed == 1, "the batched tool call must still run"

        by_id = {m.tool_call_id: m.content for m in agent._prompt.messages if m.role == MessageRole.TOOL}
        assert by_id["call_0"] == "done-0"
        assert by_id["call_fa"].startswith("Not accepted")
        assert by_id["call_final"] == "Acknowledged.", "an unbatched answer is still just acknowledged"


class TestParallelDefaultFollowsInferenceMode:
    """Concurrency is free where the provider batches the calls itself; the other modes
    would need an injected tool, so they stay opt-in."""

    def _agent(self, mode, **kw):
        return _build_agent_no_tools(mode, **kw)

    def test_function_calling_defaults_to_parallel(self):
        agent = self._agent(InferenceMode.FUNCTION_CALLING)
        assert agent.parallel_tool_calls_enabled is True

    @pytest.mark.parametrize("mode", [InferenceMode.DEFAULT, InferenceMode.XML, InferenceMode.STRUCTURED_OUTPUT])
    def test_other_modes_default_to_sequential(self, mode):
        assert self._agent(mode).parallel_tool_calls_enabled is False

    def test_no_tool_is_injected_by_the_new_default(self):
        """The point of scoping the default to FUNCTION_CALLING: no prompt change."""
        for mode in (InferenceMode.FUNCTION_CALLING, InferenceMode.DEFAULT, InferenceMode.XML):
            assert [t.name for t in self._agent(mode).tools] == []

    def test_explicit_false_wins_in_function_calling(self):
        agent = self._agent(InferenceMode.FUNCTION_CALLING, parallel_tool_calls_enabled=False)
        assert agent.parallel_tool_calls_enabled is False

    def test_explicit_true_wins_in_react(self):
        agent = self._agent(InferenceMode.DEFAULT, parallel_tool_calls_enabled=True)
        assert agent.parallel_tool_calls_enabled is True
        assert "run-parallel" in [t.name for t in agent.tools]
