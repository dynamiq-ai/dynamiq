"""Integration tests for loop-level (iterative) checkpoint/resume.

Tests cover:
- IterativeCheckpointMixin protocol
- Agent mid-loop checkpoint saves iteration state
- Agent resume skips completed loops
- Simulated crash-and-resume flow
- Backward compatibility with old checkpoints
- Round-trip serialization through InMemory and FileSystem backends
- Dependent node skip and failure propagation with checkpoints
- Agent timeout mid-loop with checkpoint state
- Orchestrator checkpoint state save/restore
- SummarizerTool checkpoint in flow context
- Multiple backend types via parametrize
"""

import pytest
from litellm import ModelResponse

from dynamiq import connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointConfig, CheckpointStatus
from dynamiq.nodes import ErrorHandling, llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.llm_summarizer import SummarizerTool
from dynamiq.nodes.types import Behavior
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableStatus

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"


def make_agent_llm(node_id: str = "agent-llm") -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


def make_agent_flow(backend, *, mid_loop: bool = True, max_loops: int = 5, flow_id: str | None = None):
    calc_tool = Python(
        id="calc-tool",
        name="calculator",
        description="Calculator tool for math operations",
        code='def run(input_data): return {"result": 42}',
    )
    agent = Agent(
        id="agent",
        name="ReAct Agent",
        llm=make_agent_llm(),
        tools=[calc_tool],
        role="Math assistant",
        max_loops=max_loops,
    )
    kwargs = {}
    if flow_id:
        kwargs["id"] = flow_id
    flow = flows.Flow(
        nodes=[agent],
        checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop=mid_loop),
        **kwargs,
    )
    return flow, agent


def mock_react_loop(mocker, tool_calls: int = 1, final_answer: str = "The result is 42."):
    """Mock LLM to simulate ReAct loop: N tool calls then final answer."""
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] <= tool_calls:
            r["choices"][0]["message"]["content"] = (
                f"Thought: Step {call_count['value']}.\n"
                f"Action: calculator\n"
                f'Action Input: {{"step": {call_count["value"]}}}'
            )
        else:
            r["choices"][0]["message"]["content"] = f"Thought: Done.\nFinal Answer: {final_answer}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_react_loop_crash_at(mocker, crash_after_loop: int, final_answer: str = "Resumed answer"):
    """Mock LLM that crashes after N tool-call loops.

    Returns (call_count, CrashError class) so the caller can catch the exception.
    """

    class CrashError(RuntimeError):
        pass

    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] > crash_after_loop:
            raise CrashError(f"Simulated crash after loop {crash_after_loop}")
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = (
            f"Thought: Step {call_count['value']}.\n"
            f"Action: calculator\n"
            f'Action Input: {{"step": {call_count["value"]}}}'
        )
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count, CrashError


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


class TestAgentMidLoopIterationCheckpoint:
    """Verify that mid-loop checkpoints capture iteration state correctly."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_mid_loop_checkpoint_captures_completed_loops(self, mocker, backend_factory, backend_type):
        """After 3 tool calls, the final checkpoint should have iteration data with completed_loops >= 1."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=3)

        result = flow.run_sync(input_data={"input": "Do 3 calculations"})
        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        internal = cp.node_states["agent"].internal_state
        assert "iteration" in internal
        assert internal["iteration"]["completed_iterations"] >= 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_mid_loop_checkpoint_captures_prompt_messages(self, mocker, backend_factory, backend_type):
        """iteration_data should contain prompt_messages with conversation history."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        flow.run_sync(input_data={"input": "Calculate step by step"})

        cp = backend.get_latest_by_flow(flow.id)
        iteration = cp.node_states["agent"].internal_state.get("iteration", {})
        iteration_data = iteration.get("iteration_data", {})
        prompt_messages = iteration_data.get("prompt_messages")
        assert prompt_messages is not None
        assert len(prompt_messages) > 2

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_mid_loop_checkpoint_captures_agent_state(self, mocker, backend_factory, backend_type):
        """iteration_data should contain agent_state with loop counter."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        flow.run_sync(input_data={"input": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        iteration = cp.node_states["agent"].internal_state.get("iteration", {})
        agent_state = iteration.get("iteration_data", {}).get("agent_state")
        assert agent_state is not None
        assert "current_loop" in agent_state
        assert "max_loops" in agent_state

    def test_no_iteration_data_without_mid_loop_flag(self, mocker):
        """Without checkpoint_mid_agent_loop=True, no mid-run iteration saves occur."""
        backend = InMemory()
        flow, agent = make_agent_flow(backend, mid_loop=False)
        mock_react_loop(mocker, tool_calls=2)

        flow.run_sync(input_data={"input": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        internal = cp.node_states["agent"].internal_state
        iteration = internal.get("iteration")
        if iteration:
            assert iteration["completed_iterations"] >= 0


class TestAgentCrashAndResume:
    """Simulate agent crash mid-loop and resume from checkpoint.

    In a real crash scenario, the process dies between mid-loop save_mid_run()
    and the flow's failure handler. The checkpoint has the agent with status="active"
    (not completed), so on resume the agent is re-scheduled. We simulate this by
    running the agent successfully to get mid-loop checkpoints, then manipulating
    the checkpoint to remove the agent from completed_node_ids.
    """

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_resume_skips_completed_loops(self, mocker, backend_factory, backend_type):
        """Agent completes 3 loops, then we simulate crash by reverting checkpoint to active.
        On resume, it should start from loop 4, not loop 1."""
        backend = backend_factory(backend_type)

        flow1, agent1 = make_agent_flow(backend, mid_loop=True, max_loops=10)
        mock_react_loop(mocker, tool_calls=3, final_answer="Done after 3 calls")
        result1 = flow1.run_sync(input_data={"input": "Do calculations"})
        assert result1.status == RunnableStatus.SUCCESS
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow1.id)
        assert "agent" in cp.node_states
        assert cp.node_states["agent"].internal_state.get("iteration") is not None

        # Simulate crash: revert agent to "active" status, remove from completed
        cp.node_states["agent"].status = CheckpointStatus.ACTIVE.value
        cp.node_states["agent"].output_data = None
        cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != "agent"]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        # Resume: LLM returns final answer immediately
        flow2, agent2 = make_agent_flow(backend, mid_loop=True, max_loops=10, flow_id=flow1.id)
        resume_call_count = mock_react_loop(mocker, tool_calls=0, final_answer="Resumed answer")

        result2 = flow2.run_sync(input_data=None, resume_from=cp.id)
        assert result2.status == RunnableStatus.SUCCESS

        # The LLM should have been called only 1 time (final answer), not 3+1 again
        assert resume_call_count["value"] == 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_resume_preserves_conversation_history(self, mocker, backend_factory, backend_type):
        """After resume, prompt messages from before the crash should be present."""
        backend = backend_factory(backend_type)

        flow1, agent1 = make_agent_flow(backend, mid_loop=True, max_loops=10)
        mock_react_loop(mocker, tool_calls=2, final_answer="Result")
        flow1.run_sync(input_data={"input": "Calculate stuff"})
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow1.id)
        iteration = cp.node_states.get("agent")
        assert iteration is not None
        iter_data = iteration.internal_state.get("iteration", {}).get("iteration_data", {})
        prompt_msgs = iter_data.get("prompt_messages", [])
        assert len(prompt_msgs) >= 4

    def test_resume_with_fresh_checkpoint_starts_from_loop_1(self, mocker):
        """Resuming from a checkpoint without iteration data starts from loop 1 (backward compat)."""
        backend = InMemory()
        flow, agent = make_agent_flow(backend, mid_loop=True, max_loops=5)
        mock_react_loop(mocker, tool_calls=1)

        flow.run_sync(input_data={"input": "test"})
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow.id)
        if "agent" in cp.node_states:
            cp.node_states["agent"].internal_state.pop("iteration", None)
            cp.node_states["agent"].status = CheckpointStatus.ACTIVE.value
            cp.node_states["agent"].output_data = None
            cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != "agent"]
            cp.status = CheckpointStatus.ACTIVE
            backend.save(cp)

        flow2, agent2 = make_agent_flow(backend, mid_loop=True, max_loops=5, flow_id=flow.id)
        resume_count = mock_react_loop(mocker, tool_calls=1, final_answer="Fresh start")

        result = flow2.run_sync(input_data=None, resume_from=cp.id)
        assert result.status == RunnableStatus.SUCCESS
        assert resume_count["value"] >= 2


class TestIterationStateSerialization:
    """Verify IterationState survives JSON/backend round-trips."""

    def test_iteration_state_in_flow_checkpoint_roundtrip(self, mocker):
        """Full flow checkpoint → save → load preserves iteration data."""
        backend = InMemory()
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        flow.run_sync(input_data={"input": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        cp_json = cp.to_json()

        from dynamiq.checkpoints.checkpoint import FlowCheckpoint

        restored_cp = FlowCheckpoint.from_json(cp_json)
        assert "agent" in restored_cp.node_states
        internal = restored_cp.node_states["agent"].internal_state
        assert "iteration" in internal

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_iteration_state_survives_backend_roundtrip(self, mocker, backend_factory, backend_type):
        """Save checkpoint to backend, load it back, iteration data intact."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        flow.run_sync(input_data={"input": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        loaded = backend.load(cp.id)
        assert loaded is not None
        internal = loaded.node_states["agent"].internal_state
        assert "iteration" in internal
        assert internal["iteration"]["completed_iterations"] >= 1


class TestEdgeCases:
    """Edge cases for iterative checkpointing."""

    def test_agent_with_zero_tool_calls_has_zero_completed_loops(self, mocker):
        """Agent that answers immediately (no tool calls) has completed_loops=0."""
        backend = InMemory()
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=0, final_answer="Immediate answer")

        result = flow.run_sync(input_data={"input": "Simple question"})
        assert result.status == RunnableStatus.SUCCESS

    def test_multiple_resume_cycles(self, mocker):
        """Agent can be checkpointed and resumed multiple times via simulated crash."""
        backend = InMemory()

        # Run 1: complete 2 loops, then simulate crash
        flow1, _ = make_agent_flow(backend, mid_loop=True, max_loops=15)
        mock_react_loop(mocker, tool_calls=2, final_answer="Phase 1 done")
        flow1.run_sync(input_data={"input": "Long task"})
        mocker.stopall()

        cp1 = backend.get_latest_by_flow(flow1.id)
        cp1.node_states["agent"].status = CheckpointStatus.ACTIVE.value
        cp1.node_states["agent"].output_data = None
        cp1.completed_node_ids = [nid for nid in cp1.completed_node_ids if nid != "agent"]
        cp1.status = CheckpointStatus.ACTIVE
        backend.save(cp1)

        # Run 2: resume, complete 2 more loops, simulate crash again
        flow2, _ = make_agent_flow(backend, mid_loop=True, max_loops=15, flow_id=flow1.id)
        mock_react_loop(mocker, tool_calls=2, final_answer="Phase 2 done")
        flow2.run_sync(input_data=None, resume_from=cp1.id)
        mocker.stopall()

        cp2 = backend.get_latest_by_flow(flow2.id)
        cp2.node_states["agent"].status = CheckpointStatus.ACTIVE.value
        cp2.node_states["agent"].output_data = None
        cp2.completed_node_ids = [nid for nid in cp2.completed_node_ids if nid != "agent"]
        cp2.status = CheckpointStatus.ACTIVE
        backend.save(cp2)

        # Run 3: resume and finish
        flow3, _ = make_agent_flow(backend, mid_loop=True, max_loops=15, flow_id=flow1.id)
        mock_react_loop(mocker, tool_calls=0, final_answer="Finally done")
        result = flow3.run_sync(input_data=None, resume_from=cp2.id)
        assert result.status == RunnableStatus.SUCCESS


class TestSkipAndFailureMidPipeline:
    """Dependent node skip/failure scenarios: checkpoint captures partial progress correctly."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_dependent_node_skipped_when_upstream_fails_raise(self, mocker, backend_factory, backend_type):
        """Input -> FailingPython(RAISE) -> Output: Output is skipped, checkpoint captures both states."""
        backend = backend_factory(backend_type)

        inp = Input(id="input", name="Input")
        failing = Python(
            id="failing",
            name="Failing",
            code='def run(input_data): raise ValueError("boom")',
            error_handling=ErrorHandling(behavior=Behavior.RAISE),
            depends=[NodeDependency(inp)],
        )
        out = Output(id="output", name="Output", depends=[NodeDependency(failing)])

        flow = flows.Flow(
            nodes=[inp, failing, out],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert "input" in cp.completed_node_ids
        assert cp.node_states["input"].status == "success"
        assert "failing" in cp.completed_node_ids
        assert cp.node_states["failing"].status == "failure"
        assert "output" in cp.completed_node_ids
        assert cp.node_states["output"].status == "skip"

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_failing_node_with_return_behavior_allows_downstream(self, mocker, backend_factory, backend_type):
        """Input -> FailingPython(RETURN) -> SuccessPython: downstream runs, checkpoint COMPLETED."""
        backend = backend_factory(backend_type)

        inp = Input(id="input", name="Input")
        failing = Python(
            id="failing-return",
            name="Failing Return",
            code='def run(input_data): raise ValueError("recoverable")',
            error_handling=ErrorHandling(behavior=Behavior.RETURN),
            depends=[NodeDependency(inp)],
        )
        success = Python(
            id="success",
            name="Success",
            code='def run(input_data): return {"ok": True}',
            depends=[NodeDependency(failing)],
        )

        flow = flows.Flow(
            nodes=[inp, failing, success],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})
        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.node_states["failing-return"].status == "failure"
        assert cp.node_states["success"].status == "success"

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_mid_pipeline_failure_checkpoint_preserves_completed_nodes(self, mocker, backend_factory, backend_type):
        """Input -> Python1(ok) -> Python2(fail,RAISE) -> Python3: checkpoint has Python1 completed."""
        backend = backend_factory(backend_type)

        inp = Input(id="input", name="Input")
        step1 = Python(
            id="step1",
            name="Step1",
            code='def run(input_data): return {"v": 1}',
            depends=[NodeDependency(inp)],
        )
        step2 = Python(
            id="step2",
            name="Step2",
            code='def run(input_data): raise RuntimeError("mid fail")',
            error_handling=ErrorHandling(behavior=Behavior.RAISE),
            depends=[NodeDependency(step1)],
        )
        step3 = Python(
            id="step3",
            name="Step3",
            code='def run(input_data): return {"v": 3}',
            depends=[NodeDependency(step2)],
        )

        flow = flows.Flow(
            nodes=[inp, step1, step2, step3],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert "input" in cp.completed_node_ids
        assert "step1" in cp.completed_node_ids
        assert cp.node_states["step1"].status == "success"
        assert cp.node_states["step2"].status == "failure"
        assert cp.node_states["step3"].status == "skip"

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_resume_after_mid_pipeline_failure_skips_completed(self, mocker, backend_factory, backend_type):
        """After mid-pipeline failure, resume skips completed nodes and re-runs failed ones."""
        backend = backend_factory(backend_type)

        def make_step2_code_fail():
            return 'def run(input_data): raise RuntimeError("fail")'

        def make_step2_code_succeed():
            return 'def run(input_data): return {"v": 2}'

        # Run 1: step2 fails
        inp = Input(id="input", name="Input")
        step1 = Python(
            id="step1", name="Step1", code='def run(input_data): return {"v": 1}', depends=[NodeDependency(inp)]
        )
        step2 = Python(
            id="step2",
            name="Step2",
            code=make_step2_code_fail(),
            error_handling=ErrorHandling(behavior=Behavior.RAISE),
            depends=[NodeDependency(step1)],
        )
        out = Output(id="output", name="Output", depends=[NodeDependency(step2)])

        flow1 = flows.Flow(
            nodes=[inp, step1, step2, out],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result1 = flow1.run_sync(input_data={"query": "test"})
        assert result1.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow1.id)
        assert cp.node_states["step1"].status == "success"

        # Patch checkpoint: remove step2 and output from completed so they re-run
        cp.completed_node_ids = ["input", "step1"]
        for nid in ["step2", "output"]:
            if nid in cp.node_states:
                del cp.node_states[nid]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        # Run 2: resume with fixed step2
        inp2 = Input(id="input", name="Input")
        step1_2 = Python(
            id="step1", name="Step1", code='def run(input_data): return {"v": 1}', depends=[NodeDependency(inp2)]
        )
        step2_2 = Python(
            id="step2",
            name="Step2",
            code=make_step2_code_succeed(),
            error_handling=ErrorHandling(behavior=Behavior.RAISE),
            depends=[NodeDependency(step1_2)],
        )
        out2 = Output(id="output", name="Output", depends=[NodeDependency(step2_2)])

        flow2 = flows.Flow(
            id=flow1.id,
            nodes=[inp2, step1_2, step2_2, out2],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result2 = flow2.run_sync(input_data=None, resume_from=cp.id)
        assert result2.status == RunnableStatus.SUCCESS


class TestAgentFailureMidLoop:
    """Agent LLM/tool errors during ReAct loop: checkpoint captures partial iteration state."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_llm_failure_after_tool_calls_captures_iteration(self, mocker, backend_factory, backend_type):
        """Agent does 2 tool calls, LLM fails on 3rd: checkpoint has iteration with 2 completed loops."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True, max_loops=10)

        call_count, CrashError = mock_react_loop_crash_at(mocker, crash_after_loop=2)
        result = flow.run_sync(input_data={"input": "Do calculations"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert "agent" in cp.node_states

        internal = cp.node_states["agent"].internal_state
        iteration = internal.get("iteration", {})
        assert iteration.get("completed_iterations", 0) >= 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_max_loops_exceeded_with_raise_behavior(self, mocker, backend_factory, backend_type):
        """Agent hits max_loops with RAISE behavior: checkpoint status is FAILED."""
        backend = backend_factory(backend_type)

        calc_tool = Python(
            id="calc-tool",
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        agent = Agent(
            id="agent",
            name="Agent",
            llm=make_agent_llm(),
            tools=[calc_tool],
            role="Test",
            max_loops=3,
        )
        flow = flows.Flow(
            nodes=[agent],
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop=True),
        )

        # LLM always returns tool calls, never final answer → hits max_loops
        mock_react_loop(mocker, tool_calls=100, final_answer="never reached")

        result = flow.run_sync(input_data={"input": "infinite task"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_max_loops_exceeded_return_behavior_captures_answer(self, mocker, backend_factory, backend_type):
        """Agent hits max_loops with RETURN behavior: checkpoint COMPLETED with crafted answer."""
        backend = backend_factory(backend_type)

        calc_tool = Python(
            id="calc-tool",
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        agent = Agent(
            id="agent",
            name="Agent",
            llm=make_agent_llm(),
            tools=[calc_tool],
            role="Test",
            max_loops=3,
            behaviour_on_max_loops=Behavior.RETURN,
        )
        flow = flows.Flow(
            nodes=[agent],
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop=True),
        )

        call_count = {"value": 0}

        def side_effect(stream: bool, *args, **kwargs):
            call_count["value"] += 1
            r = ModelResponse()
            if call_count["value"] <= 5:
                r["choices"][0]["message"]["content"] = (
                    f"Thought: Step {call_count['value']}.\n"
                    f"Action: calculator\n"
                    f'Action Input: {{"step": {call_count["value"]}}}'
                )
            else:
                r["choices"][0]["message"]["content"] = "Max loops summary answer"
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)

        result = flow.run_sync(input_data={"input": "long task"})
        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.node_states["agent"].status == "success"


class TestAgentTimeoutWithCheckpoint:
    """Agent times out during execution: checkpoint captures state up to timeout."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_timeout_creates_failed_checkpoint(self, mocker, backend_factory, backend_type):
        """Agent with timeout_seconds: if LLM hangs, agent fails and checkpoint is FAILED."""
        backend = backend_factory(backend_type)

        import time

        calc_tool = Python(
            id="calc-tool",
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        agent = Agent(
            id="agent",
            name="Agent",
            llm=make_agent_llm(),
            tools=[calc_tool],
            role="Test",
            max_loops=5,
            error_handling=ErrorHandling(timeout_seconds=0.5),
        )
        flow = flows.Flow(
            nodes=[agent],
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop=True),
        )

        def hanging_llm(stream: bool, *args, **kwargs):
            time.sleep(2)
            r = ModelResponse()
            r["choices"][0]["message"]["content"] = "Thought: Done.\nFinal Answer: too late"
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=hanging_llm)

        result = flow.run_sync(input_data={"input": "test"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_timeout_after_partial_loops_captures_iteration(self, mocker, backend_factory, backend_type):
        """Agent completes 2 loops, then hangs on 3rd: checkpoint has iteration data from loops 1-2."""
        backend = backend_factory(backend_type)

        import time

        calc_tool = Python(
            id="calc-tool",
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        agent = Agent(
            id="agent",
            name="Agent",
            llm=make_agent_llm(),
            tools=[calc_tool],
            role="Test",
            max_loops=10,
            error_handling=ErrorHandling(timeout_seconds=1.5),
        )
        flow = flows.Flow(
            nodes=[agent],
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop=True),
        )

        call_count = {"value": 0}

        def partial_then_hang(stream: bool, *args, **kwargs):
            call_count["value"] += 1
            r = ModelResponse()
            if call_count["value"] <= 2:
                r["choices"][0]["message"]["content"] = (
                    f"Thought: Step {call_count['value']}.\n"
                    f"Action: calculator\n"
                    f'Action Input: {{"step": {call_count["value"]}}}'
                )
                return r
            time.sleep(5)
            r["choices"][0]["message"]["content"] = "Thought: Done.\nFinal Answer: too late"
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=partial_then_hang)

        result = flow.run_sync(input_data={"input": "test"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        internal = cp.node_states["agent"].internal_state
        iteration = internal.get("iteration", {})
        assert iteration.get("completed_iterations", 0) >= 1


class TestSummarizerToolCheckpointInFlow:
    """SummarizerTool checkpoint state in a flow context."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_summarizer_in_flow_captures_llm_state(self, mocker, backend_factory, backend_type):
        backend = backend_factory(backend_type)

        def mock_completion(stream: bool, *args, **kwargs):
            r = ModelResponse()
            r["choices"][0]["message"]["content"] = "Summarized content"
            return r

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=mock_completion)

        summarizer_llm = llms.OpenAI(
            id="sum-llm",
            model=LLM_MODEL,
            connection=connections.OpenAI(api_key=TEST_API_KEY),
            is_postponed_component_init=True,
        )
        summarizer = SummarizerTool(id="summarizer", name="Summarizer", llm=summarizer_llm)

        flow = flows.Flow(
            nodes=[summarizer],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"input": "Some long text to summarize"})
        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert "summarizer" in cp.node_states
        internal = cp.node_states["summarizer"].internal_state
        assert "llm_state" in internal
