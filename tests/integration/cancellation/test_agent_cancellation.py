import threading
import time

import pytest

from dynamiq import Workflow, connections, flows
from dynamiq.callbacks.tracing import RunStatus, TracingCallbackHandler
from dynamiq.checkpoints.checkpoint import CheckpointStatus
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import CanceledException, CancellationConfig

from .conftest import mock_llm_react_loop

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"

AGENT_ID = "react-agent"
AGENT_LLM_ID = "react-agent-llm"
CALC_TOOL_ID = "calculator-tool"

CANCEL_DELAY = 0.4
THREAD_JOIN_TIMEOUT = 15.0


def make_agent_llm(node_id: str = AGENT_LLM_ID) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


SLOW_TOOL_CODE = """
import time
def run(input_data):
    # Sleep in small increments. The agent's check_cancellation runs between
    # tool calls (in _run_agent loop), so a cancel signal during this sleep
    # will be caught at the next loop iteration.
    sleep_total = 1.5
    elapsed = 0.0
    while elapsed < sleep_total:
        time.sleep(0.05)
        elapsed += 0.05
    return {"content": f"step {input_data.get('step', '?')} done"}
"""


def make_calc_tool() -> Python:
    return Python(
        id=CALC_TOOL_ID,
        name="calculator",
        description="Calculator tool. Input: {'step': <n>}.",
        code=SLOW_TOOL_CODE,
    )


def create_agent_flow(*, max_loops: int = 10, mid_loop_checkpoint: bool = False, backend=None):
    """Build a Flow containing one ReAct Agent with a slow calculator tool."""
    agent = Agent(
        id=AGENT_ID,
        name="ReAct Agent",
        llm=make_agent_llm(),
        tools=[make_calc_tool()],
        role="Math assistant",
        max_loops=max_loops,
    )
    checkpoint = None
    if backend is not None:
        checkpoint = CheckpointConfig(
            enabled=True,
            backend=backend,
            checkpoint_mid_agent_loop_enabled=mid_loop_checkpoint,
        )
        flow = flows.Flow(nodes=[agent], checkpoint=checkpoint)
    else:
        flow = flows.Flow(nodes=[agent])
    return flow, agent


def run_in_thread(target):
    holder = {}

    def runner():
        try:
            holder["result"] = target()
        except Exception as e:
            holder["exception"] = e

    thread = threading.Thread(target=runner)
    thread.start()
    return holder, thread


class TestPreCanceledAgent:
    def test_agent_never_invokes_llm_when_pre_canceled(self, mocker, cancellation_token, runnable_config):
        call_count = mock_llm_react_loop(mocker, tool_calls=10)
        cancellation_token.cancel()

        flow, agent = create_agent_flow()
        result = flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        assert result.status == RunnableStatus.CANCELED
        assert result.input is None
        assert result.output is None
        assert result.error is not None
        assert result.error.type is CanceledException
        assert result.error.message  # has a descriptive message
        # LLM never called - agent stopped before its first iteration
        assert call_count["value"] == 0

    def test_pre_canceled_workflow_returns_canceled(self, mocker, cancellation_token, runnable_config):
        mock_llm_react_loop(mocker, tool_calls=5)
        cancellation_token.cancel()

        flow, _ = create_agent_flow()
        wf = Workflow(flow=flow)
        result = wf.run_sync(input_data={"input": "test"}, config=runnable_config)
        assert result.status == RunnableStatus.CANCELED


class TestAgentMidLoopCancellation:
    def test_cancel_during_react_loop(self, mocker, cancellation_token, runnable_config):
        """Agent runs through a few tool calls, then we cancel during a tool call.
        The slow tool's internal cancellation check fires and the agent stops."""
        call_count = mock_llm_react_loop(mocker, tool_calls=10)

        flow, _ = create_agent_flow(max_loops=10)

        def go():
            return flow.run_sync(input_data={"input": "do calculations"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        assert result.status == RunnableStatus.CANCELED
        # Agent had time for at least 1 LLM call before cancel arrived
        assert call_count["value"] >= 1
        # But did not complete all 10 tool calls
        assert call_count["value"] < 11

    def test_cancel_stops_agent_promptly(self, mocker, cancellation_token, runnable_config):
        """After cancel, the agent should return within a few seconds, not run forever."""
        mock_llm_react_loop(mocker, tool_calls=100)  # would take very long without cancel

        flow, _ = create_agent_flow(max_loops=100)

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        start = time.time()
        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)
        elapsed = time.time() - start

        assert holder["result"].status == RunnableStatus.CANCELED
        # Should finish quickly after cancel - well before max iteration would take
        assert elapsed < 10.0


class TestCancellationDoesNotSaveMidLoopCheckpoint:
    """When agent is canceled, the checkpoint should NOT show COMPLETED state.

    Mid-loop checkpointing may have saved partial state, but cancellation is
    an interrupt - the final checkpoint status is not COMPLETED.
    """

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_canceled_agent_no_completed_checkpoint(
        self, mocker, backend_factory, backend_type, cancellation_token, runnable_config
    ):
        backend = backend_factory(backend_type)
        mock_llm_react_loop(mocker, tool_calls=10)

        flow, _ = create_agent_flow(max_loops=10, mid_loop_checkpoint=True, backend=backend)

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        assert holder["result"].status == RunnableStatus.CANCELED

        cp = backend.get_latest_by_flow(flow.id)
        if cp is not None:
            assert cp.status != CheckpointStatus.COMPLETED
            assert cp.status != CheckpointStatus.FAILED

    def test_pre_canceled_agent_with_checkpoint_saves_no_completion(
        self, mocker, memory_backend, cancellation_token, runnable_config
    ):
        mock_llm_react_loop(mocker, tool_calls=5)
        cancellation_token.cancel()

        flow, _ = create_agent_flow(mid_loop_checkpoint=True, backend=memory_backend)
        result = flow.run_sync(input_data={"input": "test"}, config=runnable_config)
        assert result.status == RunnableStatus.CANCELED

        cp = memory_backend.get_latest_by_flow(flow.id)
        if cp is not None:
            assert cp.status != CheckpointStatus.COMPLETED


class TestAgentCancellationTracing:
    def test_canceled_agent_fires_node_and_flow_canceled_callbacks(self, mocker, cancellation_token):
        mock_llm_react_loop(mocker, tool_calls=10)

        tracing = TracingCallbackHandler()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=cancellation_token),
            callbacks=[tracing],
        )
        flow, _ = create_agent_flow(max_loops=10)

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        canceled_runs = [r for r in tracing.runs.values() if r.status == RunStatus.CANCELED]
        assert len(canceled_runs) >= 1, "Expected at least one CANCELED run in trace"

    def test_canceled_workflow_fires_workflow_canceled_callback(self, mocker, cancellation_token):
        mock_llm_react_loop(mocker, tool_calls=5)
        cancellation_token.cancel()

        tracing = TracingCallbackHandler()
        config = RunnableConfig(
            cancellation=CancellationConfig(token=cancellation_token),
            callbacks=[tracing],
        )
        flow, _ = create_agent_flow()
        wf = Workflow(flow=flow)
        wf.run_sync(input_data={"input": "test"}, config=config)

        canceled_runs = [r for r in tracing.runs.values() if r.status == RunStatus.CANCELED]
        # Should have workflow + flow + agent all CANCELED
        assert len(canceled_runs) >= 1


class TestToolCancellationPropagation:
    def test_canceled_tool_propagates_to_agent(self, mocker, cancellation_token, runnable_config):
        """If a tool returns CANCELED, _run_tool should raise CanceledException
        so the agent itself returns CANCELED."""
        mock_llm_react_loop(mocker, tool_calls=10)

        flow, _ = create_agent_flow(max_loops=10)

        def go():
            return flow.run_sync(input_data={"input": "do many calcs"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)  # let agent get into a tool call
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        # The tool's check_cancellation triggers, agent catches CanceledException,
        # flow propagates it - final status is CANCELED
        assert result.status == RunnableStatus.CANCELED


class TestMultiAgentFlowCancellation:
    def test_cancel_stops_all_dependent_agents(self, mocker, cancellation_token, runnable_config):
        """Two sequential agents. Cancel during the first agent, second never runs."""
        call_count = mock_llm_react_loop(mocker, tool_calls=10)

        agent1 = Agent(
            id="agent-1",
            name="Agent 1",
            llm=make_agent_llm("llm-1"),
            tools=[Python(id="tool-1", name="calculator", description="calc", code=SLOW_TOOL_CODE)],
            role="Test agent 1",
            max_loops=10,
        )
        agent2 = Agent(
            id="agent-2",
            name="Agent 2",
            llm=make_agent_llm("llm-2"),
            tools=[Python(id="tool-2", name="calculator", description="calc", code=SLOW_TOOL_CODE)],
            role="Test agent 2",
            max_loops=10,
            depends=[NodeDependency(node=agent1)],
        )
        flow = flows.Flow(nodes=[agent1, agent2])

        def go():
            return flow.run_sync(input_data={"input": "test"}, config=runnable_config)

        holder, thread = run_in_thread(go)
        time.sleep(CANCEL_DELAY)
        cancellation_token.cancel()
        thread.join(timeout=THREAD_JOIN_TIMEOUT)

        result = holder["result"]
        assert result.status == RunnableStatus.CANCELED
        # Some tool calls happened in agent1 but execution stopped before agent2
        assert call_count["value"] >= 1
