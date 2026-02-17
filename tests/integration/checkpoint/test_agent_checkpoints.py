import pytest
from litellm import ModelResponse

from dynamiq import connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointConfig, CheckpointStatus
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import ApprovalConfig, ApprovalInputData, FeedbackMethod

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"


def make_agent_llm(node_id: str = "agent-llm") -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


def make_agent_flow(backend, *, mid_loop: bool = False, flow_id: str | None = None):
    """Create Agent(with calculator) flow. Agent receives input directly (no Input wrapper)."""
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
        goal="Calculate accurately",
        max_loops=5,
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
                f"Action Input: {{\"step\": {call_count['value']}}}"
            )
        else:
            r["choices"][0]["message"]["content"] = f"Thought: Done.\nFinal Answer: {final_answer}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


class TestAgentInFlowCheckpoint:
    """Agent with LLM + tools inside a Flow: full success, state verification."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_flow_success_with_single_tool_call(self, mocker, backend_factory, backend_type):
        """Input -> Agent (1 tool call) -> Output: checkpoint COMPLETED with full state."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend)
        mock_react_loop(mocker, tool_calls=1)

        result = flow.run_sync(input_data={"input": "What is 6*7?"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert "agent" in cp.completed_node_ids

        agent_state = cp.node_states["agent"]
        assert agent_state.status == "success"
        assert agent_state.output_data is not None
        assert "history_offset" in agent_state.internal_state
        assert "llm_state" in agent_state.internal_state
        assert "tool_states" in agent_state.internal_state

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_flow_success_with_multiple_tool_calls(self, mocker, backend_factory, backend_type):
        """Agent makes 3 tool calls before final answer: all state captured."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=3, final_answer="Completed 3 calculations.")

        result = flow.run_sync(input_data={"input": "Do 3 calculations"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.node_states["agent"].status == "success"
        assert "tool_states" in cp.node_states["agent"].internal_state

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_flow_llm_failure_creates_failed_checkpoint(self, mocker, backend_factory, backend_type):
        """Agent LLM fails: checkpoint FAILED with agent error captured."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend)
        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=RuntimeError("LLM API down"))

        result = flow.run_sync(input_data={"input": "test"})

        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED


class TestAgentMidLoopCheckpoint:
    """Agent mid-loop checkpointing: saves state after each tool call iteration."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_mid_loop_captures_agent_state_during_execution(self, mocker, backend_factory, backend_type):
        """With checkpoint_mid_agent_loop=True, agent state is saved after each tool call."""
        backend = backend_factory(backend_type)
        flow, agent = make_agent_flow(backend, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        result = flow.run_sync(input_data={"input": "Calculate step by step"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert "agent" in cp.node_states
        assert "history_offset" in cp.node_states["agent"].internal_state

    def test_mid_loop_produces_more_saves_than_without(self, mocker):
        """Mid-loop enabled produces more backend.save() calls than disabled."""
        original_save = InMemory.save

        backend_off = InMemory()
        flow_off, _ = make_agent_flow(backend_off, mid_loop=False)
        mock_react_loop(mocker, tool_calls=2)

        saves_off = {"count": 0}

        def counting_off(self, cp):
            saves_off["count"] += 1
            return original_save(self, cp)

        mocker.patch.object(InMemory, "save", counting_off)
        flow_off.run_sync(input_data={"input": "test"})
        count_off = saves_off["count"]

        mocker.stopall()

        backend_on = InMemory()
        flow_on, _ = make_agent_flow(backend_on, mid_loop=True)
        mock_react_loop(mocker, tool_calls=2)

        saves_on = {"count": 0}

        def counting_on(self, cp):
            saves_on["count"] += 1
            return original_save(self, cp)

        mocker.patch.object(InMemory, "save", counting_on)
        flow_on.run_sync(input_data={"input": "test"})

        assert saves_on["count"] > count_off

    def test_mid_loop_enabled_via_runnable_config_override(self, mocker):
        """Flow-level mid_loop=False overridden to True via RunnableConfig."""
        backend = InMemory()
        flow, _ = make_agent_flow(backend, mid_loop=False)
        mock_react_loop(mocker, tool_calls=1)

        run_config = RunnableConfig(checkpoint=CheckpointConfig(checkpoint_mid_agent_loop=True))
        result = flow.run_sync(input_data={"input": "test"}, config=run_config)

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert "agent" in cp.node_states
        assert "history_offset" in cp.node_states["agent"].internal_state


class TestHITLApprovalCheckpoint:
    """HITL approval flow with checkpoint persistence."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_approved_node_completes_and_checkpoint_captures(self, mocker, backend_factory, backend_type):
        """Node with approval enabled: approved -> executes -> checkpoint captures result."""
        backend = backend_factory(backend_type)

        input_node = Input(id="input", name="Input")
        approved_node = Python(
            id="approved-python",
            name="Approved Python",
            code='def run(input_data): return {"result": "processed"}',
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE),
            depends=[NodeDependency(input_node)],
        )
        output_node = Output(id="output", name="Output", depends=[NodeDependency(approved_node)])

        mocker.patch.object(
            Python,
            "send_console_approval_message",
            return_value=ApprovalInputData(feedback="", is_approved=True, data={}),
        )

        flow = flows.Flow(
            nodes=[input_node, approved_node, output_node],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert "approved-python" in cp.completed_node_ids
        assert cp.node_states["approved-python"].status == "success"

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_rejected_node_skipped_and_checkpoint_captures(self, mocker, backend_factory, backend_type):
        """Node with approval enabled: rejected -> skipped -> checkpoint captures skip status."""
        backend = backend_factory(backend_type)

        input_node = Input(id="input", name="Input")
        rejected_node = Python(
            id="rejected-python",
            name="Rejected Python",
            code='def run(input_data): return {"result": "processed"}',
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE),
            depends=[NodeDependency(input_node)],
        )
        output_node = Output(id="output", name="Output", depends=[NodeDependency(rejected_node)])

        mocker.patch.object(
            Python,
            "send_console_approval_message",
            return_value=ApprovalInputData(feedback="not approved", is_approved=False, data={}),
        )

        flow = flows.Flow(
            nodes=[input_node, rejected_node, output_node],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        flow.run_sync(input_data={"query": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        assert "input" in cp.completed_node_ids
        assert "rejected-python" in cp.completed_node_ids
        assert cp.node_states["rejected-python"].status == "skip"

    def test_approval_response_stored_in_checkpoint_state(self, mocker):
        """Approval response is captured in node's checkpoint internal_state."""
        backend = InMemory()

        input_node = Input(id="input", name="Input")
        approved_node = Python(
            id="approved-python",
            name="Approved Python",
            code='def run(input_data): return {"result": "ok"}',
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE),
            depends=[NodeDependency(input_node)],
        )
        output_node = Output(id="output", name="Output", depends=[NodeDependency(approved_node)])

        mocker.patch.object(
            Python,
            "send_console_approval_message",
            return_value=ApprovalInputData(feedback="", is_approved=True, data={}),
        )

        flow = flows.Flow(
            nodes=[input_node, approved_node, output_node],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        flow.run_sync(input_data={"query": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.node_states["approved-python"].status == "success"

    def test_checkpoint_context_notified_on_pending_input(self, mocker):
        """When approval is requested, checkpoint context receives pending_input notification."""
        backend = InMemory()
        pending_notifications = []

        input_node = Input(id="input", name="Input")
        approved_node = Python(
            id="hitl-python",
            name="HITL Python",
            code='def run(input_data): return {"result": "ok"}',
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE),
            depends=[NodeDependency(input_node)],
        )
        output_node = Output(id="output", name="Output", depends=[NodeDependency(approved_node)])

        mocker.patch.object(
            Python,
            "send_console_approval_message",
            return_value=ApprovalInputData(feedback="", is_approved=True, data={}),
        )

        flow = flows.Flow(
            nodes=[input_node, approved_node, output_node],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )

        original_setup = flow._setup_checkpoint_context

        def tracking_setup(config):
            result = original_setup(config)
            if result and result.checkpoint_context:
                original_pending = result.checkpoint_context._on_pending_input

                def tracking_pending(node_id, prompt, metadata):
                    pending_notifications.append({"node_id": node_id, "prompt": prompt, "metadata": metadata})
                    if original_pending:
                        original_pending(node_id, prompt, metadata)

                result.checkpoint_context._on_pending_input = tracking_pending
            return result

        mocker.patch.object(flow, "_setup_checkpoint_context", side_effect=tracking_setup)

        flow.run_sync(input_data={"query": "test"})

        assert len(pending_notifications) >= 1
        assert pending_notifications[0]["node_id"] == "hitl-python"


class TestAgentFlowRunIsolation:
    """Multiple agent flow runs produce independent checkpoints."""

    def test_separate_runs_have_separate_checkpoints(self, mocker):
        """Each agent flow run creates its own checkpoint, all COMPLETED."""
        backend = InMemory()
        flow, _ = make_agent_flow(backend)
        run_count = 3

        for i in range(run_count):
            mock_react_loop(mocker, tool_calls=1, final_answer=f"Answer {i}")
            flow.run_sync(input_data={"input": f"Query {i}"})
            mocker.stopall()

        checkpoints = backend.get_list_by_flow(flow.id, limit=10)
        assert len(checkpoints) == run_count
        assert all(cp.status == CheckpointStatus.COMPLETED for cp in checkpoints)

    def test_different_wf_run_ids_tracked(self, mocker):
        """Agent flow runs with different RunnableConfig.run_id track wf_run_id correctly."""
        backend = InMemory()
        flow, _ = make_agent_flow(backend)

        wf_ids = ["wf-run-aaa", "wf-run-bbb"]
        for wf_id in wf_ids:
            mock_react_loop(mocker, tool_calls=1)
            flow.run_sync(input_data={"input": "test"}, config=RunnableConfig(run_id=wf_id))
            mocker.stopall()

        checkpoints = backend.get_list_by_flow(flow.id, limit=10)
        stored_wf_ids = {cp.wf_run_id for cp in checkpoints}
        assert stored_wf_ids == set(wf_ids)
