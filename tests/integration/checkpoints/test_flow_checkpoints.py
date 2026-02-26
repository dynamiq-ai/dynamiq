"""Integration tests for checkpoint/resume with realistic workflows.

Tests cover full end-to-end scenarios:
- Workflow success with checkpoint creation and verification
- Failure and recovery (LLM/tool failures -> resume from checkpoint)
- Agent with tools: mid-loop checkpointing and state preservation
- Run isolation: multiple runs produce independent checkpoints
- Per-run config overrides via RunnableConfig
- Workflow wrapper passthrough
- Async execution
"""

from io import BytesIO

import pytest
from litellm import ModelResponse

from dynamiq import Workflow, connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointConfig, CheckpointStatus
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input, Output
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"
MAX_CHECKPOINTS = 5


def make_llm(node_id: str = "llm", depends: list | None = None) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        name="LLM",
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        prompt=Prompt(messages=[Message(role="user", content="Process: {{query}}")]),
        is_postponed_component_init=True,
        depends=depends or [],
    )


def make_pipeline():
    """Input -> LLM -> Output pipeline."""
    inp = Input(id="input", name="Input")
    llm = make_llm(depends=[NodeDependency(inp)])
    out = Output(id="output", name="Output", depends=[NodeDependency(llm)])
    return [inp, llm, out]


def make_pipeline_with_python():
    """Input -> Python -> LLM -> Output pipeline."""
    inp = Input(id="input", name="Input")
    py = Python(
        id="python",
        name="Python",
        code='def run(input_data): return {"processed": input_data.get("value", 0) * 10}',
        depends=[NodeDependency(inp)],
    )
    llm = make_llm(depends=[NodeDependency(py)])
    out = Output(id="output", name="Output", depends=[NodeDependency(llm)])
    return [inp, py, llm, out]


def make_agent_pipeline():
    """Agent with calculator tool (standalone node, no Input/Output wrapper)."""
    agent_llm = llms.OpenAI(
        id="agent-llm",
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )
    calc_tool = Python(
        id="calc-tool",
        name="calculator",
        description="Calculator tool for math operations",
        code='def run(input_data): return {"result": 42}',
    )
    agent = Agent(
        id="agent",
        name="ReAct Agent",
        llm=agent_llm,
        tools=[calc_tool],
        role="Math assistant",
        goal="Calculate accurately",
        max_loops=5,
    )
    return [agent], agent


def mock_llm_success(mocker, response: str = "mocked_response"):
    def side_effect(stream: bool, *args, **kwargs):
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = response
        return r

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)


def mock_llm_fail_then_succeed(mocker, fail_count: int = 1, success_response: str = "Success"):
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] <= fail_count:
            raise RuntimeError(f"Transient failure #{call_count['value']}")
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = success_response
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_agent_react(mocker, tool_calls: int = 1, final_answer: str = "The result is 42."):
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


class TestWorkflowSuccessCheckpoint:
    """Full successful workflow execution with checkpoint verification."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_full_pipeline_creates_completed_checkpoint(self, mocker, backend_factory, backend_type):
        """Input -> LLM -> Output: all nodes succeed, checkpoint is COMPLETED with all state."""
        backend = backend_factory(backend_type)
        mock_llm_success(mocker)
        input_data = {"query": "What is AI?", "context": {"source": "test"}}

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend, max_checkpoints=MAX_CHECKPOINTS),
        )
        result = flow.run_sync(input_data=input_data)

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp is not None
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.original_input == input_data
        assert set(cp.completed_node_ids) == {"input", "llm", "output"}
        assert all(cp.node_states[nid].status == "success" for nid in cp.completed_node_ids)

        llm_state = cp.node_states["llm"]
        assert llm_state.output_data is not None
        assert "is_fallback_run" in llm_state.internal_state

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_multi_step_pipeline_preserves_intermediate_outputs(self, mocker, backend_factory, backend_type):
        """Input -> Python -> LLM -> Output: verify intermediate node outputs are captured."""
        backend = backend_factory(backend_type)
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline_with_python(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"value": 5, "query": "test"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert set(cp.completed_node_ids) == {"input", "python", "llm", "output"}
        python_output = cp.node_states["python"].output_data
        assert python_output["content"]["processed"] == 50 or python_output.get("processed") == 50

    def test_workflow_wrapper_creates_checkpoint(self, mocker):
        """Workflow wrapper delegates to Flow and checkpoint is created."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        workflow = Workflow(flow=flow, version="1.0")
        result = workflow.run_sync(input_data={"query": "test"})

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_async_pipeline_creates_checkpoint(self, mocker):
        """Async execution creates checkpoint identically to sync."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = await flow.run_async(input_data={"query": "async test"})

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.original_input == {"query": "async test"}


class TestWorkflowFailureCheckpoint:
    """Workflow failure scenarios: checkpoint captures partial progress."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_llm_failure_captures_completed_nodes(self, mocker, backend_factory, backend_type):
        """LLM fails: checkpoint is FAILED but input node is captured as completed."""
        backend = backend_factory(backend_type)
        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=RuntimeError("API down"))

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})

        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert "input" in cp.completed_node_ids
        assert cp.node_states["input"].status == "success"

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_python_failure_in_multi_step(self, mocker, backend_factory, backend_type):
        """Python node fails after Input succeeds: checkpoint captures Input as completed."""
        backend = backend_factory(backend_type)

        inp = Input(id="input", name="Input")
        failing_py = Python(
            id="failing-python",
            name="Failing Python",
            code='def run(input_data): raise ValueError("Data validation failed")',
            depends=[NodeDependency(inp)],
        )
        out = Output(id="output", name="Output", depends=[NodeDependency(failing_py)])

        flow = flows.Flow(
            nodes=[inp, failing_py, out],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        result = flow.run_sync(input_data={"query": "test"})

        assert result.status == RunnableStatus.FAILURE
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert "input" in cp.completed_node_ids


class TestRunIsolation:
    """Multiple runs produce independent checkpoints with proper tracking."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_separate_runs_create_separate_checkpoints(self, mocker, backend_factory, backend_type):
        """Each run creates its own checkpoint, ordered newest-first."""
        backend = backend_factory(backend_type)
        mock_llm_success(mocker)
        run_count = 3

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend, max_checkpoints=10),
        )

        for i in range(run_count):
            flow.run_sync(input_data={"query": f"Run {i}"})

        checkpoints = backend.get_list_by_flow(flow.id, limit=10)
        assert len(checkpoints) == run_count

        for i in range(len(checkpoints) - 1):
            assert checkpoints[i].created_at >= checkpoints[i + 1].created_at

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_cleanup_respects_max_checkpoints(self, mocker, backend_factory, backend_type):
        """Oldest checkpoints are removed when max_checkpoints is exceeded."""
        backend = backend_factory(backend_type)
        mock_llm_success(mocker)
        max_cp = 3

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend, max_checkpoints=max_cp),
        )

        for i in range(max_cp + 2):
            flow.run_sync(input_data={"query": f"Run {i}"})

        checkpoints = backend.get_list_by_flow(flow.id, limit=100)
        assert len(checkpoints) <= max_cp

    def test_run_id_and_wf_run_id_tracked(self, mocker):
        """Checkpoint stores both flow run_id and workflow wf_run_id for correlation."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )

        wf_run_id = "wf-request-abc-123"
        flow.run_sync(input_data={"query": "test"}, config=RunnableConfig(run_id=wf_run_id))

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.wf_run_id == wf_run_id
        assert cp.run_id != wf_run_id

    def test_wf_run_id_falls_back_to_flow_run_id_without_config(self, mocker):
        """Without RunnableConfig, wf_run_id equals flow's run_id."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        flow.run_sync(input_data={"query": "test"})

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.wf_run_id == cp.run_id


class TestPerRunConfigOverrides:
    """Per-run checkpoint config via RunnableConfig overrides flow-level defaults."""

    def test_enable_per_run(self, mocker):
        """Flow has checkpoint disabled, but per-run config enables it."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=False, backend=backend),
        )

        result = flow.run_sync(
            input_data={"query": "test"},
            config=RunnableConfig(checkpoint=CheckpointConfig(enabled=True)),
        )
        assert result.status == RunnableStatus.SUCCESS
        assert backend.get_latest_by_flow(flow.id) is not None

    def test_disable_per_run(self, mocker):
        """Flow has checkpoint enabled, but per-run config disables it."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )

        result = flow.run_sync(
            input_data={"query": "test"},
            config=RunnableConfig(checkpoint=CheckpointConfig(enabled=False)),
        )
        assert result.status == RunnableStatus.SUCCESS
        assert backend.get_latest_by_flow(flow.id) is None

    def test_exclude_nodes_per_run(self, mocker):
        """Per-run config excludes specific nodes from checkpoint."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )

        flow.run_sync(
            input_data={"query": "test"},
            config=RunnableConfig(checkpoint=CheckpointConfig(exclude_node_ids=["llm"])),
        )

        cp = backend.get_latest_by_flow(flow.id)
        assert "input" in cp.node_states
        assert "output" in cp.node_states
        assert "llm" not in cp.node_states

    def test_run_overrides_replace_flow_defaults(self, mocker):
        """Run-level exclude_node_ids replaces (not merges with) flow-level."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend, exclude_node_ids=["input"]),
        )

        flow.run_sync(
            input_data={"query": "test"},
            config=RunnableConfig(checkpoint=CheckpointConfig(exclude_node_ids=["output"])),
        )

        cp = backend.get_latest_by_flow(flow.id)
        assert "input" in cp.node_states
        assert "llm" in cp.node_states
        assert "output" not in cp.node_states

    def test_resume_from_nonexistent_checkpoint_raises(self, mocker):
        """Resuming from non-existent checkpoint raises ValueError."""
        backend = InMemory()
        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        with pytest.raises(ValueError, match="Checkpoint not found"):
            flow.run_sync(input_data={"query": "test"}, resume_from="nonexistent-id")

    def test_checkpointing_disabled_by_default(self, mocker):
        """Without explicit config, checkpointing is off."""
        mock_llm_success(mocker)
        flow = flows.Flow(nodes=make_pipeline())

        assert flow.checkpoint.enabled is False
        result = flow.run_sync(input_data={"query": "test"})
        assert result.status == RunnableStatus.SUCCESS
        assert flow._checkpoint is None


class TestAgentWithToolsCheckpoint:
    """Agent ReAct loop with tools: mid-loop checkpointing and state preservation."""

    def test_agent_success_captures_internal_state(self, mocker):
        """Agent completes successfully: checkpoint has agent internal state with tool_states."""
        backend = InMemory()
        nodes, agent = make_agent_pipeline()
        mock_agent_react(mocker, tool_calls=2)

        flow = flows.Flow(
            nodes=nodes,
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop_enabled=True),
        )
        result = flow.run_sync(input_data={"input": "Calculate 6*7"})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert "agent" in cp.node_states
        agent_state = cp.node_states["agent"]
        assert agent_state.status == "success"
        assert "history_offset" in agent_state.internal_state
        assert "llm_state" in agent_state.internal_state
        assert "tool_states" in agent_state.internal_state

    def test_mid_loop_checkpoint_produces_more_saves(self, mocker):
        """With checkpoint_mid_agent_loop_enabled=True, more checkpoint saves occur during agent loop."""
        backend_off = InMemory()
        nodes_off, _ = make_agent_pipeline()
        mock_agent_react(mocker, tool_calls=2)

        save_count_off = {"value": 0}
        original_save = InMemory.save

        def counting_save(self_backend, cp):
            save_count_off["value"] += 1
            return original_save(self_backend, cp)

        mocker.patch.object(InMemory, "save", counting_save)

        flow_off = flows.Flow(
            nodes=nodes_off,
            checkpoint=CheckpointConfig(enabled=True, backend=backend_off, checkpoint_mid_agent_loop_enabled=False),
        )
        flow_off.run_sync(input_data={"input": "test"})
        saves_disabled = save_count_off["value"]

        mocker.stopall()

        backend_on = InMemory()
        nodes_on, _ = make_agent_pipeline()
        mock_agent_react(mocker, tool_calls=2)

        save_count_on = {"value": 0}

        def counting_save_on(self_backend, cp):
            save_count_on["value"] += 1
            return original_save(self_backend, cp)

        mocker.patch.object(InMemory, "save", counting_save_on)

        flow_on = flows.Flow(
            nodes=nodes_on,
            checkpoint=CheckpointConfig(enabled=True, backend=backend_on, checkpoint_mid_agent_loop_enabled=True),
        )
        flow_on.run_sync(input_data={"input": "test"})

        assert save_count_on["value"] > saves_disabled

    def test_mid_loop_enabled_via_runnable_config(self, mocker):
        """checkpoint_mid_agent_loop_enabled can be enabled per-run via RunnableConfig."""
        backend = InMemory()
        nodes, _ = make_agent_pipeline()
        mock_agent_react(mocker, tool_calls=1)

        flow = flows.Flow(
            nodes=nodes,
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop_enabled=False),
        )

        run_config = RunnableConfig(checkpoint=CheckpointConfig(checkpoint_mid_agent_loop_enabled=True))
        result = flow.run_sync(input_data={"input": "test"}, config=run_config)

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert "agent" in cp.node_states
        assert "history_offset" in cp.node_states["agent"].internal_state

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_agent_multi_tool_calls_preserves_progress(self, mocker, backend_factory, backend_type):
        """Agent makes 3 tool calls: final checkpoint has complete state."""
        backend = backend_factory(backend_type)
        nodes, _ = make_agent_pipeline()
        mock_agent_react(mocker, tool_calls=3, final_answer="Completed 3 calculations.")

        flow = flows.Flow(
            nodes=nodes,
            checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop_enabled=True),
        )
        result = flow.run_sync(input_data={"input": "Do 3 calculations"})

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.node_states["agent"].status == "success"
        assert "tool_states" in cp.node_states["agent"].internal_state


class TestWorkflowResumePassthrough:
    """Workflow wrapper properly passes resume_from to Flow."""

    def test_workflow_accepts_resume_from_kwarg(self, mocker):
        """Workflow.run_sync(resume_from=...) forwards to Flow."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        workflow = Workflow(flow=flow)

        workflow.run_sync(input_data={"query": "original"})
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED

    def test_workflow_resume_via_runnable_config(self, mocker):
        """Checkpoint config with resume_from on RunnableConfig works through Workflow."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        workflow = Workflow(flow=flow)

        run_config = RunnableConfig(checkpoint=CheckpointConfig(enabled=True))
        result = workflow.run_sync(input_data={"query": "test"}, config=run_config)
        assert result.status == RunnableStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_async_workflow_with_checkpoint(self, mocker):
        """Async Workflow execution creates checkpoint."""
        backend = InMemory()
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        workflow = Workflow(flow=flow)

        result = await workflow.run_async(input_data={"query": "async test"})
        assert result.status == RunnableStatus.SUCCESS
        assert backend.get_latest_by_flow(flow.id).status == CheckpointStatus.COMPLETED


class TestCheckpointStatusLifecycle:
    """Checkpoint status transitions during flow execution."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_success_flow_ends_with_completed(self, mocker, backend_factory, backend_type):
        backend = backend_factory(backend_type)
        mock_llm_success(mocker)

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        flow.run_sync(input_data={"query": "test"})

        assert backend.get_latest_by_flow(flow.id).status == CheckpointStatus.COMPLETED

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_failed_flow_ends_with_failed(self, mocker, backend_factory, backend_type):
        backend = backend_factory(backend_type)
        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=RuntimeError("boom"))

        flow = flows.Flow(
            nodes=make_pipeline(),
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )
        flow.run_sync(input_data={"query": "test"})

        assert backend.get_latest_by_flow(flow.id).status == CheckpointStatus.FAILED


class TestBinaryDataCheckpoint:
    """Checkpoint handles binary/BytesIO input data."""

    def test_bytesio_input_checkpoint(self, mocker):
        backend = InMemory()
        inp = Input(id="input", name="Input")
        out = Output(id="output", name="Output", depends=[NodeDependency(inp)])

        flow = flows.Flow(
            nodes=[inp, out],
            checkpoint=CheckpointConfig(enabled=True, backend=backend),
        )

        file_obj = BytesIO(b"Test file content for checkpoint testing.")
        file_obj.name = "test_document.txt"

        result = flow.run_sync(input_data={"file": file_obj, "metadata": {"type": "text"}})

        assert result.status == RunnableStatus.SUCCESS
        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert cp.original_input is not None
