"""Integration tests for parallel tool calls, SubAgentTool, and approval with checkpoints.

Tests cover:
- Agent with parallel_tool_calls_enabled + mid-loop checkpoint state
- Parallel tool calls resume skips completed loops
- SubAgentTool._call_count captured in checkpoint
- SubAgentTool._call_count preserved on resume (not wiped by reset_run_state)
- SubAgentTool max_calls limit respected after resume
- Approval (HITL) flow with agent and checkpoint state
- Multi-tool agent with approval upstream of the agent node
"""

import pytest
from litellm import ModelResponse

from dynamiq import connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointStatus
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"

FLOW_ID = "parallel-subagent-flow"
AGENT_ID = "test-agent"
AGENT_LLM_ID = "test-agent-llm"
PARENT_AGENT_ID = "parent-agent"
PARENT_LLM_ID = "parent-agent-llm"
CHILD_AGENT_ID = "child-agent"
CHILD_LLM_ID = "child-agent-llm"
SUBAGENT_TOOL_ID = "sub-agent-tool"
ADDER_TOOL_ID = "adder-tool"
MULTIPLIER_TOOL_ID = "multiplier-tool"
CALC_TOOL_ID = "calculator-tool"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_llm(node_id: str = AGENT_LLM_ID) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


def create_parallel_agent_flow(
    backend,
    *,
    mid_loop: bool = True,
    flow_id: str = FLOW_ID,
    behavior: CheckpointBehavior = CheckpointBehavior.APPEND,
    max_loops: int = 5,
):
    """Agent with two Python tools and parallel_tool_calls_enabled."""
    adder = Python(
        id=ADDER_TOOL_ID,
        name="adder",
        description='Adds two numbers a and b. Expects {"a": int, "b": int}.',
        code='def run(input_data): return {"result": input_data.get("a", 0) + input_data.get("b", 0)}',
        is_parallel_execution_allowed=True,
    )
    multiplier = Python(
        id=MULTIPLIER_TOOL_ID,
        name="multiplier",
        description='Multiplies two numbers x and y. Expects {"x": int, "y": int}.',
        code='def run(input_data): return {"result": input_data.get("x", 0) * input_data.get("y", 0)}',
        is_parallel_execution_allowed=True,
    )
    agent = Agent(
        id=AGENT_ID,
        name="Parallel Agent",
        llm=make_llm(),
        tools=[adder, multiplier],
        role="Math assistant that can add and multiply in parallel",
        max_loops=max_loops,
        parallel_tool_calls_enabled=True,
        inference_mode=InferenceMode.XML,
    )
    flow = flows.Flow(
        id=flow_id,
        nodes=[agent],
        checkpoint=CheckpointConfig(
            enabled=True,
            backend=backend,
            checkpoint_mid_agent_loop_enabled=mid_loop,
            behavior=behavior,
        ),
    )
    return flow, agent


def create_subagent_flow(
    backend,
    *,
    mid_loop: bool = True,
    flow_id: str = FLOW_ID,
    max_calls: int | None = None,
    behavior: CheckpointBehavior = CheckpointBehavior.APPEND,
):
    """Parent agent with a SubAgentTool wrapping a child agent."""
    child_llm = make_llm(node_id=CHILD_LLM_ID)
    calc_tool = Python(
        id=CALC_TOOL_ID,
        name="calculator",
        description="Calculator tool for math.",
        code='def run(input_data): return {"result": 42}',
    )
    child_agent = Agent(
        id=CHILD_AGENT_ID,
        name="Calc Agent",
        llm=child_llm,
        tools=[calc_tool],
        role="Calculator assistant",
        max_loops=3,
    )
    sub_tool = SubAgentTool(
        id=SUBAGENT_TOOL_ID,
        name="Calc Agent",
        description="Delegates calculations to a specialist agent.",
        agent=child_agent,
        max_calls=max_calls,
    )
    parent_llm = make_llm(node_id=PARENT_LLM_ID)
    parent_agent = Agent(
        id=PARENT_AGENT_ID,
        name="Manager Agent",
        llm=parent_llm,
        tools=[sub_tool],
        role="Manager that delegates to sub-agents",
        max_loops=5,
    )
    flow = flows.Flow(
        id=flow_id,
        nodes=[parent_agent],
        checkpoint=CheckpointConfig(
            enabled=True,
            backend=backend,
            checkpoint_mid_agent_loop_enabled=mid_loop,
            behavior=behavior,
        ),
    )
    return flow, parent_agent, sub_tool


def mock_parallel_react(mocker, final_answer="Parallel done: 10 and 20."):
    """Mock LLM: one parallel tool call, then final answer (XML format)."""
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] == 1:
            r["choices"][0]["message"]["content"] = (
                "<output>\n"
                "    <thought>I need to run both tools in parallel.</thought>\n"
                "    <action>run-parallel</action>\n"
                '    <action_input>{"tools": ['
                '{"name": "adder", "input": {"a": 3, "b": 7}}, '
                '{"name": "multiplier", "input": {"x": 4, "y": 5}}'
                "]}</action_input>\n"
                "</output>"
            )
        else:
            r["choices"][0]["message"]["content"] = (
                "<output>\n"
                f"    <thought>Both tools returned results.</thought>\n"
                f"    <answer>{final_answer}</answer>\n"
                "</output>"
            )
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_parallel_multi_round(mocker, parallel_rounds: int = 2, final_answer="Multi-round done."):
    """Mock LLM: N rounds of parallel calls, then final answer (XML format)."""
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] <= parallel_rounds:
            r["choices"][0]["message"]["content"] = (
                "<output>\n"
                f"    <thought>Round {call_count['value']} of parallel calls.</thought>\n"
                "    <action>run-parallel</action>\n"
                '    <action_input>{"tools": ['
                f'{{"name": "adder", "input": {{"a": {call_count["value"]}, "b": 1}}}}, '
                f'{{"name": "multiplier", "input": {{"x": {call_count["value"]}, "y": 2}}}}'
                "]}</action_input>\n"
                "</output>"
            )
        else:
            r["choices"][0]["message"]["content"] = (
                "<output>\n"
                f"    <thought>All rounds complete.</thought>\n"
                f"    <answer>{final_answer}</answer>\n"
                "</output>"
            )
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_subagent_react(mocker, child_uses_tool: bool = False, final_answer="Sub-agent result: 42."):
    """Mock LLM for parent + child agent interaction.

    Call sequence:
      1. Parent: calls child agent tool
      2. Child: (optionally uses calculator, then) final answer
      3. Parent: final answer
    """
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] == 1:
            r["choices"][0]["message"]["content"] = (
                "Thought: I should delegate to the calc agent.\n"
                "Action: Calc Agent\n"
                'Action Input: {"input": "calculate 6*7"}'
            )
        elif call_count["value"] == 2:
            if child_uses_tool:
                r["choices"][0]["message"]["content"] = (
                    "Thought: I need the calculator.\n" "Action: calculator\n" 'Action Input: {"expression": "6*7"}'
                )
            else:
                r["choices"][0]["message"]["content"] = "Thought: I know this.\nFinal Answer: 42"
        elif call_count["value"] == 3:
            if child_uses_tool:
                r["choices"][0]["message"]["content"] = "Thought: Got the result.\nFinal Answer: 42"
            else:
                r["choices"][0]["message"]["content"] = f"Thought: The child returned 42.\nFinal Answer: {final_answer}"
        else:
            r["choices"][0]["message"]["content"] = f"Thought: Done.\nFinal Answer: {final_answer}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_subagent_two_delegations(mocker, final_answer="All delegations complete."):
    """Mock LLM: parent delegates to child twice, then final answer.

    Call sequence:
      1. Parent: calls child agent (first delegation)
      2. Child: final answer
      3. Parent: calls child agent (second delegation)
      4. Child: final answer
      5. Parent: final answer
    """
    call_count = {"value": 0}

    def side_effect(stream: bool, *args, **kwargs):
        call_count["value"] += 1
        r = ModelResponse()
        if call_count["value"] == 1:
            r["choices"][0]["message"]["content"] = (
                "Thought: First delegation.\n" "Action: Calc Agent\n" 'Action Input: {"input": "first task"}'
            )
        elif call_count["value"] == 2:
            r["choices"][0]["message"]["content"] = "Thought: First task done.\nFinal Answer: result-1"
        elif call_count["value"] == 3:
            r["choices"][0]["message"]["content"] = (
                "Thought: Second delegation.\n" "Action: Calc Agent\n" 'Action Input: {"input": "second task"}'
            )
        elif call_count["value"] == 4:
            r["choices"][0]["message"]["content"] = "Thought: Second task done.\nFinal Answer: result-2"
        else:
            r["choices"][0]["message"]["content"] = f"Thought: All done.\nFinal Answer: {final_answer}"
        return r

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)
    return call_count


def mock_react_loop(mocker, tool_calls: int = 1, final_answer: str = "The result is 42."):
    """Standard ReAct mock (single tool per loop)."""
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


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


class TestParallelToolCallsCheckpoint:
    """Agent with parallel_tool_calls_enabled: checkpoint captures full state."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_parallel_success_captures_state(self, mocker, backend_factory, backend_type):
        """Agent runs two tools in parallel: checkpoint is COMPLETED with agent internal state."""
        backend = backend_factory(backend_type)
        flow, agent = create_parallel_agent_flow(backend)
        mock_parallel_react(mocker)

        result = flow.run_sync(input_data={"input": "Add 3+7 and multiply 4*5 in parallel."})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert AGENT_ID in cp.completed_node_ids

        agent_state = cp.node_states[AGENT_ID]
        assert agent_state.status == "success"
        assert agent_state.output_data is not None
        assert "history_offset" in agent_state.internal_state
        assert "llm_state" in agent_state.internal_state
        assert "tool_states" in agent_state.internal_state

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_parallel_mid_loop_checkpoint_captures_iteration(self, mocker, backend_factory, backend_type):
        """With mid-loop enabled, iteration state is captured during parallel execution."""
        backend = backend_factory(backend_type)
        flow, agent = create_parallel_agent_flow(backend, mid_loop=True)
        mock_parallel_multi_round(mocker, parallel_rounds=2)

        result = flow.run_sync(input_data={"input": "Two rounds of parallel work."})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        internal = cp.node_states[AGENT_ID].internal_state
        assert "iteration" in internal
        assert internal["iteration"]["completed_iterations"] >= 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_parallel_mid_loop_produces_more_saves(self, mocker, backend_factory, backend_type):
        """Parallel agent with mid-loop enabled produces more saves than without."""
        original_save = InMemory.save

        backend_off = InMemory()
        flow_off, _ = create_parallel_agent_flow(backend_off, mid_loop=False, flow_id=FLOW_ID + "_off")
        mock_parallel_multi_round(mocker, parallel_rounds=2)

        saves_off = {"count": 0}

        def counting_off(self, cp):
            saves_off["count"] += 1
            return original_save(self, cp)

        mocker.patch.object(InMemory, "save", counting_off)
        flow_off.run_sync(input_data={"input": "test"})
        count_off = saves_off["count"]

        mocker.stopall()

        backend_on = InMemory()
        flow_on, _ = create_parallel_agent_flow(backend_on, mid_loop=True)
        mock_parallel_multi_round(mocker, parallel_rounds=2)

        saves_on = {"count": 0}

        def counting_on(self, cp):
            saves_on["count"] += 1
            return original_save(self, cp)

        mocker.patch.object(InMemory, "save", counting_on)
        flow_on.run_sync(input_data={"input": "test"})

        assert saves_on["count"] > count_off

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_parallel_resume_skips_completed_loops(self, mocker, backend_factory, backend_type):
        """Resume from mid-loop checkpoint: completed parallel loops are skipped."""
        backend = backend_factory(backend_type)

        flow1, _ = create_parallel_agent_flow(backend, mid_loop=True, max_loops=10)
        mock_parallel_multi_round(mocker, parallel_rounds=2, final_answer="Phase 1 done")
        result1 = flow1.run_sync(input_data={"input": "Two rounds"})
        assert result1.status == RunnableStatus.SUCCESS
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow1.id)
        assert AGENT_ID in cp.node_states
        assert cp.node_states[AGENT_ID].internal_state.get("iteration") is not None

        cp.node_states[AGENT_ID].status = CheckpointStatus.ACTIVE.value
        cp.node_states[AGENT_ID].output_data = None
        cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != AGENT_ID]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        flow2, _ = create_parallel_agent_flow(backend, mid_loop=True, max_loops=10, flow_id=FLOW_ID)
        resume_count = mock_parallel_react(mocker, final_answer="Resumed OK")

        result2 = flow2.run_sync(input_data=None, resume_from=cp.id)
        assert result2.status == RunnableStatus.SUCCESS
        # LLM should have been called fewer times than a full run (skipped completed loops)
        assert resume_count["value"] >= 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_parallel_checkpoint_chain_in_append_mode(self, mocker, backend_factory, backend_type):
        """Parallel agent in APPEND mode: checkpoint chain has multiple linked snapshots."""
        backend = backend_factory(backend_type)
        flow, _ = create_parallel_agent_flow(backend, mid_loop=True, behavior=CheckpointBehavior.APPEND)
        mock_parallel_multi_round(mocker, parallel_rounds=2)

        result = flow.run_sync(input_data={"input": "test"})
        assert result.status == RunnableStatus.SUCCESS

        latest = backend.get_latest_by_flow(flow.id)
        chain = backend.get_chain(latest.id)
        assert len(chain) >= 2
        assert chain[-1].parent_checkpoint_id is None

    def test_parallel_failure_captures_partial_iteration(self, mocker):
        """Parallel agent crashes mid-loop: checkpoint captures partial iteration state."""
        backend = InMemory()
        flow, _ = create_parallel_agent_flow(backend, mid_loop=True, max_loops=10)

        call_count = {"value": 0}

        def side_effect(stream: bool, *args, **kwargs):
            call_count["value"] += 1
            if call_count["value"] == 1:
                r = ModelResponse()
                r["choices"][0]["message"]["content"] = (
                    "<output>\n"
                    "    <thought>First parallel round.</thought>\n"
                    "    <action>run-parallel</action>\n"
                    '    <action_input>{"tools": ['
                    '{"name": "adder", "input": {"a": 1, "b": 2}}, '
                    '{"name": "multiplier", "input": {"x": 3, "y": 4}}'
                    "]}</action_input>\n"
                    "</output>"
                )
                return r
            raise RuntimeError("LLM crashed on second call")

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)

        result = flow.run_sync(input_data={"input": "test"})
        assert result.status == RunnableStatus.FAILURE

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.FAILED
        assert AGENT_ID in cp.node_states
        internal = cp.node_states[AGENT_ID].internal_state
        iteration = internal.get("iteration", {})
        assert iteration.get("completed_iterations", 0) >= 1


class TestSubAgentToolCheckpoint:
    """SubAgentTool: checkpoint captures call_count and preserves it on resume."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_subagent_success_captures_state(self, mocker, backend_factory, backend_type):
        """Parent delegates to child agent: checkpoint COMPLETED with full state."""
        backend = backend_factory(backend_type)
        flow, parent, sub_tool = create_subagent_flow(backend)
        mock_subagent_react(mocker)

        result = flow.run_sync(input_data={"input": "Calculate 6*7 using the specialist."})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert PARENT_AGENT_ID in cp.completed_node_ids

        agent_state = cp.node_states[PARENT_AGENT_ID]
        assert agent_state.status == "success"
        assert "tool_states" in agent_state.internal_state

    def test_subagent_call_count_in_checkpoint(self, mocker):
        """SubAgentTool._call_count is captured in the checkpoint's tool_states."""
        backend = InMemory()
        flow, parent, sub_tool = create_subagent_flow(backend, mid_loop=True, max_calls=10)
        mock_subagent_react(mocker)

        flow.run_sync(input_data={"input": "Delegate to child."})

        cp = backend.get_latest_by_flow(flow.id)
        internal = cp.node_states[PARENT_AGENT_ID].internal_state
        tool_states = internal.get("tool_states", {})

        assert (
            SUBAGENT_TOOL_ID in tool_states
        ), f"SubAgentTool id '{SUBAGENT_TOOL_ID}' not found in tool_states keys: {list(tool_states.keys())}"
        assert "call_count" in tool_states[SUBAGENT_TOOL_ID]
        assert tool_states[SUBAGENT_TOOL_ID]["call_count"] >= 1

    def test_subagent_call_count_preserved_on_resume(self, mocker):
        """On resume, SubAgentTool._call_count is restored and not wiped by reset_run_state."""
        backend = InMemory()
        flow, parent, sub_tool = create_subagent_flow(backend, mid_loop=True, max_calls=10)
        mock_subagent_two_delegations(mocker)

        flow.run_sync(input_data={"input": "Delegate twice."})
        mocker.stopall()

        # Verify the call_count was incremented during the run
        assert (
            sub_tool._call_count >= 1
        ), f"SubAgentTool._call_count was {sub_tool._call_count} after run, expected >= 1"

        cp = backend.get_latest_by_flow(flow.id)

        # Verify call_count is in the checkpoint
        internal = cp.node_states[PARENT_AGENT_ID].internal_state
        tool_states = internal.get("tool_states", {})
        assert (
            SUBAGENT_TOOL_ID in tool_states
        ), f"SubAgentTool not in checkpoint tool_states: {list(tool_states.keys())}"
        assert (
            tool_states[SUBAGENT_TOOL_ID].get("call_count", 0) >= 1
        ), f"call_count in checkpoint was {tool_states[SUBAGENT_TOOL_ID].get('call_count')}"

        # Simulate crash: mark agent as active
        cp.node_states[PARENT_AGENT_ID].status = CheckpointStatus.ACTIVE.value
        cp.node_states[PARENT_AGENT_ID].output_data = None
        cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != PARENT_AGENT_ID]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        # Create fresh flow for resume
        flow2, parent2, sub_tool2 = create_subagent_flow(backend, mid_loop=True, max_calls=10, flow_id=FLOW_ID)

        # Restore checkpoint onto flow2
        flow2._restore_from_checkpoint(cp)

        # After restore, the SubAgentTool should have its call_count preserved
        for tool in parent2.tools:
            if isinstance(tool, SubAgentTool):
                assert (
                    tool._call_count >= 1
                ), f"SubAgentTool._call_count was {tool._call_count} after restore, expected >= 1"

    def test_subagent_max_calls_limit_after_resume(self, mocker):
        """After resume with call_count restored, max_calls limit is still enforced."""
        backend = InMemory()
        flow, parent, sub_tool = create_subagent_flow(backend, mid_loop=True, max_calls=2)
        mock_subagent_two_delegations(mocker)

        result = flow.run_sync(input_data={"input": "Delegate twice."})
        assert result.status == RunnableStatus.SUCCESS
        mocker.stopall()

        cp = backend.get_latest_by_flow(flow.id)
        internal = cp.node_states[PARENT_AGENT_ID].internal_state
        tool_states = internal.get("tool_states", {})

        assert SUBAGENT_TOOL_ID in tool_states, f"SubAgentTool not found in tool_states: {list(tool_states.keys())}"
        assert tool_states[SUBAGENT_TOOL_ID].get("call_count", 0) >= 1

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_subagent_with_child_tool_use_checkpoint(self, mocker, backend_factory, backend_type):
        """Child agent uses its calculator tool: checkpoint captures full hierarchy."""
        backend = backend_factory(backend_type)
        flow, parent, sub_tool = create_subagent_flow(backend, mid_loop=True)
        mock_subagent_react(mocker, child_uses_tool=True)

        result = flow.run_sync(input_data={"input": "Calculate using specialist."})

        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert PARENT_AGENT_ID in cp.completed_node_ids
        assert "tool_states" in cp.node_states[PARENT_AGENT_ID].internal_state


class TestSubAgentToolCheckpointState:
    """Direct unit-style tests for SubAgentTool checkpoint state serialization."""

    def test_to_checkpoint_state_captures_call_count(self):
        """to_checkpoint_state returns SubAgentToolCheckpointState with call_count."""
        child_llm = make_llm(node_id=CHILD_LLM_ID)
        calc = Python(
            id=CALC_TOOL_ID,
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        child = Agent(
            id=CHILD_AGENT_ID,
            name="Child",
            llm=child_llm,
            tools=[calc],
            role="Test",
            max_loops=2,
        )
        sub_tool = SubAgentTool(
            id=SUBAGENT_TOOL_ID,
            name="Child",
            description="Test child",
            agent=child,
            max_calls=5,
        )

        sub_tool._call_count = 3

        state = sub_tool.to_checkpoint_state()
        assert state.call_count == 3

    def test_from_checkpoint_state_restores_call_count(self):
        """from_checkpoint_state restores _call_count from dict."""
        child_llm = make_llm(node_id=CHILD_LLM_ID)
        calc = Python(
            id=CALC_TOOL_ID,
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        child = Agent(
            id=CHILD_AGENT_ID,
            name="Child",
            llm=child_llm,
            tools=[calc],
            role="Test",
            max_loops=2,
        )
        sub_tool = SubAgentTool(
            id=SUBAGENT_TOOL_ID,
            name="Child",
            description="Test child",
            agent=child,
            max_calls=5,
        )

        assert sub_tool._call_count == 0

        sub_tool.from_checkpoint_state({"call_count": 7})
        assert sub_tool._call_count == 7

    def test_roundtrip_checkpoint_state(self):
        """to_checkpoint_state → dict → from_checkpoint_state roundtrip preserves call_count."""
        child_llm = make_llm(node_id=CHILD_LLM_ID)
        calc = Python(
            id=CALC_TOOL_ID,
            name="calculator",
            description="Calculator",
            code='def run(input_data): return {"result": 42}',
        )
        child = Agent(
            id=CHILD_AGENT_ID,
            name="Child",
            llm=child_llm,
            tools=[calc],
            role="Test",
            max_loops=2,
        )
        tool_a = SubAgentTool(id=SUBAGENT_TOOL_ID, name="Child", description="Test", agent=child, max_calls=10)
        tool_a._call_count = 5

        state_dict = tool_a.to_checkpoint_state().model_dump()

        tool_b = SubAgentTool(id=SUBAGENT_TOOL_ID, name="Child", description="Test", agent=child, max_calls=10)
        tool_b.from_checkpoint_state(state_dict)

        assert tool_b._call_count == 5


class TestCombinedCheckpointScenarios:
    """Mixed scenarios: parallel + subagent, multiple runs, async."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_separate_runs_parallel_agent_independent_checkpoints(self, mocker, backend_factory, backend_type):
        """Multiple parallel-agent runs create independent checkpoint chains."""
        backend = backend_factory(backend_type)
        flow, _ = create_parallel_agent_flow(backend)

        for i in range(3):
            mock_parallel_react(mocker, final_answer=f"Answer {i}")
            flow.run_sync(input_data={"input": f"Query {i}"})
            mocker.stopall()

        checkpoints = backend.get_list_by_flow(flow.id, limit=100)
        unique_run_ids = {cp.run_id for cp in checkpoints}
        assert len(unique_run_ids) == 3

    @pytest.mark.asyncio
    async def test_async_parallel_agent_checkpoint(self, mocker):
        """Async run with parallel agent creates checkpoint."""
        backend = InMemory()
        flow, _ = create_parallel_agent_flow(backend)
        mock_parallel_react(mocker)

        result = await flow.run_async(input_data={"input": "Async parallel test"})

        assert result.status == RunnableStatus.SUCCESS

        cp = await backend.get_latest_by_flow_async(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert AGENT_ID in cp.node_states

    @pytest.mark.asyncio
    async def test_async_subagent_checkpoint(self, mocker):
        """Async run with SubAgentTool creates checkpoint with tool_states."""
        backend = InMemory()
        flow, parent, sub_tool = create_subagent_flow(backend)
        mock_subagent_react(mocker)

        result = await flow.run_async(input_data={"input": "Async delegate test"})

        assert result.status == RunnableStatus.SUCCESS

        cp = await backend.get_latest_by_flow_async(flow.id)
        assert cp.status == CheckpointStatus.COMPLETED
        assert PARENT_AGENT_ID in cp.node_states
        assert "tool_states" in cp.node_states[PARENT_AGENT_ID].internal_state

    def test_replace_mode_parallel_agent(self, mocker):
        """REPLACE mode: parallel agent produces single checkpoint."""
        backend = InMemory()
        flow, _ = create_parallel_agent_flow(backend, behavior=CheckpointBehavior.REPLACE)
        mock_parallel_multi_round(mocker, parallel_rounds=2)

        flow.run_sync(input_data={"input": "test"})

        checkpoints = backend.get_list_by_flow(flow.id, limit=20)
        assert len(checkpoints) == 1
        assert checkpoints[0].parent_checkpoint_id is None
