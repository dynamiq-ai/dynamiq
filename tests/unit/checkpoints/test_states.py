"""Unit tests for checkpoint state save/restore on nodes and components."""

import pytest

from dynamiq import connections
from dynamiq.checkpoints.checkpoint import (
    CheckpointContext,
    CheckpointMixin,
    CheckpointStatus,
    FlowCheckpoint,
    IterationState,
    IterativeCheckpointMixin,
)
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import AgentCheckpointState
from dynamiq.nodes.tools import Python
from dynamiq.nodes.tools.thinking_tool import ThinkingTool
from dynamiq.types.feedback import ApprovalInputData

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"


def create_test_llm(node_id: str = "test-llm") -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


class TestAgentCheckpointState:
    """Tests for Agent checkpoint state save/restore including LLM component."""

    @pytest.fixture
    def agent_node(self):
        return Agent(
            id="test-agent",
            name="Test Agent",
            llm=create_test_llm("agent-llm"),
            role="Test assistant",
            goal="Help with testing",
        )

    def test_to_checkpoint_state_returns_agent_and_llm_state(self, agent_node):
        state = agent_node.to_checkpoint_state()

        assert isinstance(state, AgentCheckpointState)
        assert state.history_offset == 2
        assert state.llm_state is not None
        assert "is_fallback_run" in state.llm_state

    def test_from_checkpoint_state_sets_agent_and_llm_state(self, agent_node):
        agent_node.from_checkpoint_state(
            {
                "history_offset": 5,
                "llm_state": {"is_fallback_run": True},
            }
        )

        assert agent_node._history_offset == 5
        assert agent_node._is_resumed is True
        assert agent_node.llm._is_fallback_run is True

    def test_restore_with_empty_state_uses_defaults(self, agent_node):
        agent_node.from_checkpoint_state({})

        assert agent_node._history_offset == 2
        assert agent_node._is_resumed is True

    def test_checkpoint_state_roundtrip(self):
        agent = Agent(
            id="roundtrip-agent",
            name="Roundtrip Agent",
            llm=create_test_llm("agent-llm"),
            role="Test",
            goal="Test",
        )
        agent._history_offset = 10
        agent.llm._is_fallback_run = True

        saved_state = agent.to_checkpoint_state()

        new_agent = Agent(
            id="new-agent",
            name="New Agent",
            llm=create_test_llm("new-llm"),
            role="Test",
            goal="Test",
        )
        new_agent.from_checkpoint_state(saved_state)

        assert new_agent._history_offset == 10
        assert new_agent._is_resumed is True
        assert new_agent.llm._is_fallback_run is True
        assert new_agent.llm._is_resumed is True


class TestAgentCheckpointMixin:
    """Tests for Agent CheckpointMixin protocol implementation."""

    @pytest.fixture
    def agent_node(self):
        return Agent(
            id="test-agent",
            name="Test Agent",
            llm=create_test_llm("agent-llm"),
            role="Test assistant",
            goal="Help with testing",
        )

    def test_agent_implements_checkpoint_mixin(self, agent_node):
        assert isinstance(agent_node, CheckpointMixin)
        assert hasattr(agent_node, "to_checkpoint_state")
        assert hasattr(agent_node, "from_checkpoint_state")
        assert hasattr(agent_node, "is_resumed")
        assert hasattr(agent_node, "reset_resumed_flag")

    def test_is_resumed_property(self, agent_node):
        assert agent_node.is_resumed is False
        agent_node.from_checkpoint_state({})
        assert agent_node.is_resumed is True

    def test_reset_resumed_flag(self, agent_node):
        agent_node.from_checkpoint_state({})
        assert agent_node.is_resumed is True
        agent_node.reset_resumed_flag()
        assert agent_node.is_resumed is False


class TestAgentToolCheckpointState:
    """Tests for Agent checkpoint state including tool components."""

    @pytest.fixture
    def agent_with_thinking_tool(self):
        thinking_tool = ThinkingTool(
            id="thinking-tool",
            name="Thinking",
            llm=create_test_llm("thinking-llm"),
            memory_enabled=True,
        )
        agent = Agent(
            id="agent-with-tools",
            name="Agent",
            llm=create_test_llm("agent-llm"),
            tools=[thinking_tool],
            role="Test",
            goal="Test",
        )
        return agent, thinking_tool

    def test_agent_saves_tool_states(self, agent_with_thinking_tool):
        agent, thinking_tool = agent_with_thinking_tool
        thinking_tool._thought_history = [{"thought": "Step 1", "focus": "planning"}]

        state = agent.to_checkpoint_state()

        assert thinking_tool.id in state.tool_states
        assert state.tool_states[thinking_tool.id]["thought_history"] == thinking_tool._thought_history

    def test_agent_restores_tool_states(self, agent_with_thinking_tool):
        agent, thinking_tool = agent_with_thinking_tool

        agent.from_checkpoint_state(
            {
                "history_offset": 5,
                "llm_state": {"is_fallback_run": False},
                "tool_states": {
                    thinking_tool.id: {
                        "thought_history": [{"thought": "Restored", "focus": "test"}],
                        "llm_state": {"is_fallback_run": True},
                    },
                },
            }
        )

        assert agent._history_offset == 5
        assert thinking_tool._thought_history == [{"thought": "Restored", "focus": "test"}]
        assert thinking_tool.llm._is_fallback_run is True

    def test_agent_tool_state_roundtrip(self, agent_with_thinking_tool):
        agent, thinking_tool = agent_with_thinking_tool
        agent._history_offset = 8
        agent.llm._is_fallback_run = True
        thinking_tool._thought_history = [{"thought": "Important", "focus": "debug"}]

        state_dict = agent.to_checkpoint_state().model_dump()

        new_thinking_tool = ThinkingTool(
            id="thinking-tool",
            name="Thinking",
            llm=create_test_llm("thinking-llm"),
            memory_enabled=True,
        )
        new_agent = Agent(
            id="agent-with-tools",
            name="Agent",
            llm=create_test_llm("agent-llm"),
            tools=[new_thinking_tool],
            role="Test",
            goal="Test",
        )
        new_agent.from_checkpoint_state(state_dict)

        assert new_agent._history_offset == 8
        assert new_agent.llm._is_fallback_run is True
        assert new_thinking_tool._thought_history == [{"thought": "Important", "focus": "debug"}]

    def test_agent_with_no_tools_has_empty_tool_states(self):
        agent = Agent(
            id="no-tools",
            name="Agent",
            llm=create_test_llm("agent-llm"),
            role="Test",
            goal="Test",
        )
        assert agent.to_checkpoint_state().tool_states == {}


class TestThinkingToolCheckpointState:
    """Tests for ThinkingTool checkpoint state roundtrip."""

    @pytest.fixture
    def thinking_tool(self):
        return ThinkingTool(
            id="thinking",
            name="Thinking",
            llm=create_test_llm("thinking-llm"),
            memory_enabled=True,
        )

    def test_saves_history_and_llm(self, thinking_tool):
        thinking_tool._thought_history = [
            {"thought": "Analysis step 1", "focus": "planning"},
            {"thought": "Analysis step 2", "focus": "implementation"},
        ]
        thinking_tool.llm._is_fallback_run = True

        state = thinking_tool.to_checkpoint_state()

        assert len(state.thought_history) == 2
        assert state.thought_history[0]["thought"] == "Analysis step 1"
        assert state.llm_state["is_fallback_run"] is True

    def test_restores_history_and_llm(self, thinking_tool):
        thinking_tool.from_checkpoint_state(
            {
                "thought_history": [{"thought": "Restored thought", "focus": "test"}],
                "llm_state": {"is_fallback_run": True},
            }
        )

        assert thinking_tool._thought_history == [{"thought": "Restored thought", "focus": "test"}]
        assert thinking_tool.llm._is_fallback_run is True
        assert thinking_tool.is_resumed is True

    def test_roundtrip(self, thinking_tool):
        thinking_tool._thought_history = [{"thought": "Deep analysis", "focus": "debug"}]
        thinking_tool.llm._is_fallback_run = True

        state_dict = thinking_tool.to_checkpoint_state().model_dump()

        new_tool = ThinkingTool(
            id="thinking",
            name="Thinking",
            llm=create_test_llm("thinking-llm"),
            memory_enabled=True,
        )
        new_tool.from_checkpoint_state(state_dict)

        assert new_tool._thought_history == [{"thought": "Deep analysis", "focus": "debug"}]
        assert new_tool.llm._is_fallback_run is True

    def test_empty_history(self, thinking_tool):
        state = thinking_tool.to_checkpoint_state()
        assert state.thought_history == []


class TestHITLCheckpointContext:
    """Tests for HITL checkpoint context and FlowCheckpoint pending input model."""

    def test_checkpoint_context_tracks_pending_input(self):
        pending_calls = []
        received_calls = []
        ctx = CheckpointContext(
            on_pending_input=lambda nid, prompt, meta: pending_calls.append((nid, prompt, meta)),
            on_input_received=lambda nid: received_calls.append(nid),
        )

        ctx.mark_pending_input("node-1", "Approve?", {"event": "approval"})
        assert pending_calls == [("node-1", "Approve?", {"event": "approval"})]

        ctx.mark_input_received("node-1")
        assert received_calls == ["node-1"]

    def test_flow_checkpoint_pending_input_lifecycle(self):
        cp = FlowCheckpoint(flow_id="test", run_id="run-1")

        cp.mark_pending_input("node-1", "Approve?", {"event": "approval"})
        assert cp.status == CheckpointStatus.PENDING_INPUT
        assert cp.has_pending_inputs()
        assert cp.get_pending_input("node-1").prompt == "Approve?"

        cp.mark_pending_input("node-2", "Confirm?")
        assert len(cp.pending_inputs) == 2

        cp.clear_pending_input("node-1")
        assert cp.status == CheckpointStatus.PENDING_INPUT

        cp.clear_pending_input("node-2")
        assert cp.status == CheckpointStatus.ACTIVE
        assert not cp.has_pending_inputs()

    def test_save_mid_run_callback(self):
        mid_run_calls = []
        ctx = CheckpointContext(on_save_mid_run=lambda nid: mid_run_calls.append(nid))

        ctx.save_mid_run("agent-1")
        ctx.save_mid_run("agent-1")
        ctx.save_mid_run("agent-2")

        assert mid_run_calls == ["agent-1", "agent-1", "agent-2"]

    def test_save_mid_run_noop_when_no_callback(self):
        ctx = CheckpointContext()
        ctx.save_mid_run("agent-1")


class TestNodeApprovalCheckpoint:
    """Tests for Node approval response checkpoint persistence."""

    def test_node_saves_approval_response(self):
        node = Python(id="n", name="N", code="def run(input_data): return {}")
        node._pending_approval_response = ApprovalInputData(feedback="approved", is_approved=True, data={"k": "v"})

        state = node.to_checkpoint_state().model_dump()
        assert state["approval_response"]["feedback"] == "approved"
        assert state["approval_response"]["is_approved"] is True

    def test_node_restores_approval_response(self):
        node = Python(id="n", name="N", code="def run(input_data): return {}")
        node.from_checkpoint_state(
            {
                "approval_response": {"feedback": "yes", "is_approved": True, "data": {"r": "ok"}},
            }
        )

        assert node._pending_approval_response is not None
        assert node._pending_approval_response.feedback == "yes"
        assert node.is_resumed is True

    def test_node_without_approval_has_no_response(self):
        node = Python(id="n", name="N", code="def run(input_data): return {}")
        state = node.to_checkpoint_state().model_dump()
        assert "approval_response" not in state or state.get("approval_response") is None


class TestIterativeCheckpointMixin:
    """Unit tests for the IterativeCheckpointMixin protocol."""

    def test_iteration_state_defaults(self):
        state = IterationState()
        assert state.completed_iterations == 0
        assert state.iteration_data == {}

    def test_iteration_state_with_data(self):
        state = IterationState(completed_iterations=5, iteration_data={"key": [1, 2, 3]})
        assert state.completed_iterations == 5
        dumped = state.model_dump()
        assert dumped["iteration_data"]["key"] == [1, 2, 3]

    def test_agent_is_iterative_checkpoint_mixin(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test")
        assert isinstance(agent, IterativeCheckpointMixin)

    def test_get_start_iteration_fresh(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test")
        assert agent.get_start_iteration() == 0

    def test_get_start_iteration_after_restore(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test")
        agent._iteration_state = IterationState(completed_iterations=3)
        agent._has_restored_iteration = True
        assert agent.get_start_iteration() == 3
        assert agent.get_start_iteration() == 0

    def test_agent_checkpoint_includes_iteration(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test", max_loops=10)
        agent._completed_loops = 7
        state = agent.to_checkpoint_state()
        dumped = state.model_dump()
        assert dumped["iteration"] is not None
        assert dumped["iteration"]["completed_iterations"] == 7

    def test_agent_from_checkpoint_restores_iteration(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test")
        agent.from_checkpoint_state(
            {
                "history_offset": 2,
                "llm_state": {},
                "iteration": {
                    "completed_iterations": 4,
                    "iteration_data": {"prompt_messages": [{"content": "hi", "role": "user", "static": False}]},
                },
            }
        )
        assert agent.get_start_iteration() == 4

    def test_agent_from_checkpoint_without_iteration_is_backward_compatible(self):
        agent = Agent(id="a", name="A", llm=create_test_llm(), role="Test")
        agent.from_checkpoint_state({"history_offset": 2, "llm_state": {}})
        assert agent.get_start_iteration() == 0

    def test_save_iteration_roundtrip(self):
        agent = Agent(id="rt", name="RT", llm=create_test_llm(), role="Test", max_loops=10)
        agent._completed_loops = 6

        state_dict = agent.to_checkpoint_state().model_dump()

        new_agent = Agent(id="rt2", name="RT2", llm=create_test_llm(), role="Test", max_loops=10)
        new_agent.from_checkpoint_state(state_dict)
        assert new_agent.get_start_iteration() == 6
