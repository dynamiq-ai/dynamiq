"""Integration tests for sandbox reconnect info in agent checkpoints.

Verifies that an agent's own sandbox survives a checkpoint round-trip:

- a started sandbox's id is captured in ``AgentCheckpointState.sandbox_state``
- the state survives JSON serialization and the InMemory / FileSystem backends
- an agent restored from that checkpoint reconnects to the SAME sandbox on
  first use (``connect(sandbox_id=...)``) instead of creating a new one
- resuming a paused flow reconnects the agent's sandbox before the ReAct loop
- restore refuses to bind a differently-typed backend to the saved id, and
  an explicitly configured sandbox_id wins over the checkpointed one

The E2B SDK is faked at ``E2BSandbox._sdk_class`` so no network is needed.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from litellm import ModelResponse

from dynamiq import connections, flows
from dynamiq.checkpoints.backends.filesystem import FileSystem
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointStatus
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.nodes import llms
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.checkpoint import AgentCheckpointState
from dynamiq.runnables import RunnableStatus
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox

TEST_API_KEY = "test-api-key"
LLM_MODEL = "gpt-4o-mini"
FLOW_ID = "sandbox-checkpoint-flow"
AGENT_ID = "sandbox-agent"
MARKER_FILE = "checkpoint_marker.txt"
MARKER_CONTENT = b"written before checkpoint"


class FakeE2BSdk:
    """Stand-in for the E2B SDK class: ``create()`` mints a new sandbox id,
    ``connect(sandbox_id=...)`` returns a fake bound to an existing one.

    Files are stored per sandbox id so content written before a checkpoint is
    visible only through a sandbox reconnected to the same id.
    """

    def __init__(self):
        self.created: list[str] = []
        self.connected: list[str] = []
        self._files: dict[str, dict[str, bytes]] = {}

    def _make_sandbox(self, sandbox_id: str) -> MagicMock:
        files = self._files.setdefault(sandbox_id, {})
        fake = MagicMock()
        fake.sandbox_id = sandbox_id
        fake.files.write.side_effect = lambda path, data: files.__setitem__(path, data)
        fake.files.read.side_effect = lambda path, *_: files[path]
        return fake

    def create(self, **kwargs) -> MagicMock:
        sandbox_id = f"sbx-{len(self.created) + 1}"
        self.created.append(sandbox_id)
        return self._make_sandbox(sandbox_id)

    def connect(self, sandbox_id: str, **kwargs) -> MagicMock:
        self.connected.append(sandbox_id)
        return self._make_sandbox(sandbox_id)


@pytest.fixture
def fake_sdk():
    sdk = FakeE2BSdk()
    with patch.object(E2BSandbox, "_sdk_class", new_callable=lambda: property(lambda self: sdk)):
        yield sdk


def make_llm(node_id: str) -> llms.OpenAI:
    return llms.OpenAI(
        id=node_id,
        model=LLM_MODEL,
        connection=connections.OpenAI(api_key=TEST_API_KEY),
        is_postponed_component_init=True,
    )


def make_sandbox_agent(agent_id: str = AGENT_ID, sandbox_id: str | None = None, max_loops: int = 5) -> Agent:
    return Agent(
        id=agent_id,
        name="Sandbox Agent",
        llm=make_llm(f"{agent_id}-llm"),
        role="Test",
        max_loops=max_loops,
        sandbox=SandboxConfig(
            enabled=True,
            backend=E2BSandbox(connection=connections.E2B(api_key=TEST_API_KEY), sandbox_id=sandbox_id),
        ),
    )


def make_flow(backend, agent: Agent) -> flows.Flow:
    return flows.Flow(
        id=FLOW_ID,
        nodes=[agent],
        checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_mid_agent_loop_enabled=True),
    )


def mock_final_answer(mocker, answer: str = "Done."):
    """Mock the LLM so the agent answers immediately without calling any tool."""

    def side_effect(stream: bool, *args, **kwargs):
        r = ModelResponse()
        r["choices"][0]["message"]["content"] = f"Thought: Done.\nFinal Answer: {answer}"
        return r

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=side_effect)


@pytest.fixture
def backend_factory(tmp_path):
    def _create(backend_type: str):
        if backend_type == "in_memory":
            return InMemory()
        return FileSystem(base_path=str(tmp_path / ".dynamiq" / "checkpoints"))

    return _create


class TestSandboxCheckpointRoundTrip:
    """Sandbox reconnect info survives checkpoint save/restore."""

    def test_started_sandbox_is_captured_and_restored_agent_reconnects(self, fake_sdk):
        agent_a = make_sandbox_agent("agent-a")
        sandbox_a = agent_a.sandbox.backend
        sandbox_a.store(MARKER_FILE, MARKER_CONTENT)  # lazily starts the sandbox
        assert fake_sdk.created == ["sbx-1"]

        # Round-trip through JSON, as a real checkpoint backend would.
        state_dict = json.loads(json.dumps(agent_a.to_checkpoint_state().model_dump()))
        assert state_dict["sandbox_state"] == {
            "type": sandbox_a.type,
            "sandbox_id": "sbx-1",
            "base_path": sandbox_a.base_path,
        }

        agent_b = make_sandbox_agent("agent-b")
        sandbox_b = agent_b.sandbox.backend
        assert sandbox_b.sandbox_id is None

        agent_b.from_checkpoint_state(state_dict)
        assert sandbox_b.sandbox_id == "sbx-1"

        # First use reconnects instead of creating; the pre-checkpoint file is visible.
        assert sandbox_b.retrieve(MARKER_FILE) == MARKER_CONTENT
        assert fake_sdk.connected == ["sbx-1"]
        assert fake_sdk.created == ["sbx-1"], "restore must not create a new sandbox"

    def test_unstarted_sandbox_is_not_captured(self, fake_sdk):
        agent = make_sandbox_agent()
        assert agent.to_checkpoint_state().sandbox_state is None
        assert fake_sdk.created == []

    def test_restore_skips_on_sandbox_type_mismatch(self, fake_sdk):
        agent = make_sandbox_agent()
        agent.from_checkpoint_state(
            {"sandbox_state": {"type": "dynamiq.sandboxes.DaytonaSandbox", "sandbox_id": "sbx-other"}}
        )
        assert agent.sandbox.backend.sandbox_id is None

    def test_explicit_sandbox_id_wins_over_checkpoint(self, fake_sdk):
        agent = make_sandbox_agent(sandbox_id="sbx-explicit")
        agent.from_checkpoint_state({"sandbox_state": {"type": agent.sandbox.backend.type, "sandbox_id": "sbx-1"}})
        assert agent.sandbox.backend.sandbox_id == "sbx-explicit"

    def test_old_checkpoint_without_sandbox_state_is_backward_compatible(self, fake_sdk):
        agent = make_sandbox_agent()
        agent.from_checkpoint_state({"history_offset": 3})
        assert agent._history_offset == 3
        assert agent.sandbox.backend.sandbox_id is None


class TestSandboxCheckpointInFlow:
    """Sandbox reconnect info flows through a real Flow checkpoint and resume."""

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_flow_checkpoint_persists_sandbox_state(self, mocker, fake_sdk, backend_factory, backend_type):
        backend = backend_factory(backend_type)
        agent = make_sandbox_agent()
        agent.sandbox.backend.store(MARKER_FILE, MARKER_CONTENT)  # start the sandbox before the run
        flow = make_flow(backend, agent)
        mock_final_answer(mocker)

        result = flow.run_sync(input_data={"input": "hello"})
        assert result.status == RunnableStatus.SUCCESS

        cp = backend.get_latest_by_flow(flow.id)
        sandbox_state = cp.node_states[AGENT_ID].internal_state.get("sandbox_state")
        assert sandbox_state is not None
        assert sandbox_state["sandbox_id"] == "sbx-1"
        assert sandbox_state["type"] == agent.sandbox.backend.type

    @pytest.mark.parametrize("backend_type", ["in_memory", "file"])
    def test_resumed_flow_reconnects_to_checkpointed_sandbox(self, mocker, fake_sdk, backend_factory, backend_type):
        """A resumed run must reuse the sandbox from the checkpoint, not create a fresh one."""
        backend = backend_factory(backend_type)

        agent1 = make_sandbox_agent()
        agent1.sandbox.backend.store(MARKER_FILE, MARKER_CONTENT)
        flow1 = make_flow(backend, agent1)
        mock_final_answer(mocker, "First run")
        assert flow1.run_sync(input_data={"input": "hello"}).status == RunnableStatus.SUCCESS
        mocker.stopall()
        assert fake_sdk.created == ["sbx-1"]

        # Simulate a crash: the agent is still "active" in the checkpoint.
        cp = backend.get_latest_by_flow(flow1.id)
        cp.node_states[AGENT_ID].status = CheckpointStatus.ACTIVE.value
        cp.node_states[AGENT_ID].output_data = None
        cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != AGENT_ID]
        cp.status = CheckpointStatus.ACTIVE
        backend.save(cp)

        # New process: fresh agent, no sandbox id configured.
        agent2 = make_sandbox_agent()
        assert agent2.sandbox.backend.sandbox_id is None
        flow2 = make_flow(backend, agent2)
        mock_final_answer(mocker, "Resumed")

        assert flow2.run_sync(input_data=None, resume_from=cp.id).status == RunnableStatus.SUCCESS

        assert agent2.sandbox.backend.sandbox_id == "sbx-1"
        assert agent2.sandbox.backend.retrieve(MARKER_FILE) == MARKER_CONTENT
        assert fake_sdk.created == ["sbx-1"], "resume must not create a second sandbox"
        assert "sbx-1" in fake_sdk.connected


class TestSandboxCheckpointStateModel:
    def test_state_model_defaults_sandbox_state_to_none(self):
        assert AgentCheckpointState().sandbox_state is None
