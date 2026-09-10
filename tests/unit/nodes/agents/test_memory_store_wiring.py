import json

import pytest

from dynamiq.connections import E2B, Dynamiq
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.storages.memory import CompositeMemoryStore, DynamiqMemoryStore, MemoryStoreConfig
from tests.unit.storages.memory.conftest import FakeMemoryStore


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


@pytest.fixture
def backend():
    return DynamiqMemoryStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
        memory_store_id="ms-123",
        user_id="u-42",
        description="What you learn about this user.",
    )


def _memory(backend, **kwargs):
    return MemoryStoreConfig(enabled=True, backend=backend, **kwargs)


def _tool_names(agent):
    return [tool.name for tool in agent.tools]


def _ops_block(agent):
    return agent.system_prompt_manager._prompt_blocks.get("operational_instructions", "")


def _sandbox():
    return SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-1"))


def test_standalone_attaches_one_tool(llm, backend):
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend))

    assert _tool_names(agent) == ["memory-store"]
    assert agent.tools[0].backend is backend


def test_a_file_store_agent_keeps_its_own_tools(llm, backend):
    """Memory no longer routes under the file tools; it sits beside them with its own."""
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        memory_store=_memory(backend),
    )

    assert _tool_names(agent) == ["file-read", "file-search", "file-list", "file-write", "memory-store"]
    file_tools = [t for t in agent.tools if t.name.startswith("file-")]
    assert all(tool.file_store is agent.file_store_backend for tool in file_tools)
    assert isinstance(agent.file_store_backend, InMemoryFileStore), "the workspace backend must not be wrapped"


def test_a_sandbox_agent_keeps_its_sandbox_tools(llm, backend):
    agent = Agent(name="a", llm=llm, sandbox=_sandbox(), memory_store=_memory(backend))

    names = _tool_names(agent)
    assert "memory-store" in names
    assert "sandbox-shell" in names, f"sandbox tools were displaced: {names}"


def test_every_workspace_gets_the_same_tool(llm, backend):
    """The point of the rework: one surface regardless of what else the agent has."""
    configurations = [
        {},
        {"file_store": FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True)},
        {"sandbox": _sandbox()},
    ]

    for extra in configurations:
        agent = Agent(name="a", llm=llm, memory_store=_memory(backend), **extra)
        assert [t.name for t in agent.tools if t.name == "memory-store"] == ["memory-store"]


def test_write_disabled_is_passed_to_the_tool(llm, backend):
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend, write_enabled=False))

    assert agent.tools[0].write_enabled is False
    assert "read-only for you" in _ops_block(agent)


def test_disabled_config_attaches_nothing(llm, backend):
    agent = Agent(name="a", llm=llm, memory_store=MemoryStoreConfig(enabled=False, backend=backend))

    assert _tool_names(agent) == []
    assert agent.memory_store_backend is None
    assert "## Memory" not in _ops_block(agent)


def test_an_agent_without_memory_is_unchanged(llm):
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
    )

    assert _tool_names(agent) == ["file-read", "file-search", "file-list", "file-write"]
    assert "## Memory" not in _ops_block(agent)


def test_the_tool_is_not_serialized(llm, backend):
    """It is rebuilt from `memory_store` on load, like the sandbox and skills tools."""
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend))

    assert agent.to_dict()["tools"] == []


def test_serialization_hides_credentials(llm, backend):
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend))

    data = agent.to_dict()

    assert data["memory_store"]["enabled"] is True
    assert data["memory_store"]["backend"]["memory_store_id"] == "ms-123"
    assert "secret-token" not in json.dumps(data, default=str)


def test_the_prompt_names_the_tool_and_its_actions(llm, backend):
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend))

    ops = _ops_block(agent)
    assert "## Memory" in ops
    assert "memory-store" in ops
    assert "'list'" in ops and "'write'" in ops


def test_the_prompt_describes_each_memory(llm):
    agent = Agent(
        name="a",
        llm=llm,
        memory_store=_memory(
            CompositeMemoryStore(
                routes={
                    "user/": FakeMemoryStore(description="What you learn about this user."),
                    "team/": FakeMemoryStore(description="Conventions the whole team follows."),
                }
            )
        ),
    )

    ops = _ops_block(agent)
    assert "- user/ - What you learn about this user." in ops
    assert "- team/ - Conventions the whole team follows." in ops
    assert "Put each fact in the one it belongs to" in ops


def test_the_sandbox_claim_to_be_memory_is_contradicted(llm, backend):
    """The sandbox block calls itself long-term memory; this must correct it, after it."""
    agent = Agent(name="a", llm=llm, sandbox=_sandbox(), memory_store=_memory(backend))

    assert "NOT your long-term memory" in _ops_block(agent)


# A file-store agent is deliberately absent: it cannot round-trip through YAML at all, with or
# without memory, because `FileSearchTool` is not exported from `dynamiq.nodes.tools` and the loader
# resolves nodes by import path. That is pre-existing and unrelated to this feature.
@pytest.mark.parametrize("with_sandbox", [False, True])
def test_yaml_round_trip_rebuilds_the_tool(llm, backend, tmp_path, with_sandbox):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    kwargs = {"sandbox": _sandbox()} if with_sandbox else {}
    agent = Agent(name="a", llm=llm, memory_store=_memory(backend), **kwargs)
    path = str(tmp_path / "wf.yaml")
    Workflow(flow=Flow(nodes=[agent])).to_yaml_file(path)

    reloaded = Workflow.from_yaml_file(path, init_components=True).flow.nodes[0]

    assert reloaded.memory_store.enabled is True
    assert isinstance(reloaded.memory_store.backend, DynamiqMemoryStore)
    assert reloaded.memory_store.backend.memory_store_id == "ms-123"
    assert reloaded.memory_store.backend.user_id == "u-42"
    assert [t.name for t in reloaded.tools if t.name == "memory-store"] == ["memory-store"]


def test_several_memories_round_trip(llm, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    def remote(store_id, description):
        return DynamiqMemoryStore(
            connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
            memory_store_id=store_id,
            user_id="u-42",
            description=description,
        )

    agent = Agent(
        name="a",
        llm=llm,
        memory_store=_memory(
            CompositeMemoryStore(
                routes={"user/": remote("ms-user", "About this user."), "team/": remote("ms-team", "Team rules.")}
            )
        ),
    )
    path = str(tmp_path / "wf.yaml")
    Workflow(flow=Flow(nodes=[agent])).to_yaml_file(path)

    reloaded = Workflow.from_yaml_file(path, init_components=True).flow.nodes[0]

    routes = reloaded.memory_store.backend.routes
    assert set(routes) == {"user/", "team/"}
    assert routes["team/"].memory_store_id == "ms-team"
    assert reloaded.memory_store.backend.describe_namespaces() == {
        "user/": "About this user.",
        "team/": "Team rules.",
    }
