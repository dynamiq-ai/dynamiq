import json

import pytest

from dynamiq.connections import E2B, Dynamiq
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file import CompositeFileStore, DynamiqFileStore, InMemoryFileStore
from dynamiq.storages.file.base import FileStoreConfig, PersistentStoreConfig


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


@pytest.fixture
def persistent_backend():
    return DynamiqFileStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
        memory_store_id="ms-123",
        user="u-42",
    )


class DeclinesCacheStore(InMemoryFileStore):
    """Stands in for a remote store: same semantics, but refuses extracted-text caching."""

    def supports_extracted_text_cache(self, file_path="") -> bool:
        return False


def _persistent(backend, **kwargs):
    return PersistentStoreConfig(enabled=True, backend=backend, **kwargs)


def _tool_names(agent):
    return [tool.name for tool in agent.tools]


def _ops_block(agent):
    return agent.system_prompt_manager._prompt_blocks.get("operational_instructions", "")


def test_file_store_mode_routes_through_a_composite(llm, persistent_backend):
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend),
    )

    assert _tool_names(agent) == ["file-read", "file-search", "file-list", "file-write"]

    backend = agent.file_store_backend
    assert isinstance(backend, CompositeFileStore)
    assert list(backend.routes) == ["memories/"]
    assert backend.routes["memories/"] is persistent_backend
    assert all(tool.file_store is backend for tool in agent.tools)


def test_file_store_mode_keeps_persistent_writes_out_of_the_workspace(llm):
    """A composite-routed write must land in the persistent backend, not the ephemeral one."""
    workspace, persistent = InMemoryFileStore(), InMemoryFileStore()
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=workspace, agent_file_write_enabled=True),
        persistent_store=_persistent(persistent),
    )

    agent.file_store_backend.store("memories/prefs.md", "remembered")
    agent.file_store_backend.store("scratch.md", "ephemeral")

    assert persistent.retrieve("memories/prefs.md") == b"remembered"
    assert workspace.retrieve("scratch.md") == b"ephemeral"
    assert not workspace.exists("memories/prefs.md")


def test_sandbox_mode_adds_dedicated_tools_and_leaves_the_sandbox_alone(llm, persistent_backend):
    agent = Agent(
        name="a",
        llm=llm,
        sandbox=SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-1")),
        persistent_store=_persistent(persistent_backend),
    )

    names = _tool_names(agent)
    assert {"memory-read", "memory-list", "memory-write"} <= set(names)

    memory_tools = [t for t in agent.tools if t.name.startswith("memory-")]
    assert all(tool.file_store is persistent_backend for tool in memory_tools)
    # Sandbox tools keep serving the sandbox itself (the shell tool has no file store at all).
    sandbox_tools = [
        t for t in agent.tools if not t.name.startswith("memory-") and "file_store" in type(t).model_fields
    ]
    assert sandbox_tools and all(tool.file_store is agent.sandbox_backend for tool in sandbox_tools)


def test_standalone_mode_yields_only_the_persistent_tools(llm, persistent_backend):
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))

    assert _tool_names(agent) == ["memory-read", "memory-list", "memory-write"]


def test_write_disabled_omits_the_write_tool(llm, persistent_backend):
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend, write_enabled=False))

    assert _tool_names(agent) == ["memory-read", "memory-list"]
    assert agent._persistent_store_writable is False


def test_persistent_tools_are_not_serialized(llm, persistent_backend):
    """They are rebuilt from `persistent_store` on load, like sandbox tools."""
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))

    assert agent.to_dict()["tools"] == []


def test_disabled_config_attaches_nothing(llm, persistent_backend):
    agent = Agent(
        name="a",
        llm=llm,
        persistent_store=PersistentStoreConfig(enabled=False, backend=persistent_backend),
    )

    assert _tool_names(agent) == []
    assert agent.persistent_store_backend is None
    assert "## Memory" not in _ops_block(agent)


def test_agent_without_a_persistent_store_is_unchanged(llm):
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
    )

    assert isinstance(agent.file_store_backend, InMemoryFileStore)
    assert "## Memory" not in _ops_block(agent)


def test_prompt_names_the_file_tools_in_file_store_mode(llm, persistent_backend):
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend),
    )

    ops = _ops_block(agent)
    assert "## Memory" in ops
    assert "memories/" in ops
    assert "file-write" in ops
    assert "memory-write" not in ops


def test_prompt_names_the_memory_tools_in_standalone_mode(llm, persistent_backend):
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))

    ops = _ops_block(agent)
    assert "memory-write" in ops
    assert "memory-list" in ops


def test_prompt_is_read_only_when_writes_are_disabled(llm, persistent_backend):
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend, write_enabled=False))

    ops = _ops_block(agent)
    assert "read-only" in ops
    assert "memory-write" not in ops


def test_file_store_mode_is_read_only_without_agent_file_write(llm, persistent_backend):
    """In composite mode writes ride on FileWriteTool, which the agent only attaches when allowed."""
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=False),
        persistent_store=_persistent(persistent_backend),
    )

    assert agent._persistent_store_writable is False
    assert "read-only" in _ops_block(agent)


def test_custom_path_prefix_is_used_everywhere(llm, persistent_backend):
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend, path_prefix="knowledge/"),
    )

    assert list(agent.file_store_backend.routes) == ["knowledge/"]
    assert "knowledge/" in _ops_block(agent)


def test_yaml_round_trip_rebuilds_the_store_and_its_tools(llm, persistent_backend, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))
    path = str(tmp_path / "wf.yaml")
    Workflow(flow=Flow(nodes=[agent])).to_yaml_file(path)

    reloaded = Workflow.from_yaml_file(path, init_components=True).flow.nodes[0]

    assert reloaded.persistent_store.enabled is True
    assert isinstance(reloaded.persistent_store.backend, DynamiqFileStore)
    assert reloaded.persistent_store.backend.memory_store_id == "ms-123"
    assert reloaded.persistent_store.backend.user == "u-42"
    assert _tool_names(reloaded) == ["memory-read", "memory-list", "memory-write"]


def test_agent_serialization_hides_persistent_credentials(llm, persistent_backend):
    agent = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))

    data = agent.to_dict()

    assert data["persistent_store"]["enabled"] is True
    assert data["persistent_store"]["backend"]["memory_store_id"] == "ms-123"
    assert "secret-token" not in json.dumps(data, default=str)


def test_reading_a_persistent_file_does_not_write_a_cache_back(llm):
    """FileReadTool caches extracted text beside the original; the durable namespace must stay clean."""
    workspace, persistent = InMemoryFileStore(), DeclinesCacheStore()
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=workspace, agent_file_write_enabled=True),
        persistent_store=_persistent(persistent),
    )
    backend = agent.file_store_backend
    backend.store("memories/prefs.md", "remembered")
    backend.store("scratch.md", "ephemeral")

    read_tool = next(t for t in agent.tools if t.name == "file-read")
    read_tool.run(input_data={"file_path": "memories/prefs.md"})
    read_tool.run(input_data={"file_path": "scratch.md"})

    assert not persistent.exists("memories/prefs.md.extracted.txt")
    assert workspace.exists("scratch.md.extracted.txt")  # unchanged for ordinary workspace files


def test_an_explicitly_built_composite_is_not_wrapped_again(llm, persistent_backend):
    """A caller may compose the backend themselves and still declare `persistent_store` for the prompt."""
    workspace = InMemoryFileStore()
    composite = CompositeFileStore(default=workspace, routes={"memories/": persistent_backend})
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=composite, agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend),
    )

    assert agent.file_store_backend is composite, "The backend was re-wrapped in a second composite."
    assert not isinstance(composite.default, CompositeFileStore)
    assert "## Memory" in _ops_block(agent), "The prompt protocol must still be enabled."


def test_a_different_route_target_still_composes(llm, persistent_backend):
    """Only an identical route is treated as already composed."""
    other = InMemoryFileStore()
    composite = CompositeFileStore(default=InMemoryFileStore(), routes={"memories/": other})
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=composite, agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend),
    )

    assert agent.file_store_backend is not composite
    assert agent.file_store_backend.routes["memories/"] is persistent_backend

