import json

import pytest
from pydantic import ValidationError

from dynamiq.connections import E2B, Dynamiq
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file import CompositeFileStore, DynamiqFileStore, InMemoryFileStore
from dynamiq.storages.file.base import FileStoreConfig, PersistentStoreConfig, memory_root


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
    """`path_prefix` names a namespace; the root is fixed, so it can only ever land under it."""
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend, path_prefix="knowledge"),
    )

    assert list(agent.file_store_backend.routes) == ["memories/knowledge/"]
    assert "memories/knowledge/" in _ops_block(agent)


@pytest.mark.parametrize(
    "supplied",
    ["knowledge", "knowledge/", "memories/knowledge", "memories/knowledge/", "/memories/knowledge/"],
)
def test_every_spelling_of_a_namespace_addresses_the_same_place(llm, persistent_backend, supplied):
    """The root is fixed and prepended, so a caller cannot land outside it or double it up."""
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=_persistent(persistent_backend, path_prefix=supplied),
    )

    assert list(agent.file_store_backend.routes) == ["memories/knowledge/"]


def test_a_namespace_cannot_escape_the_root(persistent_backend):
    with pytest.raises(ValidationError):
        _persistent(persistent_backend, path_prefix="../escape")


def test_memories_always_share_one_listable_root(llm, persistent_backend):
    """Sibling-looking namespaces still sit under the root, so one listing reaches them all."""
    other = InMemoryFileStore()
    agent = Agent(
        name="a",
        llm=llm,
        persistent_store=[
            _persistent(persistent_backend, path_prefix="user"),
            _persistent(other, path_prefix="company"),
        ],
    )

    assert memory_root(agent.persistent_stores) == "memories/"
    assert [c.normalized_prefix for c in agent.persistent_stores] == ["memories/user/", "memories/company/"]


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


def _two_memories():
    """Two memories with nothing in common but the root they hang off."""
    handbook, personal = InMemoryFileStore(), InMemoryFileStore()
    return (
        handbook,
        personal,
        [
            PersistentStoreConfig(
                enabled=True,
                backend=handbook,
                path_prefix="memories/handbook/",
                write_enabled=False,
                name="handbook",
                description="Team conventions and runbooks.",
            ),
            PersistentStoreConfig(
                enabled=True,
                backend=personal,
                path_prefix="memories/me/",
                name="user",
                description="What you learn about this specific user.",
            ),
        ],
    )


def test_several_memories_share_one_tool_set(llm):
    """The path picks the memory, so a second one costs no extra tools."""
    handbook, personal, namespaces = _two_memories()
    handbook.store("memories/handbook/deploys.md", b"Deploys are frozen on Fridays.")

    agent = Agent(name="a", llm=llm, persistent_store=namespaces)

    assert _tool_names(agent) == ["memory-read", "memory-list", "memory-write"]
    read, listing, write = (
        next(t for t in agent.tools if t.name == name) for name in ("memory-read", "memory-list", "memory-write")
    )

    assert "frozen on Fridays" in read.run(input_data={"file_path": "memories/handbook/deploys.md"}).output["content"]

    write.run(input_data={"action": "write", "file_path": "memories/me/style.md", "content": "3-line docstrings."})
    assert personal.exists("memories/me/style.md")
    assert not handbook.exists("memories/me/style.md"), "A write leaked into the wrong memory."

    # The agent is told to list the root first, so that one call has to reach every memory.
    everything = listing.run(input_data={"file_path": "memories/", "recursive": True}).output["content"]
    assert "memories/handbook/deploys.md" in everything and "memories/me/style.md" in everything


def test_several_memories_route_under_a_file_store(llm):
    """With a workspace the memories become routes on the composite the file tools already use."""
    workspace = InMemoryFileStore()
    handbook, personal, namespaces = _two_memories()
    agent = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=workspace, agent_file_write_enabled=True),
        persistent_store=namespaces,
    )

    backend = agent.file_store_backend
    assert isinstance(backend, CompositeFileStore)
    assert sorted(backend.routes) == ["memories/handbook/", "memories/me/"]
    assert all(tool.file_store is backend for tool in agent.tools)

    backend.store("memories/me/style.md", "3-line docstrings.")
    backend.store("scratch.md", "ephemeral")

    assert personal.retrieve("memories/me/style.md") == b"3-line docstrings."
    assert workspace.retrieve("scratch.md") == b"ephemeral"
    assert not workspace.exists("memories/me/style.md")
    assert not handbook.exists("memories/me/style.md")


def test_each_memory_is_described_to_the_model(llm):
    """Names and descriptions are the only way the model can tell one memory from another."""
    _, _, namespaces = _two_memories()
    agent = Agent(name="a", llm=llm, persistent_store=namespaces)

    ops = _ops_block(agent)
    write_description = next(t for t in agent.tools if t.name == "memory-write").description

    for text in (ops, write_description):
        assert "handbook: Team conventions and runbooks." in text
        assert "user: What you learn about this specific user." in text
        assert "memories/handbook/" in text and "memories/me/" in text

    assert "Read-only." in ops, "A memory that refuses writes must say so."
    assert "the later one wins" in ops, "Several memories need a stated precedence."
    # The protocol sends the agent to the directory holding both, not to either one of them.
    assert "`memory-list` memories/ BEFORE" in ops


def test_a_single_memory_is_unchanged(llm, persistent_backend):
    """An unnamed lone memory renders and binds exactly as it did before memories could be plural."""
    one = Agent(name="a", llm=llm, persistent_store=_persistent(persistent_backend))
    listed = Agent(name="a", llm=llm, persistent_store=[_persistent(persistent_backend)])

    assert _ops_block(one) == _ops_block(listed), "A list of one is the same agent."
    assert "Your memories" not in _ops_block(one)
    assert all(tool.file_store is persistent_backend for tool in one.tools), "A lone memory needs no composite."


def test_several_memories_survive_a_yaml_round_trip(llm, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    _, _, namespaces = _two_memories()
    namespaces[0].backend = DynamiqFileStore(
        connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"),
        memory_store_id="ms-handbook",
    )
    path = str(tmp_path / "wf.yaml")
    Workflow(flow=Flow(nodes=[Agent(name="a", llm=llm, persistent_store=namespaces)])).to_yaml_file(path)

    reloaded = Workflow.from_yaml_file(path, init_components=True).flow.nodes[0]

    assert [config.name for config in reloaded.persistent_stores] == ["handbook", "user"]
    assert [config.path_prefix for config in reloaded.persistent_stores] == ["handbook", "me"]
    assert [c.normalized_prefix for c in reloaded.persistent_stores] == ["memories/handbook/", "memories/me/"]
    assert reloaded.persistent_stores[0].description == "Team conventions and runbooks."
    assert reloaded.persistent_stores[0].write_enabled is False
    assert reloaded.persistent_stores[0].backend.memory_store_id == "ms-handbook"
    assert _tool_names(reloaded) == ["memory-read", "memory-list", "memory-write"]
