"""An agent records a user's preference in its memory store and applies it later (OPENAI_API_KEY required).

``MemoryStoreAPISimulator`` stands in for the server, so everything above the socket is real. The tests of
the store's HTTP contract alone, which need no model, are in tests/integration/memory_store.
"""


import pytest

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.storages.memory import CompositeMemoryStore, DynamiqMemoryStore, MemoryStoreConfig
from tests.helpers.memory_store_api import MemoryStoreAPISimulator

USER_MARKER = "# owner: qplum"
CODE_TASK = (
    "Write me a merge sort in Python and save it to sort.py. "
    f"One thing about my code style: every file starts with the comment `{USER_MARKER}`."
)
# A second, unrelated deliverable. It names neither convention: both have to come from memory.
SECOND_CODE_TASK = "Write me a binary search in Python and save it to search.py."

ROLE = "You are personal agent."

# Two separate stores. The composite refuses to mount one store twice: the prefix is stripped
# before the store is called, so two routes over one store would write to the same keys.
MEMORY_STORE_ID = "ms-user"
TEAM_MEMORY_STORE_ID = "ms-team"
USER_ID = "u-test"

MODEL = "gpt-5.4"

TEAM_CONVENTION = "Every function name must start with the prefix `zx_`, e.g. `zx_parse_config`."


@pytest.fixture(scope="module")
def openai_llm():
    # Recording a fact nobody asked to have recorded is the hard half of the protocol, and it tracks
    # model strength: a small model does it only once the role tells it to keep a memory at all.
    return OpenAI(model=MODEL, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig()


def _store(store_id: str, description: str) -> DynamiqMemoryStore:
    return DynamiqMemoryStore(
        connection=DynamiqConnection(url="https://memory-store.simulated", api_key="simulated-key"),
        memory_store_id=store_id,
        description=description,
    )


@pytest.fixture
def memory_store(monkeypatch):
    """The agent's memory, served by the simulator for the duration of one test."""
    simulator = MemoryStoreAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    return _store(MEMORY_STORE_ID, "What you learn about this user.")


def _stored_text(store):
    # The store carries no default tenant, so every call outside an agent run names one itself.
    paths = [entry.path for entry in store.list(user_id=USER_ID)]
    assert paths, "Agent recorded nothing."
    return paths, "".join(store.read(path, user_id=USER_ID) for path in paths)


def _assert_memory_holds_only_the_preference(store):
    """The standing preference belongs in memory; the merge sort is work and does not."""
    paths, stored = _stored_text(store)
    # Loose on the way in - the agent may paraphrase when recording. Applying it is what is exact.
    assert "owner" in stored.lower(), f"Preference not recorded. Store holds: {stored[:500]}"
    assert "def " not in stored, f"Code leaked into memory: {paths}"


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_memories_are_recorded_then_applied(openai_llm, run_config, memory_store):
    """Two agents over two memories: one records what it learns, the next acts on it.

    Nothing in either request mentions memory. The first agent is told a preference in passing
    while doing real work and must record it - in the user's memory, not the team's. The second
    gets an unrelated task in a new conversation and must apply both that preference and the team
    convention it was never shown. Only the per-store `description` tells the two memories apart.
    """
    personal = memory_store
    team = _store(TEAM_MEMORY_STORE_ID, "Conventions the whole team follows. Shared and curated elsewhere.")
    team.write("naming.md", TEAM_CONVENTION, user_id=USER_ID)

    backend = CompositeMemoryStore(routes={"team/": team, "me/": personal})

    def build_agent(name):
        return Agent(
            name=name,
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=8,
            # A workspace for the deliverable; memory is independent of it and has its own tool.
            file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
            memory_store=MemoryStoreConfig(enabled=True, backend=backend),
        )

    build_agent("MemoryWriter").run(input_data={"input": CODE_TASK, "user_id": USER_ID}, config=run_config)
    _assert_memory_holds_only_the_preference(personal)

    # The team memory is curated elsewhere: nothing the user volunteers belongs in it.
    assert [entry.path for entry in team.list(user_id=USER_ID)] == [
        "naming.md"
    ], "The shared team memory was written to."
    assert team.read("naming.md", user_id=USER_ID) == TEAM_CONVENTION

    # A brand-new agent and conversation; the only thing carried over is the store.
    applier = build_agent("MemoryApplier")
    applier.run(input_data={"input": SECOND_CODE_TASK, "user_id": USER_ID}, config=run_config)

    workspace = "\n".join(
        applier.file_store_backend.retrieve(info.path).decode(errors="replace")
        for info in applier.file_store_backend.list_files(recursive=True)
    )
    assert "zx_" in workspace, f"Team convention was not applied. Workspace holds: {workspace[:500]}"
    assert USER_MARKER in workspace, f"User preference was not applied. Workspace holds: {workspace[:500]}"
