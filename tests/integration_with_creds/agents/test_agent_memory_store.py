"""Integration tests for the agent memory store (OPENAI_API_KEY required).

``MemoryStoreAPISimulator`` stands in for the server, so everything above the socket is real: the
same client, URLs, query strings and status codes. Each test gets a fresh one.
"""

import json as json_lib
from datetime import datetime, timezone

import pytest

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.storages.memory import (
    CompositeMemoryStore,
    DynamiqMemoryStore,
    MemoryNotFoundError,
    MemoryStoreConfig,
    MemoryStoreError,
)

# Arbitrary on purpose: a model writes docstrings unaided, so only this proves memory reached it.
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


class _Response:
    """The subset of a `requests` response that DynamiqFileStore reads."""

    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json_lib.dumps(payload) if payload is not None else ""
        self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


class MemoryStoreAPISimulator:
    """In-process implementation of the Memory Store API (docs/memory_store_api.md).

    Stands in for the server so the agent tests can run before it exists. Deliberately strict — it
    enforces the parts a real implementation is most likely to skip, so a client change that
    violates the spec fails here rather than in production.
    """

    def __init__(self):
        # Keyed by (store_id, user_id, path) - the three things the real API scopes by. Dropping
        # either of the first two would let genuinely separate memories share one keyspace, which
        # is exactly the collision the composite exists to prevent.
        self.memories: dict[tuple[str, str, str], str] = {}
        self.calls: list[tuple[str, str]] = []

    # -- transport ---------------------------------------------------------
    def request(self, verb, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append((verb, url))

        if not (headers or {}).get("Authorization"):
            return _Response(401, {"message": "missing bearer token"})

        params, body = params or {}, json or {}
        source = body if verb == "PUT" else params

        # Required on every call: the client must never omit it.
        if not source.get("user_id"):
            return _Response(400, {"message": "user_id is required"})

        path = source.get("path")
        if path is not None and not self._path_is_safe(path):
            return _Response(400, {"message": f"illegal path: {path!r}"})

        scope = (url.split("/memory-stores/")[1].split("/")[0], source["user_id"])
        if url.endswith("/files/content"):
            return self._read(scope, path)
        if verb == "PUT":
            return self._write(scope, body)
        if verb == "DELETE":
            return self._delete(scope, path)
        return self._list(scope, params.get("path") or "")

    # -- endpoints ---------------------------------------------------------
    def _read(self, scope, path):
        if (*scope, path) not in self.memories:
            return _Response(404, {"message": f"no such memory: {path}"})
        return _Response(200, {"data": self._as_json(scope, path, include_content=True)})

    def _write(self, scope, body):
        path, content = body["path"], body["content"]
        if len(content) > 1_000_000:
            return _Response(413, {"message": "memory too large"})
        self.memories[(*scope, path)] = content  # upsert: no overwrite flag, no 409
        return _Response(200, {"data": self._as_json(scope, path)})

    def _delete(self, scope, path):
        if (*scope, path) not in self.memories:
            return _Response(404, {"message": f"no such memory: {path}"})
        del self.memories[(*scope, path)]
        return _Response(200, {"data": {"deleted": True}})

    def _list(self, scope, prefix):
        return _Response(
            200,
            {
                "data": [
                    self._as_json(scope, path)
                    for store_id, user_id, path in self.memories
                    if (store_id, user_id) == tuple(scope) and path.startswith(prefix)
                ]
            },
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _path_is_safe(path: str) -> bool:
        if not path:
            return True
        if path.startswith("/") or ".." in path.split("/"):
            return False
        return "%2e" not in path.lower()

    def _as_json(self, scope, path, include_content=False):
        payload = {
            "path": path,
            "size": len(self.memories[(*scope, path)]),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if include_content:
            payload["content"] = self.memories[(*scope, path)]
        return payload


@pytest.fixture(scope="module")
def openai_llm():
    # Recording a fact nobody asked to have recorded is the hard half of the protocol, and it tracks
    # model strength: a small model does it only once the role tells it to keep a memory at all.
    return OpenAI(model=MODEL, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=180)


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
    paths = [entry.path for entry in store.list()]
    assert paths, "Agent recorded nothing."
    return paths, "".join(store.read(path) for path in paths)


def _assert_memory_holds_only_the_preference(store):
    """The standing preference belongs in memory; the merge sort is work and does not."""
    paths, stored = _stored_text(store)
    # Loose on the way in - the agent may paraphrase when recording. Applying it is what is exact.
    assert "owner" in stored.lower(), f"Preference not recorded. Store holds: {stored[:500]}"
    assert "def " not in stored, f"Code leaked into memory: {paths}"


@pytest.mark.integration
def test_store_crud_against_the_api(memory_store):
    """Every endpoint of the contract, without a model in the loop.

    Run this first when bringing a server up: it isolates the HTTP contract from agent behaviour,
    so a failure here is the server's, not the model's.
    """
    store = memory_store
    path = "probe.md"

    entry = store.write(path, "probe body")
    assert entry.path == path, f"Server returned a different path: {entry.path}"

    assert store.read(path) == "probe body"
    assert path in [e.path for e in store.list()]

    store.write(path, "updated body")  # upsert: no overwrite flag, no 409
    assert store.read(path) == "updated body"

    assert store.delete(path) is True
    assert store.delete(path) is False, "Deleting a missing memory must return False, not raise."


@pytest.mark.integration
def test_store_error_branches(memory_store):
    """The parts of the contract a server is most likely to leave out."""
    store = memory_store

    with pytest.raises(MemoryNotFoundError):
        store.read("never-written.md")

    assert store.list("empty/") == []

    # Mounted behind a prefix, a path outside every route is refused by name.
    with pytest.raises(MemoryStoreError) as excinfo:
        CompositeMemoryStore(routes={"user/": store}).write("outside.md", "x")
    assert "user/" in str(excinfo.value)


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
    team.write("naming.md", TEAM_CONVENTION)

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
    assert [entry.path for entry in team.list()] == ["naming.md"], "The shared team memory was written to."
    assert team.read("naming.md") == TEAM_CONVENTION

    # A brand-new agent and conversation; the only thing carried over is the store.
    applier = build_agent("MemoryApplier")
    applier.run(input_data={"input": SECOND_CODE_TASK, "user_id": USER_ID}, config=run_config)

    workspace = "\n".join(
        applier.file_store_backend.retrieve(info.path).decode(errors="replace")
        for info in applier.file_store_backend.list_files(recursive=True)
    )
    assert "zx_" in workspace, f"Team convention was not applied. Workspace holds: {workspace[:500]}"
    assert USER_MARKER in workspace, f"User preference was not applied. Workspace holds: {workspace[:500]}"
