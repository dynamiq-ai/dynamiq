"""End-to-end tests for the agent memory store (OPENAI_API_KEY required).

They check the two things unit tests cannot:

- the HTTP path works end to end, and
- a real model understands the protocol and uses it *unprompted* — no request here tells an agent
  to save or look up anything. Facts are volunteered in passing while the user asks for something
  else, and the follow-up is an ordinary question in a new conversation.

Observed behaviour worth knowing when these fail: the *read* half of the protocol is reliable — an
agent lists its memory unprompted. The *write* half only fires once the agent is already doing tool
work; asked something it can answer in a single turn, it answers and records nothing. So each writer
task below involves real work, which is also how an agent with memory is actually used.

The HTTP layer is an in-process implementation of docs/memory_store_api.md
(``MemoryStoreAPISimulator`` below), so these tests are about the agent rather than the server.
Everything above the socket is real: the same ``DynamiqMemoryStore``, the same URLs, query strings,
JSON bodies and status codes. Each test gets a fresh simulator, so runs never see each other.

``test_store_crud_against_the_api`` and ``test_store_error_branches`` are the exception — they pin
the client against that contract with no model in the loop, and are what to run first when a real
server appears.
"""

import json as json_lib
import os
from datetime import datetime, timezone

import pytest

from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.storages.memory import (
    CompositeMemoryStore,
    DynamiqMemoryStore,
    MemoryNotFoundError,
    MemoryStoreConfig,
    MemoryStoreError,
)

# One request carrying both kinds of thing: a one-off deliverable and a standing preference.
# Memory must take the preference and leave the code behind — that split is what these tests check.
# Nothing here asks the agent to remember anything.
CODE_TASK = (
    "Write me a merge sort in Python and save it to sort.py. "
    "One thing about my code style: every function gets a docstring, at most 3 lines."
)
STYLE_QUESTION = "How do I like my code written?"

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
        if url.endswith("/memories/content"):
            return self._read(scope, path)
        if verb == "PUT":
            return self._write(scope, body)
        if verb == "DELETE":
            return self._delete(scope, path)
        return self._list(scope, params.get("prefix") or "")

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
    # Recording a fact nobody asked to have recorded is the hard half of the protocol, and it
    # tracks model strength: gpt-4o-mini never does it, and gpt-4o does it only when the request
    # says "remember this" in so many words — with a sandbox attached it writes the code and
    # forgets the preference. These tests are about the feature, not about coaxing a weak model,
    # so they run on a current one. Override with MEMORY_TEST_MODEL to compare.
    return OpenAI(model=MODEL, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=180)


@pytest.fixture(scope="module")
def e2b_connection():
    return E2BConnection()


def _store(store_id: str, description: str) -> DynamiqMemoryStore:
    return DynamiqMemoryStore(
        connection=DynamiqConnection(url="https://memory-store.simulated", api_key="simulated-key"),
        memory_store_id=store_id,
        user_id=USER_ID,
        description=description,
    )


@pytest.fixture
def memory_store(monkeypatch):
    """The agent's memory, served by the simulator for the duration of one test."""
    simulator = MemoryStoreAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    return _store(MEMORY_STORE_ID, "What you learn about this user.")


def _answer(result):
    assert result.status == RunnableStatus.SUCCESS, f"Run failed: {result}"
    return str(result.output.get("content", ""))


def _stored_text(store):
    paths = [entry.path for entry in store.list()]
    assert paths, "Agent recorded nothing."
    return paths, "".join(store.read(path) for path in paths)


def _assert_memory_holds_only_the_preference(store):
    """The standing preference belongs in memory; the merge sort is work and does not."""
    paths, stored = _stored_text(store)
    assert "docstring" in stored.lower(), f"Preference not recorded. Store holds: {stored[:500]}"
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
def test_memories_survive_into_a_new_conversation(openai_llm, run_config, memory_store):
    """The preference lands in memory, the code does not, and a fresh agent recalls it."""
    store = memory_store

    def build_agent(name):
        return Agent(
            name=name,
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=6,
            # A workspace for the deliverable; memory is independent of it and has its own tool.
            file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
            memory_store=MemoryStoreConfig(enabled=True, backend=store),
        )

    build_agent("MemoryWriter").run(input_data={"input": CODE_TASK}, config=run_config)
    _assert_memory_holds_only_the_preference(store)

    # A brand-new agent and conversation; the only thing carried over is the store.
    reader = build_agent("MemoryReader")
    answer = _answer(reader.run(input_data={"input": STYLE_QUESTION}, config=run_config))
    assert "docstring" in answer.lower(), f"Agent failed to recall the preference. Answer: {answer[:500]}"


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_e2b_sandbox_keeps_memories_out_of_the_sandbox(openai_llm, run_config, e2b_connection, memory_store):
    """Same request, but the code goes to a real sandbox and only the preference to memory."""
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping credentials-required test.")

    from dynamiq.sandboxes.e2b import E2BSandbox

    store = memory_store
    sandbox = E2BSandbox(connection=e2b_connection, timeout=300)
    try:
        agent = Agent(
            name="SandboxWithMemory",
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=8,
            sandbox=SandboxConfig(enabled=True, backend=sandbox),
            memory_store=MemoryStoreConfig(enabled=True, backend=store),
        )
        tool_names = {t.name for t in agent.tools}
        assert "memory-store" in tool_names, tool_names
        assert "sandbox-shell" in tool_names, f"Sandbox tools were displaced: {tool_names}"

        agent.run(input_data={"input": CODE_TASK}, config=run_config)
        _assert_memory_holds_only_the_preference(store)

        # A fresh agent on the same store recalls it, with no sandbox involved.
        reader = Agent(
            name="MemoryReader",
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=6,
            memory_store=MemoryStoreConfig(enabled=True, backend=store),
        )
        answer = _answer(reader.run(input_data={"input": STYLE_QUESTION}, config=run_config))
        assert "docstring" in answer.lower(), f"Agent failed to recall the preference. Answer: {answer[:500]}"
    finally:
        sandbox.close(kill=True)


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_two_memories_are_told_apart(openai_llm, run_config, memory_store):
    """Two memories at once - the shared team one and the user's own.

    The task is the same as everywhere else, so nothing in it points at either memory. Telling them
    apart is the whole test: the team convention is *read* and applied to the code, while the
    preference the user volunteers is *written* to the user memory and not to the team's. Only the
    per-store `description` distinguishes them.
    """
    personal = memory_store
    team = _store(TEAM_MEMORY_STORE_ID, "Conventions the whole team follows. Shared and curated elsewhere.")
    team.write("naming.md", TEAM_CONVENTION)

    backend = CompositeMemoryStore(routes={"team/": team, "me/": personal})

    agent = Agent(
        name="TwoMemories",
        llm=openai_llm,
        role=ROLE,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        memory_store=MemoryStoreConfig(enabled=True, backend=backend),
    )

    agent.run(input_data={"input": CODE_TASK}, config=run_config)

    workspace = "\n".join(
        agent.file_store_backend.retrieve(info.path).decode(errors="replace")
        for info in agent.file_store_backend.list_files(recursive=True)
    )
    assert "zx_" in workspace, f"Team convention was not read or not applied. Workspace holds: {workspace[:500]}"

    # The preference belongs in the personal store, and nowhere else.
    user_paths = [entry.path for entry in personal.list()]
    assert user_paths, "Nothing recorded in the user memory."
    recorded = "".join(personal.read(path) for path in user_paths)
    assert "docstring" in recorded.lower(), f"Preference not recorded. User memory holds: {recorded[:500]}"

    # The shared team store is untouched: same one memory, same content.
    team_paths = [entry.path for entry in team.list()]
    assert team_paths == ["naming.md"], f"The shared team memory was written to: {team_paths}"
    assert team.read("naming.md") == TEAM_CONVENTION
    assert "docstring" not in team.read("naming.md").lower(), "The preference leaked into the team memory."
