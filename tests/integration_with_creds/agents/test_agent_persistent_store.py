"""End-to-end tests for the persistent store (OPENAI_API_KEY required).

They check the two things unit tests cannot:

- the HTTP path works end to end, and
- a real model understands the protocol and uses it *unprompted* — no request here tells an agent
  to save or look up anything. Facts are volunteered in passing while the user asks for something
  else, and the follow-up is an ordinary question in a new conversation. Any memory use is the model
  acting on the persistent-store protocol in its system prompt.

Observed behaviour worth knowing when these fail: the *read* half of the protocol is reliable — an
agent lists the prefix unprompted in every mode. The *write* half only fires once the agent is
already doing tool work; asked something it can answer in a single turn, it answers and records
nothing. So each writer task below involves real work, which is also how an agent with a file store
or sandbox is actually used.

The fact each test plants is a token no model can know from pretraining, so a correct recall can
only have come from the store.

**Which store they run against** is decided by the environment, and the tests are identical either
way:

- ``DYNAMIQ_MEMORY_STORE_ID`` set → the live Memory Store API.
- unset → ``MemoryStoreAPISimulator`` below, an in-process implementation of
  docs/memory_store_api.md. Everything above the socket is real: the same ``DynamiqFileStore``, the
  same URLs, query strings, JSON bodies, base64 and status codes.

So ``test_store_crud_against_the_api`` doubles as a conformance suite — it pins the simulator today
and validates the real server the day it ships. The simulator is also the clearest statement of what
the endpoints must do; read it alongside the spec.

Each run works inside its own namespace (``it-<run id>``, resolving to ``memories/it-<run id>/``)
and deletes it afterwards, so
concurrent or repeated runs never collide and nothing is left behind in a shared store.
"""

import base64
import fnmatch
import json as json_lib
import os
import pathlib
import posixpath
import uuid
from datetime import datetime, timezone

import pytest

from dynamiq.callbacks.tracing import TracingCallbackHandler
from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.storages.file import (
    CompositeFileStore,
    DynamiqFileStore,
    FileStoreConfig,
    InMemoryFileStore,
    PersistentStoreConfig,
)
from dynamiq.storages.file.base import MEMORY_ROOT

# One request carrying both kinds of thing: a one-off deliverable and a standing preference.
# Memory must take the preference and leave the code behind — that split is what these tests check.
# Nothing here asks the agent to remember anything.
CODE_TASK = (
    "Write me a merge sort in Python and save it to sort.py. "
    "One thing about my code style: every function gets a docstring, at most 3 lines."
)
STYLE_QUESTION = "How do I like my code written?"

ROLE = "You are personal agent."


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

    Stands in for the server so the agent tests can run before it exists. It is deliberately strict
    — it enforces the parts of the contract a real implementation is most likely to skip — so a
    client change that violates the spec fails here rather than in production.
    """

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []

    # -- transport ---------------------------------------------------------
    def request(self, verb, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append((verb, url))

        if (headers or {}).get("Authorization", "") == "":
            return _Response(401, {"message": "missing bearer token"})

        params, body = params or {}, json or {}
        path = body.get("path") if verb == "PUT" else params.get("path")

        if path is not None and not self._path_is_safe(path):
            return _Response(400, {"message": f"illegal path: {path!r}"})

        if url.endswith("/files/content"):
            return self._read(path)
        if url.endswith("/files/exists"):
            return _Response(200, {"data": {"exists": path in self.files}})
        if verb == "PUT":
            return self._write(body)
        if verb == "DELETE":
            return self._delete(path)
        return self._list(params)

    # -- endpoints ---------------------------------------------------------
    def _read(self, path):
        if path not in self.files:
            return _Response(404, {"message": f"no such path: {path}"})
        record = self.files[path]
        return _Response(200, {"data": self._as_json(record, include_content=True)})

    def _write(self, body):
        path = body["path"]
        if path in self.files and not body.get("overwrite", False):
            return _Response(409, {"message": f"already exists: {path}"})

        content = base64.b64decode(body["content"])
        if len(content) > 5_000_000:
            return _Response(413, {"message": "file too large"})

        self.files[path] = {
            "path": path,
            "content": content,
            "content_type": body.get("content_type") or "application/octet-stream",
            "metadata": body.get("metadata") or {},
            "created_at": self.files.get(path, {}).get("created_at")
            or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        return _Response(200, {"data": self._as_json(self.files[path])})

    def _delete(self, path):
        if path not in self.files:
            return _Response(404, {"message": f"no such path: {path}"})
        del self.files[path]
        return _Response(200, {"data": {"deleted": True}})

    def _list(self, params):
        prefix = params.get("path") or ""
        recursive = self._as_bool(params.get("recursive"))
        pattern = params.get("pattern")

        matched = []
        for path, record in self.files.items():
            if prefix and not path.startswith(prefix):
                continue
            if not recursive and "/" in path[len(prefix) :].lstrip("/"):
                continue
            if pattern and not fnmatch.fnmatch(posixpath.basename(path), pattern):
                continue
            matched.append(self._as_json(record))
        return _Response(200, {"data": matched})

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _as_bool(value):
        """Accept the client's Python-stringified booleans as well as JSON ones."""
        return str(value).lower() == "true"

    @staticmethod
    def _path_is_safe(path: str) -> bool:
        if not path:
            return True
        if path.startswith("/") or ".." in path.split("/"):
            return False
        return "%2e" not in path.lower()

    @staticmethod
    def _as_json(record, include_content=False):
        payload = {
            "name": posixpath.basename(record["path"]),
            "path": record["path"],
            "size": len(record["content"]),
            "content_type": record["content_type"],
            "created_at": record["created_at"],
            "metadata": record["metadata"],
        }
        if include_content:
            payload["content"] = base64.b64encode(record["content"]).decode("ascii")
        return payload


@pytest.fixture(scope="module")
def openai_llm():
    # Recording a fact nobody asked to have recorded is the hard half of the protocol, and it
    # tracks model strength: gpt-4o-mini never does it, and gpt-4o does it only when the request
    # says "remember this" in so many words — with a sandbox attached it writes the code and
    # forgets the preference. These tests are about the feature, not about coaxing a weak model,
    # so they run on a current one. Override with MEMORY_TEST_MODEL to compare.
    model = os.getenv("MEMORY_TEST_MODEL", "gpt-5.4")
    return OpenAI(model=model, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=180)


@pytest.fixture(scope="module")
def e2b_connection():
    return E2BConnection()


@pytest.fixture
def persistent_store(monkeypatch):
    """A store scoped to a throwaway namespace: the live API when configured, else the simulator.

    Yields the namespace as a caller writes it — a plain name — alongside the path it resolves to,
    so the tests exercise the appending rather than restating it.
    """
    namespace = f"it-{uuid.uuid4().hex[:8]}"
    prefix = f"{MEMORY_ROOT}{namespace}/"
    store_id = os.getenv("DYNAMIQ_MEMORY_STORE_ID")
    user = os.getenv("DYNAMIQ_USER_ID", "integration-test-user")

    if store_id:
        store = DynamiqFileStore(connection=DynamiqConnection(), memory_store_id=store_id, user=user)
        yield store, namespace, prefix
        for info in store.list_files(directory=prefix, recursive=True):
            store.delete(info.path)
        return

    simulator = MemoryStoreAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    store = DynamiqFileStore(
        connection=DynamiqConnection(url="https://memory-store.simulated", api_key="simulated-key"),
        memory_store_id="ms-simulated",
        user=user,
    )
    yield store, namespace, prefix


TRACE_DIR = "/Users/mihajlobulesnij/Documents/work/dynamiq/trace/"


@pytest.fixture(scope="module")
def trace_file(request):
    """Collect a readable trace of every agent run in this module.

    Lands in TRACE_DIR — a fixed, guessable path, deliberately not ``tempfile.gettempdir()``,
    which on macOS is an unguessable /var/folders directory. Set MEMORY_TRACE_DIR to redirect it.
    The path is reported through the terminal reporter, so it shows without needing ``-s``.
    """
    path = pathlib.Path(TRACE_DIR) / "memory_traces.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    handle.write("# Persistent store — agent traces\n")
    try:
        yield handle
    finally:
        handle.close()

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"\nAgent traces written to {path}", bold=True)


@pytest.fixture
def traced_run(trace_file, run_config, persistent_store):
    """Run an agent and record what it did: tool calls, store changes, and its answer.

    Store changes are captured as a before/after diff of the prefix rather than as HTTP, so the
    trace reads the same whether the run went to the simulator or a live server.
    """
    store, namespace, prefix = persistent_store

    def snapshot():
        return {i.path: store.retrieve(i.path) for i in store.list_files(directory=prefix, recursive=True)}

    def _run(agent, prompt, label):
        before = snapshot()
        tracer = TracingCallbackHandler()
        result = agent.run(
            input_data={"input": prompt},
            config=RunnableConfig(callbacks=[tracer], request_timeout=getattr(run_config, "request_timeout", 180)),
        )

        lines = [f"\n## {label}", f"\n**User:** {prompt}\n"]
        ordered = sorted(tracer.runs.values(), key=lambda r: r.start_time)

        # One LLM turn opens a loop; the tool runs that follow it are that loop's work. Rendering
        # them together is the point of this trace: reasoning next to the calls it produced.
        steps: list[str] = []
        loop = 0
        for run in ordered:
            group = (run.metadata or {}).get("node", {}).get("group")
            if group == "llms":
                loop += 1
                steps.append(f"\n### Loop {loop}")
                decisions = _llm_decisions(run)
                if not decisions:
                    steps.append(f"\n_No tool call. Model said:_ {_llm_text(run)[:600]}")
                for name, thought, args in decisions:
                    # The turn that ends the loop picks `provide_final_answer` — its thought is the
                    # agent explaining why it is done, which is where a missing memory write shows up.
                    steps.append(f"\n**Reasoning** — {thought}")
                    steps.append(f"\n**Calls** `{name}`\n```json\n{_pretty(args, 1200)}\n```")
            elif group == "tools":
                data = {k: v for k, v in (run.input or {}).items() if k != "brief"}
                name = run.metadata["node"]["name"]
                steps.append(f"\n**Result of** `{name}`\n```\n{_tool_result(run)[:1200]}\n```")
                steps.append(f"_(input: {_pretty(data, 400)})_")
        lines += steps or ["- (no runs recorded)"]

        changed = {path: body for path, body in snapshot().items() if before.get(path) != body}
        lines.append("\n**Persistent store changes**")
        lines += [
            "- `{}`\n  ```\n  {}\n  ```".format(path, body.decode(errors="replace")[:300])
            for path, body in changed.items()
        ] or ["- (none)"]

        answer = str(result.output.get("content", "")) if result.output else f"RUN FAILED — {result.error}"
        lines.append(f"\n**Agent:** {answer[:600]}\n")
        trace_file.write("\n".join(lines) + "\n")
        trace_file.flush()
        return result

    return _run


def _llm_decisions(llm_run):
    """Return the (function, thought, arguments) triples an LLM turn chose.

    The reasoning has to come from here: the agent strips `thought` in
    `ToolCallArguments.to_action_input()` before invoking a tool, so it never reaches the tool run.
    `tool_calls` on an LLM run is a *list* of {"function": {"name", "arguments"}}, and in
    FUNCTION_CALLING mode `arguments` is the raw JSON *string* the model emitted — parse it, or
    every thought reads as "(no thought)".
    """
    decisions = []
    try:
        for call in (llm_run.output or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json_lib.loads(arguments, strict=False)
                except ValueError:
                    pass
            thought = arguments.get("thought") if isinstance(arguments, dict) else None
            rest = (
                {k: v for k, v in arguments.items() if k != "thought"} if isinstance(arguments, dict) else arguments
            )
            decisions.append((function.get("name") or "?", str(thought or "(no thought)"), rest))
    except Exception:  # reasoning is a nice-to-have, never a test failure
        pass
    return decisions


def _llm_text(llm_run):
    """The plain text of an LLM turn, for turns that called nothing."""
    try:
        content = (llm_run.output or {}).get("content")
        return str(content) if content else "(empty)"
    except Exception:
        return "(unavailable)"


def _tool_result(tool_run):
    """What a tool handed back — `content` when there is one, else the whole output."""
    output = tool_run.output
    if isinstance(output, dict) and "content" in output:
        output = output["content"]
    return str(output)


def _pretty(value, limit):
    try:
        text = json_lib.dumps(value, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "\n… (truncated)"


def _answer(result):
    assert result.status == RunnableStatus.SUCCESS, f"Run failed: {result}"
    return str(result.output.get("content", ""))


def _stored_text(store, prefix):
    paths = [info.path for info in store.list_files(directory=prefix, recursive=True)]
    assert paths, f"Agent recorded nothing under {prefix}."
    assert all(p.startswith(prefix) for p in paths), f"Wrote outside the namespace: {paths}"
    return paths, b"".join(store.retrieve(p) for p in paths).decode()


def _assert_memory_holds_only_the_preference(store, prefix):
    """The standing preference belongs in memory; the merge sort is work and does not."""
    paths, stored = _stored_text(store, prefix)
    assert "docstring" in stored.lower(), f"Preference not recorded. Store holds: {stored[:500]}"
    assert "def " not in stored, f"Code leaked into the persistent store: {paths}"


@pytest.mark.integration
def test_store_crud_against_the_api(persistent_store):
    """Every endpoint of the contract, without a model in the loop.

    Run this first when bringing a server up: it isolates the HTTP contract from agent behaviour,
    so a failure here is the server's, not the model's.
    """
    store, namespace, prefix = persistent_store
    path = f"{prefix}probe.md"

    info = store.store(path, "probe body", content_type="text/markdown", overwrite=True)
    assert info.path == path, f"Server returned a different path: {info.path}"

    assert store.exists(path) is True
    assert store.retrieve(path) == b"probe body"
    assert path in [f.path for f in store.list_files(directory=prefix, recursive=True)]

    store.store(path, "updated body", overwrite=True)
    assert store.retrieve(path) == b"updated body"

    assert store.delete(path) is True
    assert store.exists(path) is False
    assert store.delete(path) is False, "Deleting a missing path must return False, not raise."


@pytest.mark.integration
def test_store_error_branches(persistent_store):
    """The parts of the contract a server is most likely to leave out."""
    from dynamiq.storages.file.base import FileExistsError, FileNotFoundError

    store, namespace, prefix = persistent_store
    path = f"{prefix}guard.md"

    with pytest.raises(FileNotFoundError):
        store.retrieve(f"{prefix}never-written.md")

    store.store(path, "first", overwrite=True)
    with pytest.raises(FileExistsError):
        store.store(path, "second", overwrite=False)
    assert store.retrieve(path) == b"first", "A refused write must not modify the file."

    assert store.list_files(directory=f"{prefix}empty-dir/", recursive=True) == []


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_composite_memories_survive_into_a_new_conversation(openai_llm, run_config, persistent_store, traced_run):
    """Mode 1: the preference lands in memory, the code does not, and a fresh agent recalls it."""
    store, namespace, prefix = persistent_store

    def build_agent(name):
        return Agent(
            name=name,
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=6,
            # Composed explicitly so the test states the shape it exercises: an ephemeral workspace
            # with the persistent store routed under `prefix`. `persistent_store` is still declared
            # — it is what enables the prompt protocol — and composing is idempotent.
            file_store=FileStoreConfig(
                enabled=True,
                backend=CompositeFileStore(default=InMemoryFileStore(), routes={prefix: store}),
                agent_file_write_enabled=True,
            ),
            persistent_store=PersistentStoreConfig(enabled=True, backend=store, path_prefix=namespace),
        )

    traced_run(build_agent("PersistentWriter"), CODE_TASK, "Composite · conversation A · code plus a preference")
    _assert_memory_holds_only_the_preference(store, prefix)

    # A brand-new agent and conversation; the only thing carried over is the store.
    reader = build_agent("PersistentReader")
    answer = _answer(traced_run(reader, STYLE_QUESTION, "Composite · conversation B · never told the preference"))
    assert "docstring" in answer.lower(), f"Agent failed to recall the preference. Answer: {answer[:500]}"


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_e2b_sandbox_keeps_memories_out_of_the_sandbox(
    openai_llm, run_config, e2b_connection, persistent_store, traced_run
):
    """Mode 2: same request, but the code goes to a real sandbox and only the preference to memory."""
    if not os.getenv("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY is not set; skipping credentials-required test.")

    from dynamiq.sandboxes.e2b import E2BSandbox

    store, namespace, prefix = persistent_store
    sandbox = E2BSandbox(connection=e2b_connection, timeout=300)
    try:
        agent = Agent(
            name="SandboxWithMemory",
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=8,
            sandbox=SandboxConfig(enabled=True, backend=sandbox),
            persistent_store=PersistentStoreConfig(enabled=True, backend=store, path_prefix=namespace),
        )
        tool_names = {t.name for t in agent.tools}
        assert {"memory-read", "memory-list", "memory-write"} <= tool_names, tool_names
        assert "sandbox-shell" in tool_names, f"Sandbox tools were displaced: {tool_names}"

        traced_run(agent, CODE_TASK, "Sandbox · code written to the sandbox, preference mentioned")
        _assert_memory_holds_only_the_preference(store, prefix)

        # A fresh agent on the same store recalls it, with no sandbox involved.
        reader = Agent(
            name="MemoryReader",
            llm=openai_llm,
            role=ROLE,
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=6,
            persistent_store=PersistentStoreConfig(enabled=True, backend=store, path_prefix=namespace),
        )
        answer = _answer(traced_run(reader, STYLE_QUESTION, "Sandbox · new conversation, never told the preference"))
        assert "docstring" in answer.lower(), f"Agent failed to recall the preference. Answer: {answer[:500]}"
    finally:
        sandbox.close(kill=True)


# A naming rule no model can produce by chance, planted in the team memory and mentioned nowhere in
# the request. Code that follows it can only have come from the agent reading that memory.
TEAM_CONVENTION = "Every function name must start with the prefix `zx_`, e.g. `zx_parse_config`."


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_two_memories_are_told_apart(openai_llm, run_config, persistent_store, traced_run):
    """Mode 3: two memories at once — the shared team one and the user's own.

    The task is the same as everywhere else, so nothing in it points at either memory. Telling them
    apart is the whole test: the team convention is *read* and applied to the code, while the
    preference the user volunteers is *written* to the user memory and not to the team's. Only the
    per-memory `name` and `description` distinguish them.
    """
    store, run_namespace, run_prefix = persistent_store
    # Namespaces as a caller writes them; the fixed root turns each into its addressed path.
    team_namespace, user_namespace = f"{run_namespace}/team", f"{run_namespace}/me"
    team_prefix, user_prefix = f"{run_prefix}team/", f"{run_prefix}me/"
    store.store(f"{team_prefix}naming.md", TEAM_CONVENTION, content_type="text/markdown", overwrite=True)

    agent = Agent(
        name="TwoMemories",
        llm=openai_llm,
        role=ROLE,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        # A plain workspace for the deliverable; both memories are routed under it by the agent.
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        persistent_store=[
            PersistentStoreConfig(
                enabled=True,
                backend=store,
                path_prefix=team_namespace,
                write_enabled=False,
                name="team",
                description="Conventions the whole team follows. Shared and curated elsewhere.",
            ),
            PersistentStoreConfig(
                enabled=True,
                backend=store,
                path_prefix=user_namespace,
                name="user",
                description="What you learn about this specific user.",
            ),
        ],
    )

    traced_run(agent, CODE_TASK, "Two memories · team convention applied, user preference recorded")

    workspace = "\n".join(
        agent.file_store_backend.retrieve(info.path).decode(errors="replace")
        for info in agent.file_store_backend.list_files(recursive=True)
        if not info.path.startswith(run_prefix)
    )
    assert "zx_" in workspace, f"Team convention was not read or not applied. Workspace holds: {workspace[:500]}"

    user_paths = [info.path for info in store.list_files(directory=user_prefix, recursive=True)]
    assert user_paths, f"Nothing recorded in the user memory under {user_prefix}."
    recorded = b"".join(store.retrieve(path) for path in user_paths).decode()
    assert "docstring" in recorded.lower(), f"Preference not recorded. User memory holds: {recorded[:500]}"

    team_paths = [info.path for info in store.list_files(directory=team_prefix, recursive=True)]
    assert team_paths == [f"{team_prefix}naming.md"], f"The shared team memory was written to: {team_paths}"
    assert store.retrieve(f"{team_prefix}naming.md").decode() == TEAM_CONVENTION
