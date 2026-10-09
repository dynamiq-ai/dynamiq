import io
import json
import logging
from types import SimpleNamespace

import pytest
from litellm import ModelResponse
from litellm.utils import Delta

from dynamiq.artifacts import ArtifactConfig
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections import E2B, Dynamiq
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import _shared_sandbox_tools
from dynamiq.nodes.agents.shared_session import SharedSession, _shared_session
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from tests.unit.artifacts.conftest import FakeArtifactBackend


@pytest.fixture
def llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


@pytest.fixture
def store():
    return FakeArtifactBackend()


@pytest.fixture
def remote():
    return DynamiqArtifacts(connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"))


def _artifacts(backend):
    return ArtifactConfig(enabled=True, backend=backend)


def _files():
    return FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True)


def _artifact_tool(agent, user_id=None):
    tools = agent._build_artifact_tool(SimpleNamespace(user_id=user_id))
    return tools[0] if tools else None


def _blocks(agent):
    return agent.system_prompt_manager._prompt_blocks


def _ops(agent):
    return _blocks(agent).get("operational_instructions", "") or ""


def _sandbox():
    return SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-1"))


def test_the_tool_is_built_per_run_for_the_runs_end_user(llm, store):
    agent = Agent(name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store))

    first = _artifact_tool(agent, user_id="customer-a")
    second = _artifact_tool(agent, user_id="customer-b")

    assert first is not second
    assert first.backend is store
    assert (first.user_id, second.user_id) == ("customer-a", "customer-b")
    assert _artifact_tool(agent).user_id is None
    assert "artifact" not in [t.name for t in agent.tools]


def test_the_workspace_is_the_file_store_or_the_sandbox(llm, store):
    with_files = Agent(name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store))
    with_sandbox = Agent(name="b", llm=llm, sandbox=_sandbox(), artifacts=_artifacts(store))

    assert _artifact_tool(with_files).workspace is with_files.file_store_backend
    assert _artifact_tool(with_sandbox).workspace is with_sandbox.sandbox_backend


def test_without_a_workspace_the_tool_is_skipped_with_a_warning(llm, store, caplog):
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store))

    with caplog.at_level(logging.WARNING):
        assert _artifact_tool(agent) is None
    assert "## Artifacts" not in _ops(agent), "the prompt does not describe a tool the agent lacks"
    assert any("neither a sandbox nor a file store" in r.getMessage() for r in caplog.records)


def test_a_sub_agent_publishes_in_the_sandbox_it_borrows(llm, store, caplog):
    shared = E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-shared", base_path="/home/user")
    session_token = _shared_session.set(SharedSession(sandbox=shared, share_sandbox=True, owner_run_id="owner"))
    try:
        with caplog.at_level(logging.WARNING):
            sub = Agent(name="Writer", llm=llm, role="r", artifacts=_artifacts(store))
        assert not any("neither a sandbox nor a file store" in r.getMessage() for r in caplog.records)

        overlay = sub._maybe_borrow_shared_sandbox()  # what execute() does before building the tool
        overlay_token = _shared_sandbox_tools.set(overlay)
        try:
            tool = _artifact_tool(sub)
            sub._sync_react_prompt_for_shared_sandbox()
            assert tool.workspace is sub.sandbox_backend, "the tool works in the borrowed sandbox"
            assert tool.workspace.sandbox_id == "sbx-shared"
            assert "## Artifacts" in _ops(sub)
        finally:
            _shared_sandbox_tools.reset(overlay_token)
        sub._release_shared_sandbox_view()
    finally:
        _shared_session.reset(session_token)

    sub._sync_react_prompt_for_shared_sandbox()
    assert _artifact_tool(sub) is None
    assert "## Artifacts" not in _ops(sub)


def test_one_end_user_cannot_reach_anothers_artifacts(llm, store, mocker):
    _replies(
        mocker,
        _file_write({"action": "write", "file_path": "plan.md", "content": "# A's plan"}),
        _action({"action": "create", "path": "plan.md", "name": "A's plan"}),
        "Thought: Done.\nAnswer: Published.",
        _action({"action": "list"}),
        _action({"action": "get", "artifact_id": "a1"}),
        "Thought: Nothing of theirs.\nAnswer: Nothing found.",
    )
    agent = Agent(
        name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT
    )

    agent.run({"input": "Publish my plan", "user_id": "customer-a"})
    result = agent.run({"input": "Show me the plans", "user_id": "customer-b"})

    assert result.status == RunnableStatus.SUCCESS
    assert store.get("a1")[0].user_id == "customer-a", "the artifact belongs to the run's end user"
    assert store.list(user_id="customer-b") == []
    observations = json.dumps(result.output, default=str) + json.dumps(
        [m.content for m in agent._prompt.messages], default=str
    )
    assert "A's plan" not in observations, "B's list and get must not reveal A's artifact"
    assert "not found" in observations.lower()


def test_the_prompt_block_says_when_to_use_an_artifact(llm, store):
    agent = Agent(name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store))

    ops = _ops(agent)
    assert "## Artifacts" in ops
    assert "`artifact` tool" in ops
    assert "Use output files instead" in ops
    assert "'get' the artifact (it is saved into your workspace)" in ops
    assert "then 'update' with that file and the artifact's 'artifact_id'" in ops
    assert "Never 'create' a changed copy" in ops
    assert "read it with 'get' before answering" in ops, "a reader agent must know published documents live here"
    assert "You cannot make a public link" in ops, "the agent points the user at sharing instead of improvising"
    assert "'share'" not in ops


def test_the_sandbox_rule_points_at_artifacts(llm, store):
    agent = Agent(name="a", llm=llm, sandbox=_sandbox(), artifacts=_artifacts(store))

    env = _blocks(agent)["environment"]
    assert "or as artifacts for renderable deliverables" in env
    assert "Always return requested files as output files so the user" not in env


@pytest.mark.parametrize("mode", list(InferenceMode))
def test_an_agent_without_artifacts_is_byte_identical(llm, store, mode):
    kwargs = {"name": "a", "llm": llm, "sandbox": _sandbox(), "inference_mode": mode}
    plain = Agent(**kwargs)
    disabled = Agent(**kwargs, artifacts=ArtifactConfig(enabled=False, backend=store))

    assert _artifact_tool(disabled) is None
    assert _blocks(plain) == _blocks(disabled)
    assert "## Artifacts" not in json.dumps(_blocks(plain))


def test_an_artifact_only_agent_is_told_it_has_tools(llm, store):
    agent = Agent(
        name="a", llm=llm, tools=[], file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.XML
    )

    assert "{{ tool_description }}" in _blocks(agent).get("tools", "")


def test_the_tool_is_not_serialized_and_credentials_are_hidden(llm, remote):
    agent = Agent(name="a", llm=llm, file_store=_files(), artifacts=_artifacts(remote))

    data = agent.to_dict()

    assert "artifact" not in [tool["name"] for tool in data["tools"]], "built per run, never serialized"
    assert data["artifacts"]["enabled"] is True
    assert data["artifacts"]["backend"]["type"] == "dynamiq.artifacts.backends.Dynamiq"
    assert "secret-token" not in json.dumps(data, default=str)


def test_yaml_round_trip(llm, remote, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    remote = DynamiqArtifacts(connection=remote.connection, artifact_store_id="s1", user_id="customer-42")
    agent = Agent(name="a", llm=llm, sandbox=_sandbox(), artifacts=_artifacts(remote))
    path = str(tmp_path / "wf.yaml")
    Workflow(flow=Flow(nodes=[agent])).to_yaml_file(path)

    reloaded = Workflow.from_yaml_file(path, init_components=True).flow.nodes[0]

    assert isinstance(reloaded.artifacts.backend, DynamiqArtifacts)
    assert (reloaded.artifacts.backend.artifact_store_id, reloaded.artifacts.backend.user_id) == ("s1", "customer-42")
    assert _artifact_tool(reloaded).name == "artifact"


class _StreamRecorder(BaseCallbackHandler):
    def __init__(self):
        self.chunks = []

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        self.chunks.append(chunk)


def _replies(mocker, *texts):
    """Script the LLM: one reply per call, streamed when the agent streams."""
    queue = list(texts)

    def respond(stream: bool = False, *args, **kwargs):
        text = queue.pop(0)
        if stream:
            chunk = ModelResponse(stream=True)
            chunk.choices[0].delta = Delta(role="assistant", content=text)
            return iter([chunk])
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = text
        return response

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=respond)


def _action(tool_input: dict) -> str:
    return f"Thought: I will publish it.\nAction: artifact\nAction Input: {json.dumps(tool_input)}"


def _file_write(tool_input: dict) -> str:
    return f"Thought: I will write the file.\nAction: file-write\nAction Input: {json.dumps(tool_input)}"


def _file_read(tool_input: dict) -> str:
    return f"Thought: I will read the file.\nAction: file-read\nAction Input: {json.dumps(tool_input)}"


def _csv_upload():
    upload = io.BytesIO(b"region,total\nnorth,42\n")
    upload.name = "data.csv"
    return upload


def test_a_run_writes_files_and_returns_every_artifact_it_published(llm, store, mocker):
    completion = _replies(
        mocker,
        _file_write({"action": "write", "file_path": "report.html", "content": "<!doctype html><p>v1</p>"}),
        _action({"action": "create", "path": "report.html", "name": "Report"}),
        _file_write({"action": "write", "file_path": "notes.md", "content": "# Notes"}),
        _action({"action": "create", "path": "notes.md", "name": "Notes"}),
        _file_write({"action": "edit", "file_path": "report.html", "edits": [{"find": "v1", "replace": "v2"}]}),
        _action({"action": "update", "path": "report.html", "artifact_id": "a1"}),
        "Thought: Done.\nAnswer: Published the report and notes.",
    )
    recorder = _StreamRecorder()
    agent = Agent(
        name="a",
        llm=llm,
        file_store=_files(),
        artifacts=_artifacts(store),
        inference_mode=InferenceMode.DEFAULT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )

    result = agent.run({"input": "Make a report"}, config=RunnableConfig(callbacks=[recorder]))

    assert result.status == RunnableStatus.SUCCESS
    assert "file-write" in json.dumps(completion.call_args_list[0].kwargs["messages"], default=str)
    artifacts = result.output["artifacts"]
    assert [(a["id"], a["version_id"], a["version"]) for a in artifacts] == [("a1", "a1-v2", 2), ("a2", "a2-v1", 1)]
    assert json.dumps(artifacts), "refs only: JSON-serializable, no bytes"
    assert store.get("a1")[1] == "<!doctype html><p>v2</p>"

    tool_events = [
        c["choices"][0]["delta"]["content"]
        for c in recorder.chunks
        if c and c.get("choices") and c["choices"][0]["delta"].get("step") == "tool"
    ]
    artifact_events = [e for e in tool_events if e["tool"]["action_type"] == "artifact"]
    assert len(artifact_events) == 3
    assert artifact_events[-1]["output"]["artifact"]["version"] == 2


def test_an_upload_can_be_read_and_published(llm, store, mocker):
    completion = _replies(
        mocker,
        _file_read({"file_path": "data.csv"}),
        _action({"action": "create", "path": "data.csv", "name": "Data"}),
        "Thought: Done.\nAnswer: Published.",
    )
    agent = Agent(
        name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT
    )

    result = agent.run({"input": "Publish the attached data", "files": [_csv_upload()]})

    assert result.status == RunnableStatus.SUCCESS
    system_prompt = completion.call_args_list[0].kwargs["messages"][0]["content"]
    for name in ["file-read", "file-search", "file-write", "file-list"]:
        assert system_prompt.count(f"- {name}:") == 1, f"{name} is attached once"
    assert "north,42" in json.dumps(completion.call_args_list[1].kwargs["messages"], default=str)
    assert store.get("a1")[1] == "region,total\nnorth,42\n"


@pytest.mark.parametrize("other_tools", [False, True])
def test_an_upload_gives_a_workspace_less_agent_the_tool_and_its_instructions(llm, store, mocker, other_tools):
    from dynamiq.nodes.tools.python import Python

    completion = _replies(
        mocker,
        _action({"action": "create", "path": "data.csv", "name": "Data"}),
        "Thought: Done.\nAnswer: Published.",
    )
    tools = [Python(name="dummy", description="dummy tool", code="def run(inputs): return {}")] if other_tools else []
    agent = Agent(name="a", llm=llm, tools=tools, artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT)
    assert "## Artifacts" not in _ops(agent), "no workspace at init"

    result = agent.run({"input": "Publish the attached data", "files": [_csv_upload()]})

    assert result.status == RunnableStatus.SUCCESS
    system_prompt = completion.call_args_list[0].kwargs["messages"][0]["content"]
    assert "## Artifacts" in system_prompt
    assert "- artifact:" in system_prompt
    assert [a["id"] for a in result.output["artifacts"]] == ["a1"]
    assert store.get("a1")[1] == "region,total\nnorth,42\n"


def test_an_output_file_beside_artifacts_reaches_the_run_output(llm, store, mocker):
    _replies(
        mocker,
        _file_write({"action": "write", "file_path": "totals.csv", "content": "region,total\nnorth,42\n"}),
        "Thought: Done.\nOutput Files: totals.csv\nAnswer: Here are the totals.",
    )
    agent = Agent(
        name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT
    )

    result = agent.run({"input": "Give me the totals as a file"})

    assert result.status == RunnableStatus.SUCCESS
    assert [f.name for f in result.output["files"]] == ["totals.csv"]


def _new_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def _delegate(tool_input: dict) -> str:
    return f"Thought: The writer does this.\nAction: Writer\nAction Input: {json.dumps(tool_input)}"


@pytest.mark.parametrize("delegate_final", [False, True])
def test_a_sub_agents_artifacts_are_in_the_parents_output(llm, store, mocker, delegate_final):
    replies = [
        _delegate({"input": "Write and publish the Q3 notes", "delegate_final": delegate_final}),
        _file_write({"action": "write", "file_path": "notes.md", "content": "# Q3"}),
        _action({"action": "create", "path": "notes.md", "name": "Q3 notes"}),
        "Thought: Done.\nAnswer: Published the Q3 notes.",
    ]
    if not delegate_final:
        replies.append("Thought: The writer published it.\nAnswer: The Q3 notes are published.")
    _replies(mocker, *replies)
    writer = Agent(
        name="Writer",
        llm=_new_llm(),
        file_store=_files(),
        artifacts=_artifacts(store),
        inference_mode=InferenceMode.DEFAULT,
    )
    parent = Agent(
        name="Manager", llm=llm, tools=[writer], delegation_allowed=True, inference_mode=InferenceMode.DEFAULT
    )

    result = parent.run({"input": "Get the Q3 notes published"})

    assert result.status == RunnableStatus.SUCCESS
    assert [(a["id"], a["version"]) for a in result.output["artifacts"]] == [("a1", 1)]


def test_an_artifact_changed_by_parent_and_sub_agent_is_listed_once_at_its_latest_version(llm, store, mocker):
    _replies(
        mocker,
        _file_write({"action": "write", "file_path": "notes.md", "content": "# Q3"}),
        _action({"action": "create", "path": "notes.md", "name": "Q3 notes"}),
        _delegate({"input": "Add the Q4 forecast to artifact a1"}),
        _action({"action": "get", "artifact_id": "a1"}),
        _file_write({"action": "write", "file_path": "artifacts/a1/v1/notes.md", "content": "# Q3\n# Q4 forecast"}),
        _action({"action": "update", "path": "artifacts/a1/v1/notes.md", "artifact_id": "a1"}),
        "Thought: Done.\nAnswer: Added the forecast.",
        "Thought: Done.\nAnswer: The notes have the Q4 forecast.",
    )
    writer = Agent(
        name="Writer",
        llm=_new_llm(),
        file_store=_files(),
        artifacts=_artifacts(store),
        inference_mode=InferenceMode.DEFAULT,
    )
    parent = Agent(
        name="Manager",
        llm=llm,
        tools=[writer],
        file_store=_files(),
        artifacts=_artifacts(store),
        inference_mode=InferenceMode.DEFAULT,
    )

    result = parent.run({"input": "Publish the Q3 notes, then have the writer add Q4"})

    assert result.status == RunnableStatus.SUCCESS
    assert [(a["id"], a["version"]) for a in result.output["artifacts"]] == [("a1", 2)]


def test_repeated_gets_are_not_served_from_the_tool_cache(llm, store, mocker):
    _replies(
        mocker,
        _file_write({"action": "write", "file_path": "doc.md", "content": "first"}),
        _action({"action": "create", "path": "doc.md", "name": "Doc"}),
        _action({"action": "get", "artifact_id": "a1"}),
        _file_write(
            {"action": "edit", "file_path": "artifacts/a1/v1/doc.md", "edits": [{"find": "first", "replace": "second"}]}
        ),
        _action({"action": "update", "path": "artifacts/a1/v1/doc.md", "artifact_id": "a1"}),
        _action({"action": "get", "artifact_id": "a1"}),
        "Thought: Done.\nAnswer: ok",
    )
    agent = Agent(
        name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT
    )
    gets = mocker.spy(FakeArtifactBackend, "get")

    result = agent.run({"input": "go"})

    assert result.status == RunnableStatus.SUCCESS
    reads = [call for call in gets.call_args_list if call.kwargs.get("include_content", True)]
    assert len(reads) == 2, "the second identical 'get' must reach the store"
    assert result.output["artifacts"][0]["version"] == 2


def test_reading_an_artifact_does_not_make_it_a_deliverable(llm, store, mocker):
    existing = store.create(file_name="old.md", name="Old", kind="markdown", content="from yesterday")
    _replies(
        mocker,
        _action({"action": "get", "artifact_id": existing.id}),
        "Thought: Read it.\nAnswer: It says 'from yesterday'.",
    )
    agent = Agent(
        name="a", llm=llm, file_store=_files(), artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT
    )

    result = agent.run({"input": "What does the old note say?"})

    assert result.status == RunnableStatus.SUCCESS
    assert "artifacts" not in result.output
