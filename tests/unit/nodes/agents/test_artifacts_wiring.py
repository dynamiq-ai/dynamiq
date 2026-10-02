import json

import pytest
from litellm import ModelResponse
from litellm.utils import Delta

from dynamiq.artifacts import ArtifactConfig
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections import E2B, Dynamiq
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
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


def _artifact_tool(agent):
    tools = agent._build_artifact_tool()
    return tools[0] if tools else None


def _blocks(agent):
    return agent.system_prompt_manager._prompt_blocks


def _ops(agent):
    return _blocks(agent).get("operational_instructions", "") or ""


def _sandbox():
    return SandboxConfig(enabled=True, backend=E2BSandbox(connection=E2B(api_key="t"), sandbox_id="sbx-1"))


def test_the_tool_is_built_per_run(llm, store):
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store))

    first = _artifact_tool(agent)
    second = _artifact_tool(agent)

    assert first is not second
    assert first.backend is store
    assert "artifact" not in [t.name for t in agent.tools]


def test_the_workspace_is_the_file_store_or_the_sandbox(llm, store):
    with_files = Agent(
        name="a",
        llm=llm,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore()),
        artifacts=_artifacts(store),
    )
    with_sandbox = Agent(name="b", llm=llm, sandbox=_sandbox(), artifacts=_artifacts(store))

    assert _artifact_tool(with_files).file_source is with_files.file_store_backend
    assert _artifact_tool(with_sandbox).file_source is with_sandbox.sandbox_backend


def test_disabled_config_attaches_nothing(llm, store):
    agent = Agent(name="a", llm=llm, artifacts=ArtifactConfig(enabled=False, backend=store))

    assert _artifact_tool(agent) is None
    assert "## Artifacts" not in _ops(agent)


def test_the_prompt_block_says_when_to_use_an_artifact(llm, store):
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store))

    ops = _ops(agent)
    assert "## Artifacts" in ops
    assert "`artifact` tool" in ops
    assert "Use output files instead" in ops
    assert "pass 'path'" in ops
    assert "read it with 'get' before answering" in ops, "a reader agent must know published documents live here"


def test_the_sandbox_rule_points_at_artifacts(llm, store):
    agent = Agent(name="a", llm=llm, sandbox=_sandbox(), artifacts=_artifacts(store))

    env = _blocks(agent)["environment"]
    assert "or as artifacts for renderable deliverables" in env
    assert "Always return requested files as output files so the user" not in env


@pytest.mark.parametrize("mode", list(InferenceMode))
def test_an_agent_without_artifacts_is_byte_identical(llm, store, mode):
    """The feature must be invisible unless configured: same prompt blocks, same schemas."""
    kwargs = {"name": "a", "llm": llm, "sandbox": _sandbox(), "inference_mode": mode}
    plain = Agent(**kwargs)
    disabled = Agent(**kwargs, artifacts=ArtifactConfig(enabled=False, backend=store))

    assert _blocks(plain) == _blocks(disabled)
    assert "## Artifacts" not in json.dumps(_blocks(plain))


def test_an_artifact_only_agent_is_told_it_has_tools(llm, store):
    agent = Agent(name="a", llm=llm, tools=[], artifacts=_artifacts(store), inference_mode=InferenceMode.XML)

    assert "{{ tool_description }}" in _blocks(agent).get("tools", "")


def test_the_tool_is_not_serialized_and_credentials_are_hidden(llm, remote):
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(remote))

    data = agent.to_dict()

    assert data["tools"] == []
    assert data["artifacts"]["enabled"] is True
    assert data["artifacts"]["backend"]["type"] == "dynamiq.artifacts.backends.Dynamiq"
    assert "secret-token" not in json.dumps(data, default=str)


def test_yaml_round_trip(llm, remote, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow

    remote = DynamiqArtifacts(connection=remote.connection, artifact_store_id="s1", user_id="customer-42")
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(remote))
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


def test_a_run_returns_every_artifact_it_touched(llm, store, mocker):
    _replies(
        mocker,
        _action({"action": "create", "name": "Report", "content": "<!doctype html><p>v1</p>"}),
        _action({"action": "create", "name": "Notes", "content": "# Notes"}),
        _action({"action": "update", "artifact_id": "a1", "edits": [{"find": "v1", "replace": "v2"}]}),
        "Thought: Done.\nAnswer: Published the report and notes.",
    )
    recorder = _StreamRecorder()
    agent = Agent(
        name="a",
        llm=llm,
        artifacts=_artifacts(store),
        inference_mode=InferenceMode.DEFAULT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )

    result = agent.run({"input": "Make a report"}, config=RunnableConfig(callbacks=[recorder]))

    assert result.status == RunnableStatus.SUCCESS
    artifacts = result.output["artifacts"]
    assert [(a["id"], a["version_id"], a["version"]) for a in artifacts] == [("a1", "a1-v2", 2), ("a2", "a2-v1", 1)]
    assert json.dumps(artifacts), "refs only: JSON-serializable, no bytes"

    tool_events = [
        c["choices"][0]["delta"]["content"]
        for c in recorder.chunks
        if c and c.get("choices") and c["choices"][0]["delta"].get("step") == "tool"
    ]
    assert len(tool_events) == 3
    assert all(e["tool"]["action_type"] == "artifact" for e in tool_events)
    assert tool_events[-1]["output"]["artifact"]["version"] == 2


def test_a_run_without_artifacts_has_no_artifacts_key(llm, store, mocker):
    _replies(mocker, "Thought: Nothing to publish.\nAnswer: Hi.")
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT)

    result = agent.run({"input": "Say hi"})

    assert result.status == RunnableStatus.SUCCESS
    assert "artifacts" not in result.output


def test_repeated_gets_are_not_served_from_the_tool_cache(llm, store, mocker):
    """The agent caches tool results by input; a cached 'get' would hide a newer version."""
    _replies(
        mocker,
        _action({"action": "create", "name": "Doc", "content": "first"}),
        _action({"action": "get", "artifact_id": "a1"}),
        _action({"action": "update", "artifact_id": "a1", "content": "second"}),
        _action({"action": "get", "artifact_id": "a1"}),
        "Thought: Done.\nAnswer: ok",
    )
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT)
    gets = mocker.spy(FakeArtifactBackend, "get")

    result = agent.run({"input": "go"})

    assert result.status == RunnableStatus.SUCCESS
    assert gets.call_count == 2, "the second identical 'get' must reach the store"
    assert result.output["artifacts"][0]["version"] == 2


def test_reading_an_artifact_does_not_make_it_a_deliverable(llm, store, mocker):
    existing = store.create(file_name="old.md", name="Old", kind="markdown", content="from yesterday")
    _replies(
        mocker,
        _action({"action": "get", "artifact_id": existing.id}),
        "Thought: Read it.\nAnswer: It says 'from yesterday'.",
    )
    agent = Agent(name="a", llm=llm, artifacts=_artifacts(store), inference_mode=InferenceMode.DEFAULT)

    result = agent.run({"input": "What does the old note say?"})

    assert result.status == RunnableStatus.SUCCESS
    assert "artifacts" not in result.output
