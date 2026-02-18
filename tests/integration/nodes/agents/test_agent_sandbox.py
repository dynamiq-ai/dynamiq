"""Integration tests for Agent with Sandbox (mocked sandbox and LLM)."""

from typing import ClassVar

import pytest
from pydantic import Field

from dynamiq import Workflow, connections
from dynamiq.connections import E2B
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.file_tools import FileReadTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class MockSandbox(Sandbox):
    """Mock sandbox implementation for tests; records commands and returns fixed output.
    When Map clones the node, each clone gets its own sandbox with its own commands_called.
    Set MockSandbox._shared_calls = [] in tests that need to see calls from all clones (e.g. Map).
    """

    commands_called: list[dict] = Field(default_factory=list, description="Record of run_command_shell calls.")
    mock_files: dict[str, bytes] = Field(
        default_factory=dict, description="Mock files available in the sandbox (path -> content)."
    )
    _shared_calls: ClassVar[list[dict] | None] = None  # Class-level: set in Map test to collect calls from clones

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        rec = {
            "command": command,
            "timeout": timeout,
            "run_in_background_enabled": run_in_background_enabled,
        }
        self.commands_called.append(rec)
        shared = MockSandbox._shared_calls
        if shared is not None:
            shared.append(rec)
        return ShellCommandResult(stdout="hello from sandbox", stderr="", exit_code=0)

    def list_files(self, target_dir=None) -> list[str]:
        return list(self.mock_files.keys())

    def retrieve(self, file_path: str) -> bytes:
        if file_path in self.mock_files:
            return self.mock_files[file_path]
        raise FileNotFoundError(f"Mock file not found: {file_path}")

    def get_tools(self, llm=None) -> list[Node]:
        return [SandboxShellTool(sandbox=self)]


@pytest.fixture
def mock_llm_sandbox_shell_response(mocker):
    """LLM returns: first call = use SandboxShellTool, second call = final answer."""
    from litellm import ModelResponse

    xml_tool_call = """<output>
  <thought>I will run a shell command in the sandbox.</thought>
  <action>SandboxShellTool</action>
  <action_input>{"command": "echo hello from sandbox"}</action_input>
</output>"""

    xml_final_answer = """<output>
  <thought>Command executed successfully.</thought>
  <answer>Shell command completed. Output: hello from sandbox</answer>
</output>"""

    call_count = [0]

    def response(stream: bool, *args, **kwargs):
        call_count[0] += 1
        content = xml_tool_call if call_count[0] == 1 else xml_final_answer
        model_r = ModelResponse()
        model_r["choices"][0]["message"]["content"] = content
        return model_r

    mock_llm = mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=response)
    yield mock_llm


@pytest.fixture
def mock_sandbox():
    """MockSandbox instance that records run_command_shell calls."""
    return MockSandbox()


def test_agent_with_sandbox_executes_shell_tool(mock_llm_sandbox_shell_response, mock_sandbox):
    """Agent with sandbox uses SandboxShellTool; MockSandbox.run_command_shell is called with expected command."""
    sandbox = mock_sandbox

    agent = Agent(
        name="Sandbox Agent",
        llm=OpenAI(
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="test-api-key"),
        ),
        inference_mode=InferenceMode.XML,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        max_loops=5,
    )

    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(
        input_data={"input": "Run a shell command that echoes hello."},
        config=RunnableConfig(),
    )

    assert result.status == RunnableStatus.SUCCESS
    content = result.output[agent.id].get("output")["content"]

    assert "hello from sandbox" in content

    # Use the sandbox instance attached to the tool (workflow may use a copy of the node)
    assert len(sandbox.commands_called) == 1
    assert sandbox.commands_called[0]["command"] == "echo hello from sandbox"
    assert sandbox.commands_called[0].get("timeout", 60) == 60


def test_map_agent_with_sandbox_executes_shell_tool(mock_llm_sandbox_shell_response, mock_sandbox):
    """Map over agent with sandbox: each item gets a shell tool call; MockSandbox records commands."""
    from dynamiq.nodes.operators import Map

    sandbox = mock_sandbox

    agent = Agent(
        name="Sandbox Agent",
        llm=OpenAI(
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="test-api-key"),
        ),
        inference_mode=InferenceMode.XML,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        max_loops=5,
    )

    map_node = Map(node=agent, max_workers=2)
    wf = Workflow(flow=Flow(nodes=[map_node]))

    result = wf.run(
        input_data={"input": [{"input": "input"}, {"input": "input"}]},
        config=RunnableConfig(),
    )

    assert result.status == RunnableStatus.SUCCESS


def test_agent_e2b_sandbox_yaml_roundtrip_no_duplicate_tools(tmp_path):
    """Roundtrip: to_yaml_file → from_yaml_file → to_yaml_file → from_yaml_file with init_components.

    Ensures sandbox tools (e.g. SandboxShellTool) are not duplicated when they were
    already present in the serialized tools list from a previous to_dict.
    """
    e2b_conn = E2B(id="e2b-conn", api_key="test-key")
    openai_conn = OpenAIConnection(id="openai-conn", api_key="test-key")

    backend = E2BSandbox(connection=e2b_conn)
    sandbox_config = SandboxConfig(enabled=True, backend=backend)

    agent = Agent(
        id="sandbox-agent",
        name="Sandbox Agent",
        llm=OpenAI(
            id="sandbox-agent-llm",
            connection=openai_conn,
            model="gpt-4o",
        ),
        summarization_config=SummarizationConfig(enabled=True),
        role="a helpful assistant that can execute shell commands in a sandbox.",
        sandbox=sandbox_config,
        max_loops=15,
    )

    workflow = Workflow(
        id="sandbox-workflow",
        flow=Flow(id="sandbox-flow", name="Sandbox Agent Flow", nodes=[agent]),
        version="1",
    )

    yaml_path = tmp_path / "agent_e2b_sandbox.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    assert len(loaded.flow.nodes) == 1
    loaded_agent = loaded.flow.nodes[0]
    assert isinstance(loaded_agent, Agent)
    assert loaded_agent.sandbox is not None
    assert loaded_agent.sandbox.enabled
    assert isinstance(loaded_agent.sandbox.backend, E2BSandbox)

    shell_tools_first = [t for t in loaded_agent.tools if isinstance(t, SandboxShellTool)]
    assert len(shell_tools_first) == 1, "First load should have exactly one SandboxShellTool"

    file_read_tools_first = [t for t in loaded_agent.tools if isinstance(t, FileReadTool)]
    assert len(file_read_tools_first) == 1, "First load should have exactly one FileReadTool"

    context_management_tools_first = [t for t in loaded_agent.tools if isinstance(t, ContextManagerTool)]
    assert len(context_management_tools_first) == 1, "First load should have exactly one ContextManagerTool"

    assert (
        file_read_tools_first[0].absolute_file_paths_allowed is True
    ), "Sandbox-backed FileReadTool must have absolute_file_paths_allowed=True"

    roundtrip_path = tmp_path / "agent_e2b_sandbox_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip = Workflow.from_yaml_file(str(roundtrip_path), init_components=True)

    roundtrip_agent = roundtrip.flow.nodes[0]
    assert isinstance(roundtrip_agent, Agent)

    shell_tools_roundtrip = [t for t in roundtrip_agent.tools if isinstance(t, SandboxShellTool)]
    assert len(shell_tools_roundtrip) == 1, (
        "After roundtrip, SandboxShellTool must not be duplicated: "
        "to_dict serializes tools including sandbox tools, then __init__ must not add them again."
    )

    file_read_tools_roundtrip = [t for t in roundtrip_agent.tools if isinstance(t, FileReadTool)]
    assert len(file_read_tools_roundtrip) == 1, (
        "After roundtrip, FileReadTool must not be duplicated: "
        "sandbox tools are excluded from serialization and recreated from sandbox config on load."
    )

    context_management_tools_roundtrip = [t for t in roundtrip_agent.tools if isinstance(t, ContextManagerTool)]
    assert len(context_management_tools_roundtrip) == 1, (
        "After roundtrip, ContextManagerTool must not be duplicated: "
        "sandbox tools are excluded from serialization and recreated from sandbox config on load."
    )
    assert file_read_tools_roundtrip[0].absolute_file_paths_allowed is True

    assert roundtrip_agent.sandbox is not None
    assert roundtrip_agent.sandbox.enabled
    assert isinstance(roundtrip_agent.sandbox.backend, E2BSandbox)


def test_agent_with_sandbox_returns_files(mock_llm_sandbox_shell_response):
    """Agent with sandbox collects and returns files from the output directory."""
    sandbox = MockSandbox(
        mock_files={
            "/home/user/output/result.txt": b"Hello, this is the result.",
            "/home/user/output/data.csv": b"col1,col2\n1,2\n3,4\n",
        }
    )

    agent = Agent(
        name="Sandbox Agent",
        llm=OpenAI(
            model="gpt-4o-mini",
            connection=connections.OpenAI(api_key="test-api-key"),
        ),
        inference_mode=InferenceMode.XML,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        max_loops=5,
    )

    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(
        input_data={"input": "Run a shell command that echoes hello."},
        config=RunnableConfig(),
    )

    assert result.status == RunnableStatus.SUCCESS

    output = result.output[agent.id]["output"]
    assert "hello from sandbox" in output["content"]

    # Verify files from sandbox output directory are returned
    files = output.get("files", [])
    assert len(files) == 2

    file_names = {f.name for f in files}
    assert file_names == {"result.txt", "data.csv"}

    # Verify file content is correct
    for f in files:
        f.seek(0)
        if f.name == "result.txt":
            assert f.read() == b"Hello, this is the result."
        elif f.name == "data.csv":
            assert f.read() == b"col1,col2\n1,2\n3,4\n"
