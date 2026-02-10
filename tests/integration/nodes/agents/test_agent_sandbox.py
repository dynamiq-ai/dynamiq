"""Integration tests for Agent with Sandbox (mocked sandbox and LLM)."""

import pytest
from pydantic import Field

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class MockSandbox(Sandbox):
    """Mock sandbox implementation for tests; records commands and returns fixed output.
    When Map clones the node, each clone gets its own sandbox with its own commands_called.
    Set MockSandbox._shared_calls = [] in tests that need to see calls from all clones (e.g. Map).
    """

    commands_called: list[dict] = Field(default_factory=list, description="Record of run_command_shell calls.")
    _shared_calls: list[dict] | None = None  # Class-level: set in Map test to collect calls from clones

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
        shared = getattr(MockSandbox, "_shared_calls", None)
        if shared is not None:
            shared.append(rec)
        return ShellCommandResult(stdout="hello from sandbox", stderr="", exit_code=0)

    def get_tools(self) -> list[Node]:
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
    """Map over agent with sandbox: each item gets a shell tool call; MockSandbox records commands.
    Each agent share sandbox instance.
    """
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
