"""Integration tests for agent with sandbox"""

import os

import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import ShellCommandResult
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class TestSandbox:
    """Minimal sandbox for tests; no real E2B VM is created."""

    def __init__(self, connection=None, timeout: int = 300):
        self.connection = connection
        self.timeout = timeout

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        if command.strip().startswith("echo"):
            # "echo hello from SANDBOX" -> stdout "hello from SANDBOX"
            parts = command.strip().split(maxsplit=1)
            stdout = (parts[1] + " from SANDBOX") if len(parts) > 1 else " from SANDBOX"
            return ShellCommandResult(stdout=stdout, stderr="", exit_code=0)
        return ShellCommandResult(stdout="", stderr="Command not found", exit_code=1)

    def get_tools(self) -> list[Node]:
        return [SandboxShellTool(sandbox=self)]


@pytest.fixture(scope="module")
def openai_llm():
    return OpenAI(
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
    )


@pytest.mark.integration
def test_agent_with_sandbox_executes_shell(openai_llm):
    """Agent with sandbox runs a simple shell command via sandbox tools; LLM required."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")

    sandbox = TestSandbox(connection=None, timeout=300)

    agent = Agent(
        name="Sandbox Agent",
        llm=openai_llm,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        inference_mode=InferenceMode.XML,
        max_loops=5,
        role="You are helpful assistant that can run commands in the sandbox.",
    )

    wf = Workflow(flow=Flow(nodes=[agent]))

    result = wf.run(
        input_data={"input": "Run this command in the sandbox and tell me the output: echo hello"},
        config=RunnableConfig(),
    )
    assert result.status == RunnableStatus.SUCCESS
    content = result.output.get(agent.id, {}).get("output", {}).get("content", "")
    assert content is not None
    assert "hello from SANDBOX" in str(content), f"Expected 'hello from SANDBOX' in output, got: {content[:500]}"
