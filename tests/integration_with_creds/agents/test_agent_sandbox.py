"""Integration tests for agent with sandbox (OPENAI_API_KEY required; E2B tests also need E2B_API_KEY)."""

import os

import pytest
from pydantic import Field

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class TestSandbox(Sandbox):
    """Minimal sandbox for tests; no real E2B VM. Subclasses Sandbox for Pydantic/typing."""

    timeout: int = Field(default=300, description="Timeout (compatibility with E2B-style backends).")

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        if command.strip().startswith("echo"):
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


@pytest.fixture(scope="module")
def e2b_connection():
    """E2B connection from E2B_API_KEY; skip if not set."""
    pytest.importorskip("e2b_desktop")
    from dynamiq.connections import E2B

    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        pytest.skip("E2B_API_KEY is not set; skipping E2B sandbox test.")
    return E2B(api_key=api_key)


@pytest.mark.integration
def test_agent_with_sandbox_executes_shell(openai_llm):
    """Agent with sandbox runs a simple shell command via sandbox tools; LLM required."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")

    sandbox = TestSandbox(timeout=300)

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


@pytest.mark.integration
def test_agent_with_e2b_sandbox_executes_shell(openai_llm, e2b_connection):
    """Agent with E2B sandbox runs a simple shell command via sandbox tools; OPENAI_API_KEY and E2B_API_KEY required."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; skipping credentials-required test.")

    from dynamiq.sandboxes.e2b import E2BSandbox

    sandbox = E2BSandbox(connection=e2b_connection, timeout=300)
    try:
        agent = Agent(
            name="E2B Sandbox Agent",
            llm=openai_llm,
            sandbox=SandboxConfig(enabled=True, backend=sandbox),
            inference_mode=InferenceMode.XML,
            max_loops=5,
            role="You are a helpful assistant that can run commands in the sandbox.",
        )

        wf = Workflow(flow=Flow(nodes=[agent]))

        result = wf.run(
            input_data={"input": "Run this command in the sandbox and tell me the output: echo hello"},
            config=RunnableConfig(),
        )
        assert result.status == RunnableStatus.SUCCESS
        content = result.output.get(agent.id, {}).get("output", {}).get("content", "")
        assert content is not None
        assert "hello" in str(content), f"Expected 'hello' in E2B output, got: {content[:500]}"
    finally:
        sandbox.close(kill=True)
