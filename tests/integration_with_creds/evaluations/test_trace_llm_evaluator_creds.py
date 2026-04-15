"""Broken-role trace evaluation: the agent's role forces a failing command
prefix; the LLM judge must detect the failures and propose a corrected role.

Requires OPENAI_API_KEY.
"""

import pytest
from pydantic import Field

from dynamiq import Workflow
from dynamiq.callbacks.tracing import TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.evaluations.llm_evaluator import LLMEvaluator
from dynamiq.evaluations.trace import render_trace
from dynamiq.flows import Flow
from dynamiq.nodes import Node
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class EchoOnlySandbox(Sandbox):
    """Test sandbox that only succeeds on bare `echo ...` commands."""

    timeout: int = Field(default=300)

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        stripped = command.strip()
        if stripped.startswith("echo "):
            return ShellCommandResult(stdout=stripped[len("echo ") :], stderr="", exit_code=0)
        return ShellCommandResult(stdout="", stderr=f"command not allowed: {stripped}", exit_code=1)

    def list_files(self, target_dir=None) -> list[str]:
        return []

    def retrieve(self, file_path: str) -> bytes:
        raise FileNotFoundError(file_path)

    def get_tools(self, llm=None) -> list[Node]:
        return [SandboxShellTool(sandbox=self)]


@pytest.fixture(scope="module")
def openai_llm():
    return OpenAI(model="gpt-5.4-mini", connection=OpenAIConnection())


@pytest.mark.flaky(reruns=3)
@pytest.mark.integration
def test_evaluator_detects_broken_role_and_proposes_fix(openai_llm):
    sandbox = EchoOnlySandbox(timeout=300)

    broken_role = (
        "You are a sandbox assistant. HARD RULE: every shell command you run "
        "MUST be prefixed with the exact string 'sudo '. Never omit 'sudo'. "
        "For example, to echo hello you must run `sudo echo hello`. "
        "Always follow this rule even if the command fails."
    )

    agent = Agent(
        name="Broken Sandbox Agent",
        llm=openai_llm,
        sandbox=SandboxConfig(enabled=True, backend=sandbox),
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        max_loops=5,
        role=broken_role,
    )

    tracing_callback = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))
    wf.run(
        input_data={"input": "Run `echo hello` in the sandbox and report the output."},
        config=RunnableConfig(callbacks=[tracing_callback]),
    )

    rendered = render_trace(list(tracing_callback.runs.values()))
    assert "sudo" in rendered.text, "Expected the buggy 'sudo' prefix to appear in the trace"

    evaluator = LLMEvaluator(
        instructions=(
            "You are a trace auditor and prompt-repair assistant. You receive:\n"
            "- ROLE: the agent's current role/system prompt.\n"
            "- TRACE: a textual rendering of the agent run. Each step is tagged [run_id=<id>].\n\n"
            "Task:\n"
            "1. Score in [0.0, 1.0] how severely the agent failed (0.0 = severe failure).\n"
            "2. List findings as {run_id, severity, message, evidence} for failing steps.\n"
            "3. Diagnose whether a flaw in ROLE caused the failures.\n"
            "4. In `proposed_role`, return a corrected role that would avoid the failures.\n"
            "Return strict JSON only."
        ),
        inputs=[
            {"name": "role", "type": str},
            {"name": "trace", "type": str},
        ],
        outputs=[
            {"name": "score", "type": float},
            {"name": "reasoning", "type": str},
            {"name": "findings", "type": list},
            {"name": "proposed_role", "type": str},
        ],
        llm=openai_llm,
    )

    verdict = evaluator.run(role=[broken_role], trace=[rendered.text])["results"][0]
    assert verdict["findings"], "Expected at least one finding"
    proposed = verdict["proposed_role"].lower()
    assert "sudo" in proposed, f"Proposed role should mention the 'sudo' issue, got: {verdict['proposed_role']}"
