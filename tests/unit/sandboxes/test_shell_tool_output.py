"""What the sandbox shell tool tells the model."""

from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.tools.shell import SandboxShellTool


class StderrSandbox(Sandbox):
    def get_tools(self, llm=None):
        return []

    def run_command_shell(self, command: str, timeout: int = 60, run_in_background_enabled: bool = False):
        return ShellCommandResult(stdout="trained 3 epochs\n", stderr="WARNING: CUDA not available\n", exit_code=0)


def test_a_successful_command_shows_the_model_its_stderr_too():
    # Warnings and logs written to stderr (Python logging, pip) were dropped when the exit code was 0.
    tool = SandboxShellTool(sandbox=StderrSandbox())

    result = tool.run(input_data={"command": "python train.py"})

    assert "trained 3 epochs" in result.output["content"]
    assert "WARNING: CUDA not available" in result.output["content"]


def test_the_shell_tool_does_not_promise_files_are_returned_without_being_named():
    assert "automatically collected" not in SandboxShellTool(sandbox=StderrSandbox()).description
