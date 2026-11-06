"""Shell commands and package installs of the E2B and Daytona code interpreter tools."""

import os
import shutil
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import ERROR_OUTPUT_MAX_CHARS, CodeInterpreterInputSchema
from dynamiq.nodes.tools.daytona_sandbox import DaytonaInterpreterTool
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool


class CommandExit(Exception):
    """Shaped like the E2B SDK's CommandExitException, which wait() raises on a non-zero exit."""

    def __init__(self, exit_code: int, stdout: str = "", stderr: str = ""):
        super().__init__(f"exit code {exit_code}")
        self.exit_code, self.stdout, self.stderr = exit_code, stdout, stderr


def test_a_failing_e2b_tool_command_is_an_error_the_model_can_act_on():
    # `grep` with no match exits 1: that must reach the model, not end the run.
    handle = MagicMock()
    handle.wait.side_effect = CommandExit(1, stderr="")
    sandbox = MagicMock()
    sandbox.commands.run.return_value = handle
    tool = E2BInterpreterTool.model_construct(name="e2b", id="e2b")

    with pytest.raises(ToolExecutionException) as error:
        tool._execute_shell_command("grep -r TODO src", sandbox)

    assert error.value.recoverable
    assert "exit code 1" in str(error.value)


@pytest.mark.parametrize("tool_class", [E2BInterpreterTool, DaytonaInterpreterTool])
def test_a_failing_command_error_carries_only_the_end_of_a_long_log(tool_class):
    # The agent hands a recoverable error to the model whole: 780 KB of pytest log must not become one observation.
    log = "".join(f"PASSED tests/test_{i:06d}.py::test_case\n" for i in range(20000)) + "1 failed\n"
    sandbox = MagicMock()
    sandbox.commands.run.return_value.wait.side_effect = CommandExit(1, stdout=log)
    sandbox.process.exec.return_value = MagicMock(exit_code=1, result=log)
    tool = tool_class.model_construct(name="tool", id="tool")

    with pytest.raises(ToolExecutionException) as error:
        tool._execute_shell_command("pytest", sandbox)

    message = str(error.value)
    assert error.value.recoverable
    assert len(message) < ERROR_OUTPUT_MAX_CHARS + 200
    assert f"[{len(log) - ERROR_OUTPUT_MAX_CHARS:,} earlier characters cut]" in message
    assert message.endswith(log[-ERROR_OUTPUT_MAX_CHARS:])


def test_a_failing_e2b_tool_command_shows_short_streams_whole():
    handle = MagicMock()
    handle.wait.side_effect = CommandExit(2, stdout="collected 3 items\n", stderr="ERROR: file not found: tests/x.py\n")
    sandbox = MagicMock()
    sandbox.commands.run.return_value = handle
    tool = E2BInterpreterTool.model_construct(name="e2b", id="e2b")

    with pytest.raises(ToolExecutionException) as error:
        tool._execute_shell_command("pytest tests/x.py", sandbox)

    assert str(error.value) == (
        "Command failed with exit code 2. stderr: ERROR: file not found: tests/x.py\n stdout: collected 3 items\n"
    )


def pip_command(tool_class, packages: str) -> str | None:
    """The pip command a tool runs to install a package list, or None if it runs none."""
    sandbox = MagicMock()
    sandbox.commands.run.return_value = MagicMock(exit_code=0)
    sandbox.process.exec.return_value = MagicMock(exit_code=0)
    tool_class.model_construct(name="tool", id="tool")._install_packages(sandbox, packages)
    run = sandbox.commands.run if tool_class is E2BInterpreterTool else sandbox.process.exec
    return run.call_args.args[0] if run.call_args else None


def arguments_pip_gets(command: str, tmp_path) -> list[str]:
    """Run a pip command through the shell, as the sandboxes do, with a pip that prints its arguments."""
    pip = tmp_path / "pip"
    pip.write_text('#!/bin/sh\nshift 2\nfor arg in "$@"; do printf "%s\\n" "$arg"; done\n')
    pip.chmod(0o755)
    env = {"PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}", "GH_TOKEN": "ghp_token"}
    result = subprocess.run(["sh", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    return result.stdout.splitlines()


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
@pytest.mark.parametrize("tool_class", [E2BInterpreterTool, DaytonaInterpreterTool])
@pytest.mark.parametrize(
    "packages, requirements",
    [
        # Unquoted, the shell read ">=2.0" as a redirect: the latest pandas was installed and a file "=2.0" written.
        ("pandas>=2.0, numpy", ["pandas>=2.0", "numpy"]),
        ("pandas >= 2.0", ["pandas >= 2.0"]),
        ("https://example.com/pkg.whl?sig=a&exp=b", ["https://example.com/pkg.whl?sig=a&exp=b"]),
        # A comma inside one requirement does not split it.
        ("numpy>=1.20,<2.0", ["numpy>=1.20,<2.0"]),
        ("numpy>=1.20, <2.0, pandas", ["numpy>=1.20,<2.0", "pandas"]),
        ("pandas[excel,performance]>=2.0", ["pandas[excel,performance]>=2.0"]),
        # Lists the shell already read right keep working: quotes a model adds, variables, pip options.
        ("pandas, numpy", ["pandas", "numpy"]),
        ("pandas numpy", ["pandas", "numpy"]),
        ("'pandas>=2.0'", ["pandas>=2.0"]),
        ('"pandas>=2.0", "numpy"', ["pandas>=2.0", "numpy"]),
        ("'mypkg @ git+https://github.com/org/repo.git'", ["mypkg @ git+https://github.com/org/repo.git"]),
        ("git+https://$GH_TOKEN@github.com/org/repo.git", ["git+https://ghp_token@github.com/org/repo.git"]),
        ("https://example.com/pkg.whl?sig=YWJj== numpy", ["https://example.com/pkg.whl?sig=YWJj==", "numpy"]),
        ("--upgrade -r requirements.txt", ["--upgrade", "-r", "requirements.txt"]),
    ],
)
def test_pip_gets_each_requirement_as_it_was_written(tool_class, packages, requirements, tmp_path):
    assert arguments_pip_gets(pip_command(tool_class, packages), tmp_path) == requirements


@pytest.mark.parametrize("tool_class", [E2BInterpreterTool, DaytonaInterpreterTool])
def test_a_package_list_without_requirements_runs_no_pip(tool_class):
    assert pip_command(tool_class, " , ''") is None


def e2b_remote():
    remote = MagicMock()
    remote.commands.run.return_value = MagicMock(wait=MagicMock(return_value=MagicMock(exit_code=0, stdout="ok")))
    return remote


def persistent_e2b_tool(remote):
    tool = E2BInterpreterTool.model_construct(
        name="e2b", id="e2b", persistent_sandbox=True, timeout=600, is_files_allowed=False, files=None
    )
    tool._sandbox = remote
    return tool


def test_a_short_per_call_timeout_does_not_cut_the_sandbox_life_short():
    # A 30 s budget for one command must not leave the persistent sandbox with 30 s to live.
    remote = e2b_remote()
    tool = persistent_e2b_tool(remote)

    tool.execute(CodeInterpreterInputSchema(shell_command="ls", timeout=30))

    remote.set_timeout.assert_called_once_with(600)
    assert tool.timeout == 600


def test_a_call_late_in_the_sandbox_life_renews_it():
    # E2B's set_timeout(x) means "expire x seconds from now". Made with 600 s, the sandbox would expire 50 s into
    # a call made at t=550 with a 300 s timeout.
    clock = {"now": 0}
    remote = e2b_remote()
    remote.expires_at = 600
    remote.set_timeout.side_effect = lambda seconds: setattr(remote, "expires_at", clock["now"] + seconds)
    tool = persistent_e2b_tool(remote)

    clock["now"] = 550
    tool.execute(CodeInterpreterInputSchema(shell_command="make build", timeout=300))

    assert remote.expires_at >= 550 + 300


def test_a_long_per_call_timeout_keeps_the_sandbox_alive_for_that_call_only():
    remote = e2b_remote()
    tool = persistent_e2b_tool(remote)

    tool.execute(CodeInterpreterInputSchema(shell_command="make build", timeout=1800))
    tool.execute(CodeInterpreterInputSchema(shell_command="ls", timeout=60))

    assert remote.set_timeout.call_args_list == [call(1800), call(600)]
    assert tool.timeout == 600


def test_a_long_per_call_timeout_extends_a_sandbox_made_for_the_call():
    # Without a persistent sandbox, each call gets a new one, made with the configured timeout.
    remote = e2b_remote()
    tool = E2BInterpreterTool.model_construct(name="e2b", id="e2b", timeout=600, is_files_allowed=False, files=None)

    with patch.object(E2BInterpreterTool, "_create_sandbox", return_value=remote):
        tool.execute(CodeInterpreterInputSchema(shell_command="make build", timeout=1800))

    remote.set_timeout.assert_called_once_with(1800)
    assert tool.timeout == 600
