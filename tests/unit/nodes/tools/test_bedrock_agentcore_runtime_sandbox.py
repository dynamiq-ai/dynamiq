"""Unit tests for BedrockAgentCoreRuntimeInterpreterTool."""

import io
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.bedrock_agentcore_runtime_sandbox import (
    AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER,
    BedrockAgentCoreRuntimeInterpreterTool,
)
from dynamiq.sandboxes.base import ShellCommandResult

ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-abc123"


def shell_ok(stdout="", stderr=None, exit_code=0):
    return ShellCommandResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def make_tool(**kwargs):
    kwargs.setdefault("connection", AWSConnection(region="us-east-1"))
    kwargs.setdefault("agent_runtime_arn", ARN)
    return BedrockAgentCoreRuntimeInterpreterTool(**kwargs)


def make_sandbox_mock():
    sandbox = MagicMock()
    sandbox.session_id = "dq-session-000000000000000000000000"
    sandbox.run_command_shell.return_value = shell_ok()
    sandbox.upload_file.side_effect = lambda name, content, destination_path=None, **kw: (
        destination_path or f"/mnt/workspace/{name}"
    )
    return sandbox


def commands_run(sandbox):
    return [call.args[0] for call in sandbox.run_command_shell.call_args_list]


# --- construction ------------------------------------------------------------


def test_defaults_to_an_aws_connection():
    tool = BedrockAgentCoreRuntimeInterpreterTool(agent_runtime_arn=ARN)
    assert isinstance(tool.connection, AWSConnection)


def test_base_path_defaults_to_the_session_storage_mount():
    tool = make_tool()
    assert tool.base_path == "/mnt/workspace"
    assert tool.input_dir == "/mnt/workspace/input"
    assert tool.output_dir == "/mnt/workspace/output"


def test_description_note_is_appended_once():
    tool = make_tool()
    assert AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER in tool.description

    rebuilt = make_tool(description=tool.description)
    assert rebuilt.description.count(AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER) == 1


def test_construction_does_not_touch_a_missing_api_key():
    """AWS connections have no api_key; the base class's eager path must not be used."""
    tool = make_tool(persistent_sandbox=False)
    assert tool.persistent_sandbox is False


def test_persistent_init_failure_degrades_to_per_execution_sessions():
    """A bad region or expired token must not blow up the constructor."""
    with patch.object(
        BedrockAgentCoreRuntimeInterpreterTool, "_create_sandbox", side_effect=ToolExecutionException("no creds")
    ):
        tool = make_tool(persistent_sandbox=True)

    assert tool.persistent_sandbox is False
    assert tool._sandbox is None


def test_persistent_init_success_keeps_the_session():
    sandbox = make_sandbox_mock()
    with patch.object(BedrockAgentCoreRuntimeInterpreterTool, "_create_sandbox", return_value=sandbox):
        tool = make_tool(persistent_sandbox=True)

    assert tool.persistent_sandbox is True
    assert tool._sandbox is sandbox


# --- sandbox plumbing --------------------------------------------------------


def test_create_sandbox_passes_configuration_through():
    tool = make_tool(qualifier="v4", session_seed="conv-1", s3_relay_bucket="relay", inline_transfer_max_bytes=99)
    with patch("dynamiq.sandboxes.bedrock_agentcore_runtime.BedrockAgentCoreRuntimeSandbox") as factory:
        tool._create_sandbox()

    kwargs = factory.call_args.kwargs
    assert kwargs["agent_runtime_arn"] == ARN
    assert kwargs["qualifier"] == "v4"
    assert kwargs["session_seed"] == "conv-1"
    assert kwargs["s3_relay_bucket"] == "relay"
    assert kwargs["inline_transfer_max_bytes"] == 99
    factory.return_value.ensure_started.assert_called_once()


def test_get_sandbox_id_returns_the_session_id():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    assert tool._get_sandbox_id(sandbox) == sandbox.session_id


def test_get_sandbox_host_is_none():
    """AgentCore exposes no inbound ports."""
    assert make_tool()._get_sandbox_host(make_sandbox_mock()) is None


def test_reconnect_sandbox_binds_the_given_session_id():
    tool = make_tool()
    with patch("dynamiq.sandboxes.bedrock_agentcore_runtime.BedrockAgentCoreRuntimeSandbox") as factory:
        tool._reconnect_sandbox("dq-restored-0000000000000000000000")

    assert factory.call_args.kwargs["session_id"] == "dq-restored-0000000000000000000000"


def test_destroy_sandbox_stops_the_session():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    tool._destroy_sandbox(sandbox)
    sandbox.close.assert_called_once_with(kill=True)


# --- shell command composition -----------------------------------------------


def test_compose_shell_command_adds_cd_and_exports():
    tool = make_tool()
    composed = tool._compose_shell_command("ls", env={"TOKEN": "abc"}, cwd="/mnt/workspace/sub")
    assert composed == "cd /mnt/workspace/sub && export TOKEN=abc && ls"


def test_compose_shell_command_quotes_hostile_values():
    tool = make_tool()
    composed = tool._compose_shell_command("ls", env={"X": "a; rm -rf /"}, cwd="/tmp/a b")
    assert "'a; rm -rf /'" in composed
    assert "'/tmp/a b'" in composed


def test_compose_shell_command_without_env_or_cwd_is_untouched():
    assert make_tool()._compose_shell_command("echo hi") == "echo hi"


def test_execute_shell_command_returns_stdout():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stdout="listing")

    assert tool._execute_shell_command("ls", sandbox) == "listing"


def test_execute_shell_command_raises_recoverable_on_failure():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stderr="permission denied", exit_code=13)

    with pytest.raises(ToolExecutionException) as excinfo:
        tool._execute_shell_command("cat /etc/shadow", sandbox)
    assert excinfo.value.recoverable is True
    assert "permission denied" in str(excinfo.value)


def test_execute_shell_command_honours_a_per_call_timeout():
    """Unlike the Code Interpreter backend, timeout applies per invocation."""
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._execute_shell_command("sleep 5", sandbox, timeout=42)

    assert sandbox.run_command_shell.call_args.kwargs["timeout"] == 42


# --- python execution --------------------------------------------------------


def test_execute_python_code_uploads_then_runs_the_script():
    """Runtime has no executeCode, so Python is materialized as a file and invoked."""
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stdout="42\n")

    output = tool._execute_python_code("print(6*7)", sandbox)

    assert output == "42\n"
    uploaded = sandbox.upload_file.call_args.kwargs["destination_path"]
    assert uploaded.startswith("/mnt/workspace/.dynamiq_exec_")
    assert sandbox.upload_file.call_args.args[1] == b"print(6*7)"
    assert any(cmd.startswith("python3 ") and uploaded in cmd for cmd in commands_run(sandbox))


def test_execute_python_code_removes_the_script_afterwards():
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._execute_python_code("print(1)", sandbox)

    assert any(cmd.startswith("rm -f") and ".dynamiq_exec_" in cmd for cmd in commands_run(sandbox))


def test_execute_python_code_cleans_up_even_when_execution_fails():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.side_effect = [shell_ok(stderr="Traceback", exit_code=1), shell_ok()]

    with pytest.raises(ToolExecutionException):
        tool._execute_python_code("raise SystemExit(1)", sandbox)

    assert any(cmd.startswith("rm -f") for cmd in commands_run(sandbox))


def test_execute_python_code_prefixes_params():
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._execute_python_code("print(threshold)", sandbox, params={"threshold": 7})

    source = sandbox.upload_file.call_args.args[1].decode()
    assert "threshold = 7" in source
    assert "print(threshold)" in source


def test_execute_python_code_reports_stderr_alongside_stdout():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stdout="out", stderr="warning")

    output = tool._execute_python_code("print('out')", sandbox)

    assert "out" in output
    assert "[stderr] warning" in output


def test_execute_python_code_wraps_upload_errors_as_recoverable():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.upload_file.side_effect = RuntimeError("relay down")

    with pytest.raises(ToolExecutionException) as excinfo:
        tool._execute_python_code("print(1)", sandbox)
    assert excinfo.value.recoverable is True


# --- package installation ----------------------------------------------------


def test_install_packages_quotes_version_pins():
    """An unquoted '>=' would be parsed as a redirection and silently install latest."""
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._install_packages(sandbox, "pandas>=2.0, requests")

    command = commands_run(sandbox)[0]
    assert "'pandas>=2.0'" in command
    assert "requests" in command


def test_install_packages_uses_python_m_pip():
    """pip3 is absent from a slim base image; python3 -m pip always works."""
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._install_packages(sandbox, "requests")

    assert commands_run(sandbox)[0].startswith("python3 -m pip install")


def test_install_packages_noop_for_empty_input():
    tool = make_tool()
    sandbox = make_sandbox_mock()

    tool._install_packages(sandbox, "")
    tool._install_packages(sandbox, "  ,  ")

    sandbox.run_command_shell.assert_not_called()


def test_install_packages_raises_on_failure():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stderr="No matching distribution", exit_code=1)

    with pytest.raises(ToolExecutionException, match="No matching distribution"):
        tool._install_packages(sandbox, "nonexistent-pkg")


# --- file transfer -----------------------------------------------------------


def test_upload_file_delegates_to_the_sandbox():
    tool = make_tool()
    sandbox = make_sandbox_mock()

    path = tool._upload_file_to_sandbox(io.BytesIO(b"data"), "/mnt/workspace/input/f.csv", sandbox)

    assert path == "/mnt/workspace/input/f.csv"
    assert sandbox.upload_file.call_args.args[1] == b"data"


def test_upload_file_wraps_errors_as_recoverable():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.upload_file.side_effect = ValueError("too large, no relay bucket")

    with pytest.raises(ToolExecutionException) as excinfo:
        tool._upload_file_to_sandbox(io.BytesIO(b"x"), "/mnt/workspace/f", sandbox)
    assert excinfo.value.recoverable is True


def test_download_file_delegates_to_the_sandbox():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.retrieve.return_value = b"contents"

    assert tool._download_file_bytes("/mnt/workspace/out.txt", sandbox) == b"contents"


def test_download_file_wraps_errors_as_recoverable():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.retrieve.side_effect = FileNotFoundError("missing")

    with pytest.raises(ToolExecutionException):
        tool._download_file_bytes("/mnt/workspace/nope", sandbox)


def test_run_shell_command_returns_exit_code_and_stdout():
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = shell_ok(stdout="ok", exit_code=0)

    assert tool._run_shell_command("mkdir -p /x", sandbox) == (0, "ok")


def test_run_shell_command_maps_a_missing_exit_code_to_failure():
    """A stream without contentStop must not read as success."""
    tool = make_tool()
    sandbox = make_sandbox_mock()
    sandbox.run_command_shell.return_value = ShellCommandResult(stdout="", exit_code=None, error="transport")

    exit_code, _ = tool._run_shell_command("mkdir -p /x", sandbox)
    assert exit_code == 1


# --- lifecycle ---------------------------------------------------------------


def test_close_stops_a_persistent_session():
    sandbox = make_sandbox_mock()
    with patch.object(BedrockAgentCoreRuntimeInterpreterTool, "_create_sandbox", return_value=sandbox):
        tool = make_tool(persistent_sandbox=True)

    tool.close()

    sandbox.close.assert_called_once_with(kill=True)
    assert tool._sandbox is None


def test_close_is_a_noop_without_a_persistent_session():
    make_tool().close()


def test_close_swallows_stop_errors():
    sandbox = make_sandbox_mock()
    with patch.object(BedrockAgentCoreRuntimeInterpreterTool, "_create_sandbox", return_value=sandbox):
        tool = make_tool(persistent_sandbox=True)
    sandbox.close.side_effect = RuntimeError("already gone")

    tool.close()
