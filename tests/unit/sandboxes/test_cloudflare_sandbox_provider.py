"""Unit tests for CloudflareSandbox provider with a mocked bridge client."""

from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Cloudflare as CloudflareConnection
from dynamiq.connections.cloudflare_sandbox import CloudflareExecResult

SANDBOX_ID = "testsandbox23456"


def exec_result(exit_code=0, stdout="", stderr=""):
    return CloudflareExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.fixture
def mock_cloudflare_client():
    """Create a fully mocked Cloudflare bridge client for the sandbox provider."""
    mock_client = MagicMock()
    mock_client.create_sandbox.return_value = SANDBOX_ID
    mock_client.is_running.return_value = True
    mock_client.exec.return_value = exec_result()

    with patch.object(CloudflareConnection, "get_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def cloudflare_sandbox(mock_cloudflare_client):
    """Create a CloudflareSandbox instance with mocked client."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox

    return CloudflareSandbox(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
    )


def test_run_command_shell(cloudflare_sandbox, mock_cloudflare_client):
    """Shell command execution maps to ShellCommandResult."""

    def side_effect(sandbox_id, command, **kwargs):
        if isinstance(command, str) and command.startswith("mkdir"):
            return exec_result()
        return exec_result(stdout="hello world")

    mock_cloudflare_client.exec.side_effect = side_effect

    result = cloudflare_sandbox.run_command_shell("echo hello world", timeout=30)

    assert result.is_success
    assert result.stdout == "hello world"
    assert result.exit_code == 0


def test_run_command_shell_background(cloudflare_sandbox, mock_cloudflare_client):
    """Background shell commands return immediately."""
    result = cloudflare_sandbox.run_command_shell("sleep 100", run_in_background_enabled=True)

    assert result.background is True
    background_calls = [call for call in mock_cloudflare_client.exec.call_args_list if "nohup" in str(call.args[1])]
    assert len(background_calls) == 1


def test_run_command_shell_error(cloudflare_sandbox, mock_cloudflare_client):
    """Transport errors surface as ShellCommandResult.error."""

    def side_effect(sandbox_id, command, **kwargs):
        if isinstance(command, str) and command.startswith("mkdir"):
            return exec_result()
        raise Exception("connection lost")

    mock_cloudflare_client.exec.side_effect = side_effect

    result = cloudflare_sandbox.run_command_shell("echo test")

    assert not result.is_success
    assert "connection lost" in result.error


def test_run_command_shell_passes_envs(mock_cloudflare_client):
    """Configured envs are passed to every command."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox

    sandbox = CloudflareSandbox(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        envs={"FOO": "bar"},
    )

    sandbox.run_command_shell("printenv FOO")

    shell_call = mock_cloudflare_client.exec.call_args_list[-1]
    assert shell_call.kwargs.get("env") == {"FOO": "bar"}


def test_upload_file(cloudflare_sandbox, mock_cloudflare_client):
    """File upload writes to base_path by default."""
    path = cloudflare_sandbox.upload_file("test.txt", b"content here")

    assert path == "/workspace/test.txt"
    mock_cloudflare_client.write_file.assert_called_once_with(SANDBOX_ID, "/workspace/test.txt", b"content here")


def test_upload_file_with_destination(cloudflare_sandbox, mock_cloudflare_client):
    """File upload honors an explicit destination path."""
    path = cloudflare_sandbox.upload_file("test.txt", b"content", destination_path="/workspace/custom/test.txt")

    assert path == "/workspace/custom/test.txt"
    mock_cloudflare_client.write_file.assert_called_once_with(SANDBOX_ID, "/workspace/custom/test.txt", b"content")


def test_list_files(cloudflare_sandbox, mock_cloudflare_client):
    """Listing files parses find output."""

    def side_effect(sandbox_id, command, **kwargs):
        if isinstance(command, str) and command.startswith("find"):
            return exec_result(stdout="/workspace/output/file1.txt\n/workspace/output/file2.csv\n")
        return exec_result()

    mock_cloudflare_client.exec.side_effect = side_effect

    files = cloudflare_sandbox.list_files("/workspace/output")

    assert files == ["/workspace/output/file1.txt", "/workspace/output/file2.csv"]


def test_exists_true(cloudflare_sandbox, mock_cloudflare_client):
    """exists() returns True when test -e succeeds."""
    assert cloudflare_sandbox.exists("test.txt") is True


def test_exists_false(cloudflare_sandbox, mock_cloudflare_client):
    """exists() returns False when test -e fails."""

    def side_effect(sandbox_id, command, **kwargs):
        if isinstance(command, str) and command.startswith("test -e"):
            return exec_result(exit_code=1)
        return exec_result()

    mock_cloudflare_client.exec.side_effect = side_effect

    assert cloudflare_sandbox.exists("missing.txt") is False


def test_retrieve(cloudflare_sandbox, mock_cloudflare_client):
    """retrieve() reads file bytes resolved against base_path."""
    mock_cloudflare_client.read_file.return_value = b"file bytes"

    content = cloudflare_sandbox.retrieve("test.txt")

    assert content == b"file bytes"
    mock_cloudflare_client.read_file.assert_called_once_with(SANDBOX_ID, "/workspace/test.txt")


def test_get_sandbox_info_with_port(cloudflare_sandbox, mock_cloudflare_client):
    """Sandbox info with a port creates a tunnel and returns its URL."""
    mock_cloudflare_client.create_tunnel.return_value = {
        "id": "t1",
        "port": 3000,
        "url": "https://random.trycloudflare.com",
        "hostname": "random.trycloudflare.com",
    }

    info = cloudflare_sandbox.get_sandbox_info(port=3000)

    assert info.public_url == "https://random.trycloudflare.com"
    assert info.public_host == "random.trycloudflare.com"
    assert info.sandbox_id == SANDBOX_ID


def test_get_sandbox_info_without_port(cloudflare_sandbox, mock_cloudflare_client):
    """Sandbox info without port returns basic info and creates no tunnel."""
    info = cloudflare_sandbox.get_sandbox_info()

    assert info.base_path == "/workspace"
    assert info.public_url is None
    mock_cloudflare_client.create_tunnel.assert_not_called()


def test_get_sandbox_info_tunnel_error(cloudflare_sandbox, mock_cloudflare_client):
    """Tunnel failures surface as public_url_error, not exceptions."""
    mock_cloudflare_client.create_tunnel.side_effect = Exception("tunnel refused")

    info = cloudflare_sandbox.get_sandbox_info(port=3000)

    assert info.public_url is None
    assert "tunnel refused" in info.public_url_error


def test_close_kill(cloudflare_sandbox, mock_cloudflare_client):
    """close(kill=True) destroys the sandbox."""
    cloudflare_sandbox._ensure_sandbox()

    cloudflare_sandbox.close(kill=True)

    mock_cloudflare_client.destroy_sandbox.assert_called_once_with(SANDBOX_ID)
    assert cloudflare_sandbox.sandbox_id is None


def test_close_no_kill(cloudflare_sandbox, mock_cloudflare_client):
    """close() without kill keeps the sandbox alive for reconnection."""
    cloudflare_sandbox._ensure_sandbox()

    cloudflare_sandbox.close(kill=False)

    mock_cloudflare_client.destroy_sandbox.assert_not_called()
    assert cloudflare_sandbox.sandbox_id == SANDBOX_ID


def test_reconnect_to_existing_sandbox(mock_cloudflare_client):
    """Reconnecting by id checks liveness instead of creating a new sandbox."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox

    sandbox = CloudflareSandbox(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        sandbox_id="existingid234567",
    )

    sandbox._ensure_sandbox()

    mock_cloudflare_client.is_running.assert_called_once_with("existingid234567")
    mock_cloudflare_client.create_sandbox.assert_not_called()


def test_reconnect_failure_raises_connection_error(mock_cloudflare_client):
    """Unreachable sandbox ids raise SandboxConnectionError with the Cloudflare provider."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox
    from dynamiq.sandboxes.exceptions import SandboxConnectionError

    mock_cloudflare_client.is_running.return_value = False

    sandbox = CloudflareSandbox(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        sandbox_id="existingid234567",
    )

    with pytest.raises(SandboxConnectionError, match="Cloudflare"):
        sandbox._ensure_sandbox()


def test_base_path_outside_workspace_rejected(mock_cloudflare_client):
    """base_path outside /workspace fails fast at construction."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox

    with pytest.raises(ValueError, match="must stay inside /workspace"):
        CloudflareSandbox(
            connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
            base_path="/home/user",
        )


def test_get_tools(cloudflare_sandbox):
    """get_tools returns the expected tool set."""
    tools = cloudflare_sandbox.get_tools()

    tool_names = [t.name for t in tools]
    assert "sandbox-shell" in tool_names
    assert "sandbox-info" in tool_names


def test_to_dict_excludes_envs_for_tracing(mock_cloudflare_client):
    """envs are serialized normally but excluded from tracing dumps."""
    from dynamiq.sandboxes.cloudflare import CloudflareSandbox

    sandbox = CloudflareSandbox(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        envs={"SECRET": "value"},
    )

    data = sandbox.to_dict()
    assert data["envs"] == {"SECRET": "value"}
    assert data["type"] == "dynamiq.sandboxes.CloudflareSandbox"

    tracing_data = sandbox.to_dict(for_tracing=True)
    assert "envs" not in tracing_data
    assert tracing_data["connection"] == {"id": sandbox.connection.id, "type": "dynamiq.connections.Cloudflare"}
