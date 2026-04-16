"""Unit tests for DaytonaSandbox provider with mocked Daytona SDK."""

from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Daytona as DaytonaConnection


@pytest.fixture
def mock_daytona_env():
    """Create a fully mocked Daytona SDK environment for sandbox provider."""
    mock_client = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.id = "test-sandbox-123"
    mock_sandbox.get_user_home_dir.return_value = "/home/daytona"

    mock_client.create.return_value = mock_sandbox
    mock_client.get.return_value = mock_sandbox

    # Default process.exec result
    default_result = MagicMock()
    default_result.exit_code = 0
    default_result.result = ""
    mock_sandbox.process.exec.return_value = default_result

    with patch.object(DaytonaConnection, "get_client", return_value=mock_client):
        yield {
            "client": mock_client,
            "sandbox": mock_sandbox,
        }


@pytest.fixture
def daytona_sandbox(mock_daytona_env):
    """Create a DaytonaSandbox instance with mocked SDK."""
    from dynamiq.sandboxes.daytona import DaytonaSandbox

    sandbox = DaytonaSandbox(
        connection=DaytonaConnection(api_key="test-key", api_url="https://test.api"),
    )
    return sandbox


def test_run_command_shell(daytona_sandbox, mock_daytona_env):
    """Test shell command execution maps to ShellCommandResult."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    exec_result = MagicMock()
    exec_result.exit_code = 0
    exec_result.result = "hello world"
    mock_sdk_sandbox.process.exec.return_value = exec_result

    result = daytona_sandbox.run_command_shell("echo hello world", timeout=30)

    assert result.is_success
    assert result.stdout == "hello world"
    assert result.exit_code == 0


def test_run_command_shell_background(daytona_sandbox, mock_daytona_env):
    """Test background shell command execution."""
    result = daytona_sandbox.run_command_shell("sleep 100", run_in_background_enabled=True)

    assert result.background is True


def test_run_command_shell_error(daytona_sandbox, mock_daytona_env):
    """Test shell command that raises exception."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    mock_sdk_sandbox.process.exec.side_effect = Exception("connection lost")

    result = daytona_sandbox.run_command_shell("echo test")

    assert not result.is_success
    assert "connection lost" in result.error


def test_upload_file(daytona_sandbox, mock_daytona_env):
    """Test file upload to sandbox."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]

    path = daytona_sandbox.upload_file("test.txt", b"content here")

    assert path == "/home/daytona/test.txt"
    mock_sdk_sandbox.fs.upload_file.assert_called_once_with(b"content here", "/home/daytona/test.txt")


def test_upload_file_with_destination(daytona_sandbox, mock_daytona_env):
    """Test file upload with explicit destination path."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]

    path = daytona_sandbox.upload_file("test.txt", b"content", destination_path="/custom/path/test.txt")

    assert path == "/custom/path/test.txt"
    mock_sdk_sandbox.fs.upload_file.assert_called_once_with(b"content", "/custom/path/test.txt")


def test_list_files(daytona_sandbox, mock_daytona_env):
    """Test listing files in sandbox directory."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]

    file1 = MagicMock()
    file1.name = "file1.txt"
    file1.is_dir = False

    file2 = MagicMock()
    file2.name = "file2.csv"
    file2.is_dir = False

    mock_sdk_sandbox.fs.list_files.return_value = [file1, file2]

    files = daytona_sandbox.list_files("/home/daytona/output")

    assert len(files) == 2
    assert "/home/daytona/output/file1.txt" in files
    assert "/home/daytona/output/file2.csv" in files


def test_exists_true(daytona_sandbox, mock_daytona_env):
    """Test file exists returns True."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    mock_sdk_sandbox.fs.get_file_info.return_value = MagicMock()

    assert daytona_sandbox.exists("test.txt") is True


def test_exists_false(daytona_sandbox, mock_daytona_env):
    """Test file exists returns False when not found."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    mock_sdk_sandbox.fs.get_file_info.side_effect = Exception("not found")

    assert daytona_sandbox.exists("missing.txt") is False


def test_retrieve(daytona_sandbox, mock_daytona_env):
    """Test file retrieval returns bytes."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    mock_sdk_sandbox.fs.download_file.return_value = b"file bytes"

    content = daytona_sandbox.retrieve("test.txt")

    assert content == b"file bytes"


def test_get_sandbox_info_with_port(daytona_sandbox, mock_daytona_env):
    """Test sandbox info with port returns preview URL."""
    mock_sdk_sandbox = mock_daytona_env["sandbox"]
    mock_preview = MagicMock()
    mock_preview.url = "https://preview.daytona.io:3000"
    mock_sdk_sandbox.get_preview_link.return_value = mock_preview

    info = daytona_sandbox.get_sandbox_info(port=3000)

    assert info.public_url == "https://preview.daytona.io:3000"
    assert info.public_host == "preview.daytona.io:3000"
    assert info.sandbox_id == "test-sandbox-123"


def test_get_sandbox_info_without_port(daytona_sandbox, mock_daytona_env):
    """Test sandbox info without port returns basic info."""
    info = daytona_sandbox.get_sandbox_info()

    assert info.base_path == "/home/daytona"
    assert info.public_url is None


def test_close_kill(daytona_sandbox, mock_daytona_env):
    """Test close with kill=True deletes the sandbox."""
    mock_client = mock_daytona_env["client"]
    mock_sdk_sandbox = mock_daytona_env["sandbox"]

    # Trigger sandbox creation
    daytona_sandbox._ensure_sandbox()

    daytona_sandbox.close(kill=True)

    mock_client.delete.assert_called_once_with(mock_sdk_sandbox)
    assert daytona_sandbox.sandbox_id is None


def test_close_no_kill(daytona_sandbox, mock_daytona_env):
    """Test close without kill keeps sandbox alive."""
    mock_client = mock_daytona_env["client"]

    # Trigger sandbox creation
    daytona_sandbox._ensure_sandbox()

    daytona_sandbox.close(kill=False)

    mock_client.delete.assert_not_called()
    assert daytona_sandbox.sandbox_id == "test-sandbox-123"


def test_reconnect_to_existing_sandbox(mock_daytona_env):
    """Test reconnecting to an existing sandbox by ID."""
    from dynamiq.sandboxes.daytona import DaytonaSandbox

    mock_client = mock_daytona_env["client"]

    sandbox = DaytonaSandbox(
        connection=DaytonaConnection(api_key="test-key", api_url="https://test.api"),
        sandbox_id="existing-sandbox-id",
    )

    sandbox._ensure_sandbox()

    mock_client.get.assert_called_once_with("existing-sandbox-id")


def test_get_tools(daytona_sandbox):
    """Test that get_tools returns the expected tool set."""
    tools = daytona_sandbox.get_tools()

    tool_names = [t.name for t in tools]
    assert "sandbox-shell" in tool_names
    assert "sandbox-info" in tool_names
