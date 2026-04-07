"""Unit tests for DaytonaInterpreterTool with mocked Daytona SDK."""

import io
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Daytona as DaytonaConnection


@pytest.fixture
def mock_daytona_sdk():
    """Create a fully mocked Daytona SDK environment."""
    mock_client = MagicMock()
    mock_sandbox = MagicMock()
    mock_sandbox.id = "test-sandbox-id"
    mock_client.create.return_value = mock_sandbox

    default_exec_result = MagicMock()
    default_exec_result.exit_code = 0
    default_exec_result.result = ""
    mock_sandbox.process.exec.return_value = default_exec_result

    with patch.object(DaytonaConnection, "get_client", return_value=mock_client):
        yield {"client": mock_client, "sandbox": mock_sandbox}


@pytest.fixture
def daytona_tool(mock_daytona_sdk):
    """Create a DaytonaInterpreterTool with mocked SDK."""
    from dynamiq.nodes.tools.daytona_sandbox import DaytonaInterpreterTool

    tool = DaytonaInterpreterTool(
        connection=DaytonaConnection(api_key="test-key", api_url="https://test.api"),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
    )
    return tool


def test_python_execution(daytona_tool, mock_daytona_sdk):
    """Test Python code execution returns correct output."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]
    mock_result = MagicMock()
    mock_result.stdout = "42"
    mock_result.stderr = ""
    mock_result.error = None
    mock_sandbox.code_interpreter.run_code.return_value = mock_result

    input_data = DaytonaInterpreterInputSchema(python="print(42)")
    result = daytona_tool.execute(input_data)

    assert result["content"]["code_execution"] == "42"
    mock_sandbox.code_interpreter.run_code.assert_called_once()


def test_shell_command_execution(daytona_tool, mock_daytona_sdk):
    """Test shell command execution returns correct output."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]

    shell_result = MagicMock()
    shell_result.exit_code = 0
    shell_result.result = "hello world"

    def side_effect(cmd, **kwargs):
        if cmd.startswith("mkdir"):
            r = MagicMock()
            r.exit_code = 0
            r.result = ""
            return r
        return shell_result

    mock_sandbox.process.exec.side_effect = side_effect

    input_data = DaytonaInterpreterInputSchema(shell_command="echo hello world")
    result = daytona_tool.execute(input_data)

    assert result["content"]["shell_command_execution"] == "hello world"


def test_package_installation(daytona_tool, mock_daytona_sdk):
    """Test package installation via pip."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]

    pip_result = MagicMock()
    pip_result.exit_code = 0
    pip_result.result = ""

    code_result = MagicMock()
    code_result.stdout = "done"
    code_result.stderr = ""
    code_result.error = None

    def exec_side_effect(cmd, **kwargs):
        if "pip install" in cmd:
            return pip_result
        r = MagicMock()
        r.exit_code = 0
        r.result = ""
        return r

    mock_sandbox.process.exec.side_effect = exec_side_effect
    mock_sandbox.code_interpreter.run_code.return_value = code_result

    input_data = DaytonaInterpreterInputSchema(packages="pandas,numpy", python="print('done')")
    result = daytona_tool.execute(input_data)

    assert result["content"]["packages_installation"] == "Installed packages: pandas,numpy"
    # Verify pip install was called with the packages
    calls = [str(c) for c in mock_sandbox.process.exec.call_args_list]
    assert any("pip install" in c and "pandas" in c for c in calls)


def test_file_upload(daytona_tool, mock_daytona_sdk):
    """Test file upload to sandbox."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]

    code_result = MagicMock()
    code_result.stdout = "ok"
    code_result.stderr = ""
    code_result.error = None
    mock_sandbox.code_interpreter.run_code.return_value = code_result

    test_file = io.BytesIO(b"test content")
    test_file.name = "test.txt"

    input_data = DaytonaInterpreterInputSchema(
        python="print('ok')",
        files=[test_file],
    )
    result = daytona_tool.execute(input_data)

    assert "files_uploaded" in result["content"]
    mock_sandbox.fs.upload_file.assert_called()


def test_file_download(daytona_tool, mock_daytona_sdk):
    """Test file download returns base64 content."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]
    mock_sandbox.fs.download_file.return_value = b"file content here"

    input_data = DaytonaInterpreterInputSchema(download_files=["/home/user/output/result.txt"])
    result = daytona_tool.execute(input_data)

    assert "/home/user/output/result.txt" in result["content"]["files"]
    import base64

    expected_b64 = base64.b64encode(b"file content here").decode("utf-8")
    assert result["content"]["files"]["/home/user/output/result.txt"] == expected_b64


def test_python_error_handling(daytona_tool, mock_daytona_sdk):
    """Test that Python execution errors raise ToolExecutionException."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]
    mock_error = MagicMock()
    mock_error.name = "NameError"
    mock_error.value = "name 'x' is not defined"

    mock_result = MagicMock()
    mock_result.stdout = ""
    mock_result.stderr = ""
    mock_result.error = mock_error
    mock_sandbox.code_interpreter.run_code.return_value = mock_result

    input_data = DaytonaInterpreterInputSchema(python="print(x)")

    with pytest.raises(ToolExecutionException, match="NameError"):
        daytona_tool.execute(input_data)


def test_shell_command_error(daytona_tool, mock_daytona_sdk):
    """Test that failed shell commands raise ToolExecutionException."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]

    error_result = MagicMock()
    error_result.exit_code = 1
    error_result.result = "command not found"

    def side_effect(cmd, **kwargs):
        if cmd.startswith("mkdir"):
            r = MagicMock()
            r.exit_code = 0
            r.result = ""
            return r
        return error_result

    mock_sandbox.process.exec.side_effect = side_effect

    input_data = DaytonaInterpreterInputSchema(shell_command="nonexistent_command")

    with pytest.raises(ToolExecutionException, match="command not found"):
        daytona_tool.execute(input_data)


def test_close_deletes_sandbox(daytona_tool, mock_daytona_sdk):
    """Test that close() calls daytona.delete()."""
    mock_client = mock_daytona_sdk["client"]
    mock_sandbox = mock_daytona_sdk["sandbox"]

    daytona_tool.close()

    mock_client.delete.assert_called_once_with(mock_sandbox)


def test_persistent_sandbox_false(mock_daytona_sdk):
    """Test that sandbox is destroyed after execution when persistent_sandbox=False."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema
    from dynamiq.nodes.tools.daytona_sandbox import DaytonaInterpreterTool

    mock_client = mock_daytona_sdk["client"]
    mock_sandbox = mock_daytona_sdk["sandbox"]

    mock_result = MagicMock()
    mock_result.stdout = "hello"
    mock_result.stderr = ""
    mock_result.error = None
    mock_sandbox.code_interpreter.run_code.return_value = mock_result

    tool = DaytonaInterpreterTool(
        connection=DaytonaConnection(api_key="test-key", api_url="https://test"),
        persistent_sandbox=False,
        is_optimized_for_agents=False,
    )

    input_data = DaytonaInterpreterInputSchema(python="print('hello')")
    tool.execute(input_data)

    # Verify sandbox was destroyed after execution
    mock_client.delete.assert_called_once_with(mock_sandbox)


def test_optimized_for_agents_output(daytona_tool, mock_daytona_sdk):
    """Test formatted output when is_optimized_for_agents=True."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as DaytonaInterpreterInputSchema

    mock_sandbox = mock_daytona_sdk["sandbox"]
    mock_result = MagicMock()
    mock_result.stdout = "42"
    mock_result.stderr = ""
    mock_result.error = None
    mock_sandbox.code_interpreter.run_code.return_value = mock_result

    daytona_tool.is_optimized_for_agents = True

    input_data = DaytonaInterpreterInputSchema(python="print(42)")
    result = daytona_tool.execute(input_data)

    assert isinstance(result["content"], str)
    assert "## Output" in result["content"]
    assert "42" in result["content"]
    assert "files" in result


def test_backward_compatible_imports():
    """Verify key imports from e2b_sandbox and code_interpreter work."""
    from dynamiq.nodes.tools.code_interpreter import BaseCodeInterpreterTool, SandboxCreationErrorHandling
    from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool

    assert E2BInterpreterTool is not None
    assert BaseCodeInterpreterTool is not None
    assert SandboxCreationErrorHandling is not None
