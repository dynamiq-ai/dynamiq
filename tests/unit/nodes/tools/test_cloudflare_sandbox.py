"""Unit tests for CloudflareInterpreterTool with a mocked bridge client."""

import io
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Cloudflare as CloudflareConnection
from dynamiq.connections.cloudflare_sandbox import CloudflareExecResult

SANDBOX_ID = "testsandboxid234"


def exec_result(exit_code=0, stdout="", stderr=""):
    return CloudflareExecResult(exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.fixture
def mock_cloudflare_client():
    """Create a fully mocked Cloudflare bridge client."""
    mock_client = MagicMock()
    mock_client.create_sandbox.return_value = SANDBOX_ID
    mock_client.is_running.return_value = True
    mock_client.exec.return_value = exec_result()

    with patch.object(CloudflareConnection, "get_client", return_value=mock_client):
        yield mock_client


@pytest.fixture
def cloudflare_tool(mock_cloudflare_client):
    """Create a CloudflareInterpreterTool with mocked client."""
    from dynamiq.nodes.tools.cloudflare_sandbox import CloudflareInterpreterTool

    return CloudflareInterpreterTool(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
    )


def make_exec_side_effect(mock_client, python_result=None, shell_result=None, pip_result=None):
    """Route client.exec calls by command prefix, defaulting to success."""

    def side_effect(sandbox_id, command, **kwargs):
        cmd = command if isinstance(command, str) else " ".join(command)
        if cmd.startswith("python3") and python_result is not None:
            return python_result
        if "pip install" in cmd and pip_result is not None:
            return pip_result
        if cmd.startswith(("mkdir", "rm -f", "test -d")):
            return exec_result()
        if shell_result is not None:
            return shell_result
        return exec_result()

    mock_client.exec.side_effect = side_effect


def test_python_execution(cloudflare_tool, mock_cloudflare_client):
    """Python code is uploaded as a script and executed with python3."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="42"))

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(python="print(42)"))

    assert result["content"]["code_execution"] == "42"
    mock_cloudflare_client.write_file.assert_called()
    script_path = mock_cloudflare_client.write_file.call_args[0][1]
    assert script_path.startswith("/workspace/.dynamiq/script_")
    assert mock_cloudflare_client.write_file.call_args[0][2] == b"print(42)"


def test_python_execution_with_params(cloudflare_tool, mock_cloudflare_client):
    """Params are injected as Python globals before the user code."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="7"))

    cloudflare_tool.execute(CodeInterpreterInputSchema(python="print(x)", params={"x": 7}))

    uploaded_code = mock_cloudflare_client.write_file.call_args[0][2].decode()
    assert "x = 7" in uploaded_code
    assert "print(x)" in uploaded_code


def test_shell_command_execution(cloudflare_tool, mock_cloudflare_client):
    """Shell command execution returns stdout."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, shell_result=exec_result(stdout="hello world"))

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(shell_command="echo hello world"))

    assert result["content"]["shell_command_execution"] == "hello world"


def test_package_installation(cloudflare_tool, mock_cloudflare_client):
    """Packages are installed via pip before code runs."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="done"), pip_result=exec_result())

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(packages="pandas,numpy", python="print('done')"))

    assert result["content"]["packages_installation"] == "Installed packages: pandas,numpy"
    calls = [str(c) for c in mock_cloudflare_client.exec.call_args_list]
    assert any("pip install" in c and "pandas" in c for c in calls)


def test_file_upload(cloudflare_tool, mock_cloudflare_client):
    """Uploaded files are written under /workspace/input."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="ok"))

    test_file = io.BytesIO(b"test content")
    test_file.name = "test.txt"

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(python="print('ok')", files=[test_file]))

    assert "files_uploaded" in result["content"]
    uploaded_paths = [call[0][1] for call in mock_cloudflare_client.write_file.call_args_list]
    assert "/workspace/input/test.txt" in uploaded_paths


def test_file_download(cloudflare_tool, mock_cloudflare_client):
    """Downloaded files are returned as base64 content."""
    import base64

    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_cloudflare_client.read_file.return_value = b"file content here"

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(download_files=["/workspace/output/result.txt"]))

    expected_b64 = base64.b64encode(b"file content here").decode("utf-8")
    assert result["content"]["files"]["/workspace/output/result.txt"] == expected_b64


def test_python_error_handling(cloudflare_tool, mock_cloudflare_client):
    """Nonzero python exit codes raise ToolExecutionException with stderr."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(
        mock_cloudflare_client,
        python_result=exec_result(exit_code=1, stderr="NameError: name 'x' is not defined"),
    )

    with pytest.raises(ToolExecutionException, match="NameError"):
        cloudflare_tool.execute(CodeInterpreterInputSchema(python="print(x)"))


def test_shell_command_error(cloudflare_tool, mock_cloudflare_client):
    """Failed shell commands raise ToolExecutionException."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, shell_result=exec_result(exit_code=127, stderr="command not found"))

    with pytest.raises(ToolExecutionException, match="command not found"):
        cloudflare_tool.execute(CodeInterpreterInputSchema(shell_command="nonexistent_command"))


def test_stderr_appended_on_success(cloudflare_tool, mock_cloudflare_client):
    """Stderr with exit code 0 is appended to output, not raised."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="42", stderr="FutureWarning: soon"))

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(python="print(42)"))

    assert result["content"]["code_execution"] == "42\n[stderr] FutureWarning: soon"


def test_close_destroys_sandbox(cloudflare_tool, mock_cloudflare_client):
    """close() destroys the persistent sandbox."""
    cloudflare_tool.close()

    mock_cloudflare_client.destroy_sandbox.assert_called_once_with(SANDBOX_ID)
    assert cloudflare_tool._sandbox is None


def test_persistent_sandbox_false(mock_cloudflare_client):
    """Sandbox is destroyed after execution when persistent_sandbox=False."""
    from dynamiq.nodes.tools.cloudflare_sandbox import CloudflareInterpreterTool
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="hello"))

    tool = CloudflareInterpreterTool(
        connection=CloudflareConnection(api_key="test-key", url="https://bridge.test.workers.dev"),
        persistent_sandbox=False,
        is_optimized_for_agents=False,
    )

    tool.execute(CodeInterpreterInputSchema(python="print('hello')"))

    mock_cloudflare_client.destroy_sandbox.assert_called_once_with(SANDBOX_ID)


def test_optimized_for_agents_output(cloudflare_tool, mock_cloudflare_client):
    """Formatted markdown output when is_optimized_for_agents=True."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    make_exec_side_effect(mock_cloudflare_client, python_result=exec_result(stdout="42"))
    cloudflare_tool.is_optimized_for_agents = True

    result = cloudflare_tool.execute(CodeInterpreterInputSchema(python="print(42)"))

    assert isinstance(result["content"], str)
    assert "## Output" in result["content"]
    assert "42" in result["content"]
    assert "files" in result


def test_reconnect_sandbox(cloudflare_tool, mock_cloudflare_client):
    """Reconnection returns a handle bound to the same sandbox id."""
    sandbox = cloudflare_tool._reconnect_sandbox("existingid234567")

    mock_cloudflare_client.is_running.assert_called_once_with("existingid234567")
    assert sandbox.sandbox_id == "existingid234567"


def test_reconnect_sandbox_unreachable(cloudflare_tool, mock_cloudflare_client):
    """Reconnection to an unreachable sandbox raises."""
    mock_cloudflare_client.is_running.return_value = False

    with pytest.raises(ValueError, match="not reachable"):
        cloudflare_tool._reconnect_sandbox("existingid234567")


def test_from_checkpoint_state_reconnects(cloudflare_tool, mock_cloudflare_client):
    """Checkpoint restore reconnects to the saved sandbox_id."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterCheckpointState

    state = CodeInterpreterCheckpointState(sandbox_id="checkpointid2345", installed_packages=[])
    cloudflare_tool.from_checkpoint_state(state)

    mock_cloudflare_client.is_running.assert_called_with("checkpointid2345")
    assert cloudflare_tool._sandbox.sandbox_id == "checkpointid2345"


def test_description_uses_workspace_paths(cloudflare_tool):
    """Tool description must reference /workspace and not claim variable persistence."""
    assert "/workspace" in cloudflare_tool.description
    assert "{home_dir}" not in cloudflare_tool.description
    assert "/home/user" not in cloudflare_tool.description
    assert "Variables do NOT persist" in cloudflare_tool.description


def test_tool_serialization_roundtrip(mock_cloudflare_client, tmp_path):
    """Workflow with the tool node round-trips through YAML dump/load."""
    from dynamiq import Workflow
    from dynamiq.flows import Flow
    from dynamiq.nodes.tools.cloudflare_sandbox import CloudflareInterpreterTool

    tool = CloudflareInterpreterTool(
        id="cf-tool",
        connection=CloudflareConnection(id="cf-conn", api_key="test-key", url="https://bridge.test.workers.dev"),
    )
    workflow = Workflow(id="wf", flow=Flow(id="flow", nodes=[tool]))

    yaml_path = tmp_path / "cloudflare_tool.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)

    loaded_tool = loaded.flow.nodes[0]
    assert isinstance(loaded_tool, CloudflareInterpreterTool)
    assert loaded_tool.type == "dynamiq.nodes.tools.CloudflareInterpreterTool"
    assert loaded_tool.connection.type == "dynamiq.connections.Cloudflare"
    assert loaded_tool.connection.url == "https://bridge.test.workers.dev"
    assert loaded_tool.base_path == "/workspace"
