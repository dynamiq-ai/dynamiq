"""Unit tests for BedrockAgentCoreSandbox provider with mocked boto3 client."""

from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import AWS as AWSConnection


def make_stream(*results):
    """Build a mocked invoke_code_interpreter response with result events."""
    return {"sessionId": "test-session-123", "stream": iter([{"result": r} for r in results])}


def make_result(stdout="", stderr="", exit_code=0, is_error=False, content=None):
    """Build a single result event payload mirroring the AgentCore response shape."""
    return {
        "content": content if content is not None else [{"type": "text", "text": stdout}],
        "structuredContent": {"stdout": stdout, "stderr": stderr, "exitCode": exit_code},
        "isError": is_error,
    }


@pytest.fixture
def mock_agentcore_env():
    """Create a fully mocked boto3 bedrock-agentcore environment for the sandbox provider."""
    boto_client = MagicMock()
    boto_client.start_code_interpreter_session.return_value = {
        "codeInterpreterIdentifier": "aws.codeinterpreter.v1",
        "sessionId": "test-session-123",
    }
    boto_client.get_code_interpreter_session.return_value = {"status": "READY"}

    handlers = {}

    def invoke_side_effect(**kwargs):
        handler = handlers.get(kwargs["name"])
        if handler is not None:
            return handler(kwargs)
        return make_stream(make_result())

    boto_client.invoke_code_interpreter.side_effect = invoke_side_effect

    boto3_session = MagicMock()
    boto3_session.client.return_value = boto_client

    with patch.object(AWSConnection, "get_boto3_session", return_value=boto3_session):
        yield {"client": boto_client, "handlers": handlers}


@pytest.fixture
def agentcore_sandbox(mock_agentcore_env):
    """Create a BedrockAgentCoreSandbox instance with mocked boto3 client."""
    from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox

    return BedrockAgentCoreSandbox(
        connection=AWSConnection(access_key_id="test-key", secret_access_key="test-secret", region="us-west-2"),
    )


def test_run_command_shell(agentcore_sandbox, mock_agentcore_env):
    """Test shell command execution maps to ShellCommandResult and wraps with a timeout."""
    commands = []

    def handler(kwargs):
        commands.append(kwargs["arguments"]["command"])
        return make_stream(make_result(stdout="hello world"))

    mock_agentcore_env["handlers"]["executeCommand"] = handler

    result = agentcore_sandbox.run_command_shell("echo hello world", timeout=30)

    assert result.is_success
    assert result.stdout == "hello world"
    assert result.exit_code == 0
    # base_path is "." so no cd prefix, but the positive timeout must wrap the command.
    assert commands == ["timeout 30s sh -c 'echo hello world'"]


def test_run_command_shell_cd_into_base_path(mock_agentcore_env):
    """With a non-default base_path, commands run from that directory."""
    from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox

    sandbox = BedrockAgentCoreSandbox(
        connection=AWSConnection(access_key_id="k", secret_access_key="s", region="us-west-2"),
        base_path="workspace/agent1",
    )
    commands = []

    def handler(kwargs):
        commands.append(kwargs["arguments"]["command"])
        return make_stream(make_result(stdout="ok"))

    mock_agentcore_env["handlers"]["executeCommand"] = handler

    sandbox.run_command_shell("ls", timeout=0)

    assert commands[-1] == "cd workspace/agent1 && ls"


def test_run_command_shell_background(agentcore_sandbox, mock_agentcore_env):
    """Test background shell command execution uses startCommandExecution."""
    result = agentcore_sandbox.run_command_shell("sleep 100", run_in_background_enabled=True)

    assert result.background is True
    calls = mock_agentcore_env["client"].invoke_code_interpreter.call_args_list
    assert any(c.kwargs["name"] == "startCommandExecution" for c in calls)


def test_run_command_shell_error(agentcore_sandbox, mock_agentcore_env):
    """Test shell command that raises exception returns error result."""
    mock_agentcore_env["client"].invoke_code_interpreter.side_effect = Exception("connection lost")

    result = agentcore_sandbox.run_command_shell("echo test")

    assert not result.is_success
    assert "connection lost" in result.error


def test_upload_file(agentcore_sandbox, mock_agentcore_env):
    """Test file upload writes to the sandbox via writeFiles."""
    write_calls = []

    def write_handler(kwargs):
        write_calls.append(kwargs["arguments"])
        return make_stream(make_result())

    mock_agentcore_env["handlers"]["writeFiles"] = write_handler

    path = agentcore_sandbox.upload_file("test.txt", b"content here")

    assert path == "test.txt"
    assert write_calls == [{"content": [{"path": "test.txt", "blob": b"content here"}]}]


def test_upload_file_with_destination(agentcore_sandbox, mock_agentcore_env):
    """Test file upload with explicit destination path creates parent dirs."""
    write_calls = []

    def write_handler(kwargs):
        write_calls.append(kwargs["arguments"])
        return make_stream(make_result())

    mock_agentcore_env["handlers"]["writeFiles"] = write_handler

    path = agentcore_sandbox.upload_file("test.txt", b"content", destination_path="custom/path/test.txt")

    assert path == "custom/path/test.txt"
    assert write_calls == [{"content": [{"path": "custom/path/test.txt", "blob": b"content"}]}]
    commands = [
        c.kwargs["arguments"]["command"]
        for c in mock_agentcore_env["client"].invoke_code_interpreter.call_args_list
        if c.kwargs["name"] == "executeCommand"
    ]
    assert any(c == "mkdir -p custom/path" for c in commands)


def test_list_files(agentcore_sandbox, mock_agentcore_env):
    """Test listing files in the sandbox directory via find."""
    commands = []

    def handler(kwargs):
        commands.append(kwargs["arguments"]["command"])
        return make_stream(make_result(stdout="./output/file1.txt\n./output/file2.csv\n"))

    mock_agentcore_env["handlers"]["executeCommand"] = handler

    files = agentcore_sandbox.list_files("./output")

    assert files == ["./output/file1.txt", "./output/file2.csv"]
    assert commands == ["find ./output -maxdepth 3 -type f 2>/dev/null | head -50"]


def test_exists_true(agentcore_sandbox, mock_agentcore_env):
    """Test file exists returns True on zero exit code and uses `test -f` (files only)."""
    commands = []

    def handler(kwargs):
        commands.append(kwargs["arguments"]["command"])
        return make_stream(make_result(exit_code=0))

    mock_agentcore_env["handlers"]["executeCommand"] = handler

    assert agentcore_sandbox.exists("test.txt") is True
    assert commands == ["test -f test.txt"]


def test_exists_false(agentcore_sandbox, mock_agentcore_env):
    """Test file exists returns False on non-zero exit code."""
    mock_agentcore_env["handlers"]["executeCommand"] = lambda kwargs: make_stream(
        make_result(exit_code=1, is_error=True)
    )

    assert agentcore_sandbox.exists("missing.txt") is False


def test_retrieve(agentcore_sandbox, mock_agentcore_env):
    """Test file retrieval returns bytes and sends the resolved path to readFiles."""
    read_calls = []

    def handler(kwargs):
        read_calls.append(kwargs["arguments"])
        return make_stream(
            {
                "content": [
                    {"type": "resource", "resource": {"type": "blob", "uri": "file:///test.txt", "blob": b"file bytes"}}
                ],
                "isError": False,
            }
        )

    mock_agentcore_env["handlers"]["readFiles"] = handler

    assert agentcore_sandbox.retrieve("test.txt") == b"file bytes"
    assert read_calls == [{"paths": ["test.txt"]}]


def test_retrieve_missing_file(agentcore_sandbox, mock_agentcore_env):
    """Test retrieval of a missing file raises FileNotFoundError."""
    mock_agentcore_env["handlers"]["readFiles"] = lambda kwargs: make_stream(
        {"content": [{"type": "text", "text": "no such file"}], "isError": True}
    )

    with pytest.raises(FileNotFoundError, match="no such file"):
        agentcore_sandbox.retrieve("missing.txt")


def test_get_sandbox_info_with_port(agentcore_sandbox, mock_agentcore_env):
    """Test sandbox info with port explains that public URLs are unavailable."""
    agentcore_sandbox._ensure_sandbox()

    info = agentcore_sandbox.get_sandbox_info(port=3000)

    assert info.public_url is None
    assert "public ports" in info.public_url_error
    assert info.sandbox_id == "test-session-123"


def test_get_sandbox_info_without_port(agentcore_sandbox, mock_agentcore_env):
    """Test sandbox info without port returns basic info."""
    info = agentcore_sandbox.get_sandbox_info()

    assert info.base_path == "."
    assert info.public_url is None
    assert info.public_url_error is None


def test_close_kill(agentcore_sandbox, mock_agentcore_env):
    """Test close with kill=True stops the session."""
    agentcore_sandbox._ensure_sandbox()

    agentcore_sandbox.close(kill=True)

    mock_agentcore_env["client"].stop_code_interpreter_session.assert_called_once_with(
        codeInterpreterIdentifier="aws.codeinterpreter.v1", sessionId="test-session-123"
    )
    assert agentcore_sandbox.session_id is None


def test_close_no_kill(agentcore_sandbox, mock_agentcore_env):
    """Test close without kill keeps the session alive for reconnection."""
    agentcore_sandbox._ensure_sandbox()

    agentcore_sandbox.close(kill=False)

    mock_agentcore_env["client"].stop_code_interpreter_session.assert_not_called()
    assert agentcore_sandbox.session_id == "test-session-123"


def test_reconnect_to_existing_session(mock_agentcore_env):
    """Test reconnecting to an existing READY session by ID."""
    from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox

    sandbox = BedrockAgentCoreSandbox(
        connection=AWSConnection(access_key_id="test-key", secret_access_key="test-secret", region="us-west-2"),
        session_id="existing-session-id",
    )

    session = sandbox._ensure_sandbox()

    assert session.session_id == "existing-session-id"
    mock_agentcore_env["client"].get_code_interpreter_session.assert_called_once_with(
        codeInterpreterIdentifier="aws.codeinterpreter.v1", sessionId="existing-session-id"
    )
    mock_agentcore_env["client"].start_code_interpreter_session.assert_not_called()


def test_reconnect_terminated_session_raises(mock_agentcore_env):
    """Test reconnecting to a TERMINATED session raises SandboxConnectionError."""
    from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox
    from dynamiq.sandboxes.exceptions import SandboxConnectionError

    mock_agentcore_env["client"].get_code_interpreter_session.return_value = {"status": "TERMINATED"}

    sandbox = BedrockAgentCoreSandbox(
        connection=AWSConnection(access_key_id="test-key", secret_access_key="test-secret", region="us-west-2"),
        session_id="dead-session-id",
    )

    with pytest.raises(SandboxConnectionError, match="dead-session-id"):
        sandbox._ensure_sandbox()


def test_get_tools(agentcore_sandbox):
    """Test that get_tools returns the expected tool set."""
    tools = agentcore_sandbox.get_tools()

    tool_names = [t.name for t in tools]
    assert "sandbox-shell" in tool_names
    assert "sandbox-info" in tool_names


def test_supports_views_false(agentcore_sandbox):
    """AgentCore sandboxes do not support shared views."""
    assert agentcore_sandbox.supports_views is False


def test_serialization_round_trip(agentcore_sandbox):
    """Test to_dict produces the expected type string and excludes the connection body."""
    data = agentcore_sandbox.to_dict(for_tracing=True)

    assert data["type"] == "dynamiq.sandboxes.BedrockAgentCoreSandbox"
    assert data["code_interpreter_identifier"] == "aws.codeinterpreter.v1"
    assert "secret_access_key" not in str(data.get("connection", {}))
