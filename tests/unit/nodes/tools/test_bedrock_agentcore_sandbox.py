"""Unit tests for BedrockAgentCoreInterpreterTool with mocked boto3 client."""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from dynamiq.connections import AWS as AWSConnection


def make_stream(*results):
    """Build a mocked invoke_code_interpreter response with result events."""
    return {"sessionId": "test-session-id", "stream": iter([{"result": r} for r in results])}


def make_result(stdout="", stderr="", exit_code=0, is_error=False, content=None):
    """Build a single result event payload mirroring the AgentCore response shape."""
    return {
        "content": content if content is not None else [{"type": "text", "text": stdout}],
        "structuredContent": {"stdout": stdout, "stderr": stderr, "exitCode": exit_code},
        "isError": is_error,
    }


@pytest.fixture
def mock_agentcore():
    """Create a fully mocked boto3 bedrock-agentcore client environment."""
    boto_client = MagicMock()
    boto_client.start_code_interpreter_session.return_value = {
        "codeInterpreterIdentifier": "aws.codeinterpreter.v1",
        "sessionId": "test-session-id",
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
def agentcore_tool(mock_agentcore):
    """Create a BedrockAgentCoreInterpreterTool with mocked boto3 client."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    return BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(access_key_id="test-key", secret_access_key="test-secret", region="us-west-2"),
        persistent_sandbox=True,
        is_optimized_for_agents=False,
    )


def test_python_execution(agentcore_tool, mock_agentcore):
    """Test Python code execution returns correct output."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(make_result(stdout="42"))

    result = agentcore_tool.execute(CodeInterpreterInputSchema(python="print(42)"))

    assert result["content"]["code_execution"] == "42"
    calls = mock_agentcore["client"].invoke_code_interpreter.call_args_list
    code_calls = [c for c in calls if c.kwargs["name"] == "executeCode"]
    assert len(code_calls) == 1
    assert code_calls[0].kwargs["arguments"] == {"code": "print(42)", "language": "python"}
    assert code_calls[0].kwargs["sessionId"] == "test-session-id"


def test_shell_command_execution(agentcore_tool, mock_agentcore):
    """Test shell command execution composes cwd and returns correct output."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    def command_handler(kwargs):
        command = kwargs["arguments"]["command"]
        if command.startswith(("mkdir", "test ")):
            return make_stream(make_result())
        return make_stream(make_result(stdout="hello world"))

    mock_agentcore["handlers"]["executeCommand"] = command_handler

    result = agentcore_tool.execute(CodeInterpreterInputSchema(shell_command="echo hello world"))

    assert result["content"]["shell_command_execution"] == "hello world"
    calls = mock_agentcore["client"].invoke_code_interpreter.call_args_list
    shell_calls = [c.kwargs["arguments"]["command"] for c in calls if c.kwargs["name"] == "executeCommand"]
    assert any(c == "cd ./output && echo hello world" for c in shell_calls)


def test_shell_command_env_composition(agentcore_tool, mock_agentcore):
    """Test env vars and cwd are composed into the shell command."""
    composed = agentcore_tool._compose_shell_command("printenv FOO", env={"FOO": "bar baz"}, cwd="./work")
    assert composed == "cd ./work && export FOO='bar baz' && printenv FOO"


def test_package_installation(agentcore_tool, mock_agentcore):
    """Test package installation via pip."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(make_result(stdout="done"))

    result = agentcore_tool.execute(CodeInterpreterInputSchema(packages="pandas,numpy", python="print('done')"))

    assert result["content"]["packages_installation"] == "Installed packages: pandas,numpy"
    calls = mock_agentcore["client"].invoke_code_interpreter.call_args_list
    commands = [c.kwargs["arguments"]["command"] for c in calls if c.kwargs["name"] == "executeCommand"]
    assert any(c == "pip install -qq pandas numpy" for c in commands)


def test_file_upload(agentcore_tool, mock_agentcore):
    """Test file upload sends normalized relative path and raw bytes."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    write_calls = []

    def write_handler(kwargs):
        write_calls.append(kwargs["arguments"])
        return make_stream(make_result())

    mock_agentcore["handlers"]["writeFiles"] = write_handler
    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(make_result(stdout="ok"))

    test_file = io.BytesIO(b"test content")
    test_file.name = "test.txt"

    result = agentcore_tool.execute(CodeInterpreterInputSchema(python="print('ok')", files=[test_file]))

    assert "files_uploaded" in result["content"]
    assert write_calls == [{"content": [{"path": "input/test.txt", "blob": b"test content"}]}]


def test_file_download(agentcore_tool, mock_agentcore):
    """Test file download returns base64 content and sends the normalized path."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    read_calls = []

    def read_handler(kwargs):
        read_calls.append(kwargs["arguments"])
        return make_stream(
            {
                "content": [
                    {
                        "type": "resource",
                        "resource": {"type": "blob", "uri": "file:///output/result.txt", "blob": b"file content here"},
                    }
                ],
                "isError": False,
            }
        )

    mock_agentcore["handlers"]["readFiles"] = read_handler

    result = agentcore_tool.execute(CodeInterpreterInputSchema(download_files=["./output/result.txt"]))

    expected_b64 = base64.b64encode(b"file content here").decode("utf-8")
    assert result["content"]["files"]["./output/result.txt"] == expected_b64
    # The path sent to AgentCore must be normalized (workspace-relative, no leading "./").
    assert read_calls == [{"paths": ["output/result.txt"]}]


def test_file_download_text_resource(agentcore_tool, mock_agentcore):
    """Test file download handles text resources."""
    mock_agentcore["handlers"]["readFiles"] = lambda kwargs: make_stream(
        {
            "content": [
                {"type": "resource", "resource": {"type": "text", "uri": "file:///a.txt", "text": "plain text"}}
            ],
            "isError": False,
        }
    )

    sandbox = agentcore_tool._sandbox
    assert agentcore_tool._download_file_bytes("a.txt", sandbox) == b"plain text"


def test_python_error_handling(agentcore_tool, mock_agentcore):
    """Test that Python execution errors raise ToolExecutionException."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(
        make_result(stderr="NameError: name 'x' is not defined", exit_code=1, is_error=True)
    )

    with pytest.raises(ToolExecutionException, match="NameError"):
        agentcore_tool.execute(CodeInterpreterInputSchema(python="print(x)"))


def test_shell_command_error(agentcore_tool, mock_agentcore):
    """Test that failed shell commands raise ToolExecutionException."""
    from dynamiq.nodes.agents.exceptions import ToolExecutionException
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    def command_handler(kwargs):
        command = kwargs["arguments"]["command"]
        if command.startswith(("mkdir", "test ")):
            return make_stream(make_result())
        return make_stream(make_result(stderr="command not found", exit_code=127, is_error=True))

    mock_agentcore["handlers"]["executeCommand"] = command_handler

    with pytest.raises(ToolExecutionException, match="command not found"):
        agentcore_tool.execute(CodeInterpreterInputSchema(shell_command="nonexistent_command"))


def test_close_stops_session(agentcore_tool, mock_agentcore):
    """Test that close() stops the AgentCore session."""
    agentcore_tool.close()

    mock_agentcore["client"].stop_code_interpreter_session.assert_called_once_with(
        codeInterpreterIdentifier="aws.codeinterpreter.v1", sessionId="test-session-id"
    )
    assert agentcore_tool._sandbox is None


def test_persistent_sandbox_false(mock_agentcore):
    """Test that the session is stopped after execution when persistent_sandbox=False."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(make_result(stdout="hello"))

    tool = BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(access_key_id="test-key", secret_access_key="test-secret", region="us-west-2"),
        persistent_sandbox=False,
        is_optimized_for_agents=False,
    )

    tool.execute(CodeInterpreterInputSchema(python="print('hello')"))

    mock_agentcore["client"].stop_code_interpreter_session.assert_called_once()


def test_optimized_for_agents_output(agentcore_tool, mock_agentcore):
    """Test formatted output when is_optimized_for_agents=True."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

    mock_agentcore["handlers"]["executeCode"] = lambda kwargs: make_stream(make_result(stdout="42"))
    agentcore_tool.is_optimized_for_agents = True

    result = agentcore_tool.execute(CodeInterpreterInputSchema(python="print(42)"))

    assert isinstance(result["content"], str)
    assert "## Output" in result["content"]
    assert "42" in result["content"]
    assert "files" in result


def test_default_connection_created(mock_agentcore):
    """Test that a default AWS connection is created when none is provided."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    tool = BedrockAgentCoreInterpreterTool()

    assert isinstance(tool.connection, AWSConnection)


def test_persistent_initialization_without_connection_fields(mock_agentcore):
    """Persistent mode initializes eagerly even when the AWS connection has no explicit fields."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    tool = BedrockAgentCoreInterpreterTool(persistent_sandbox=True)

    assert tool._sandbox is not None
    mock_agentcore["client"].start_code_interpreter_session.assert_called_once()


def test_persistent_init_falls_back_to_lazy_on_config_error(mock_agentcore):
    """A botocore config error during eager persistent init degrades to lazy per-execute init."""
    from botocore.exceptions import NoRegionError

    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    mock_agentcore["client"].start_code_interpreter_session.side_effect = NoRegionError()

    tool = BedrockAgentCoreInterpreterTool(persistent_sandbox=True)

    # Construction must not raise; sandbox stays uninitialized until first execute.
    assert tool._sandbox is None


def test_session_timeout_clamped(mock_agentcore):
    """Session timeout is clamped to the AgentCore maximum of 28800 seconds."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool

    BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(access_key_id="k", secret_access_key="s", region="us-west-2"),
        persistent_sandbox=True,
        timeout=999999,
    )

    call = mock_agentcore["client"].start_code_interpreter_session.call_args
    assert call.kwargs["sessionTimeoutSeconds"] == 28800


def test_reconnect_sandbox_ready(agentcore_tool, mock_agentcore):
    """Reconnect returns a session handle when the session is READY."""
    session = agentcore_tool._reconnect_sandbox("existing-session-id")

    assert session.session_id == "existing-session-id"
    assert session.identifier == "aws.codeinterpreter.v1"


def test_reconnect_sandbox_terminated(agentcore_tool, mock_agentcore):
    """Reconnect raises when the session is TERMINATED (cannot be restarted)."""
    mock_agentcore["client"].get_code_interpreter_session.return_value = {"status": "TERMINATED"}

    with pytest.raises(ValueError, match="TERMINATED"):
        agentcore_tool._reconnect_sandbox("dead-session-id")


def test_from_checkpoint_state_falls_back_to_new_session(agentcore_tool, mock_agentcore):
    """Checkpoint restore creates a new session when the saved one is terminated."""
    from dynamiq.nodes.tools.code_interpreter import CodeInterpreterCheckpointState

    mock_agentcore["client"].get_code_interpreter_session.return_value = {"status": "TERMINATED"}
    mock_agentcore["client"].start_code_interpreter_session.reset_mock()

    state = CodeInterpreterCheckpointState(sandbox_id="dead-session-id", installed_packages=[])
    agentcore_tool.from_checkpoint_state(state)

    mock_agentcore["client"].start_code_interpreter_session.assert_called_once()
    assert agentcore_tool._sandbox is not None


def test_throttling_retry_on_session_start(mock_agentcore):
    """Session creation retries on AWS throttling errors."""
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool
    from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling

    throttling_error = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}, "StartCodeInterpreterSession"
    )
    mock_agentcore["client"].start_code_interpreter_session.side_effect = [
        throttling_error,
        throttling_error,
        {"codeInterpreterIdentifier": "aws.codeinterpreter.v1", "sessionId": "retried-session-id"},
    ]

    tool = BedrockAgentCoreInterpreterTool(
        connection=AWSConnection(access_key_id="k", secret_access_key="s", region="us-west-2"),
        persistent_sandbox=True,
        creation_error_handling=SandboxCreationErrorHandling(
            max_retries=3, initial_wait_seconds=0.01, max_wait_seconds=0.02, jitter=0.0
        ),
    )

    assert tool._sandbox.session_id == "retried-session-id"
    assert mock_agentcore["client"].start_code_interpreter_session.call_count == 3


def test_stop_session_ignores_resource_not_found(agentcore_tool, mock_agentcore):
    """Stopping an already-stopped session is not an error."""
    mock_agentcore["client"].stop_code_interpreter_session.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}}, "StopCodeInterpreterSession"
    )

    agentcore_tool.close()

    assert agentcore_tool._sandbox is None


def test_backward_compatible_imports():
    """Verify key imports from bedrock_agentcore_sandbox and the shared client module work."""
    from dynamiq.connections.agentcore import (
        AgentCoreCodeInterpreterClient,
        AgentCoreInvocationResult,
        AgentCoreSession,
        AgentCoreThrottlingError,
    )
    from dynamiq.nodes.tools import BedrockAgentCoreInterpreterTool
    from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool as DirectImport

    assert BedrockAgentCoreInterpreterTool is DirectImport
    assert AgentCoreCodeInterpreterClient is not None
    assert AgentCoreSession is not None
    assert AgentCoreThrottlingError is not None
    assert AgentCoreInvocationResult is not None
