"""Unit tests for the shared AgentCore code-interpreter client helpers."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EventStreamError, ReadTimeoutError

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections.agentcore import (
    AgentCoreCodeInterpreterClient,
    AgentCoreInvocationResult,
    AgentCoreThrottlingError,
    normalize_sandbox_path,
)


def make_client():
    """Build a client whose boto3 client is a MagicMock."""
    boto_client = MagicMock()
    boto3_session = MagicMock()
    boto3_session.client.return_value = boto_client
    with patch.object(AWSConnection, "get_boto3_session", return_value=boto3_session):
        client = AgentCoreCodeInterpreterClient(AWSConnection(region="us-west-2"))
    return client, boto_client


# --- normalize_sandbox_path --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("input/data.csv", "input/data.csv"),
        ("./input/data.csv", "input/data.csv"),
        ("/abs/file.csv", "/abs/file.csv"),
        (".", "."),
        ("/", "."),
        ("input/../output/x", "output/x"),
        ("a/b/../../../etc/passwd", "../etc/passwd"),
    ],
)
def test_normalize_sandbox_path(raw, expected):
    assert normalize_sandbox_path(raw) == expected


# --- error translation -------------------------------------------------------


def test_translate_error_none_response_does_not_crash():
    """A ReadTimeoutError (whose .response is None) must not raise AttributeError."""
    err = ReadTimeoutError(endpoint_url="https://x")
    translated = AgentCoreCodeInterpreterClient._translate_error(err)
    assert translated is err


@pytest.mark.parametrize(
    "code",
    ["ThrottlingException", "throttlingException", "TooManyRequestsException", "ServiceQuotaExceededException"],
)
def test_translate_error_retryable_codes(code):
    """Both REST (PascalCase) and event-stream (camelCase) codes map to the retryable error."""
    err = ClientError({"Error": {"Code": code, "Message": "x"}}, "op")
    assert isinstance(AgentCoreCodeInterpreterClient._translate_error(err), AgentCoreThrottlingError)


def test_translate_error_mid_stream_event_stream_error():
    """A throttling exception raised mid-stream (EventStreamError, camelCase) is translated."""
    err = EventStreamError({"Error": {"Code": "throttlingException", "Message": "x"}}, "InvokeCodeInterpreter")
    assert isinstance(AgentCoreCodeInterpreterClient._translate_error(err), AgentCoreThrottlingError)


def test_translate_error_non_retryable_passthrough():
    err = ClientError({"Error": {"Code": "ValidationException", "Message": "x"}}, "op")
    assert AgentCoreCodeInterpreterClient._translate_error(err) is err


# --- client construction disables botocore retries ---------------------------


def test_client_disables_botocore_retries():
    """The client must not use botocore's default legacy retries (non-idempotent invoke)."""
    # The Config passed to session.client("bedrock-agentcore", config=...) must disable retries.
    with patch.object(AWSConnection, "get_boto3_session") as gs:
        gs.return_value = MagicMock()
        AgentCoreCodeInterpreterClient(AWSConnection(region="us-west-2"))
        _, call_kwargs = gs.return_value.client.call_args
        assert call_kwargs["config"].retries == {"total_max_attempts": 1}


# --- invoke stream parsing ---------------------------------------------------


def _stream(*results):
    return {"stream": iter([{"result": r} for r in results])}


def test_invoke_single_result():
    client, boto_client = make_client()
    boto_client.invoke_code_interpreter.return_value = _stream(
        {
            "content": [{"type": "text", "text": "hello"}],
            "structuredContent": {"stdout": "hello", "exitCode": 0},
            "isError": False,
        }
    )
    result = client.invoke(MagicMock(identifier="id", session_id="s"), "executeCode", {"code": "x"})
    assert result.stdout == "hello"
    assert result.exit_code == 0
    assert result.is_error is False


def test_invoke_merges_multiple_result_events():
    """Content accumulates and stdout concatenates across events; isError is sticky."""
    client, boto_client = make_client()
    boto_client.invoke_code_interpreter.return_value = _stream(
        {"content": [{"type": "text", "text": "part1"}], "structuredContent": {"stdout": "part1", "exitCode": 0}},
        {
            "content": [{"type": "text", "text": "part2"}],
            "structuredContent": {"stdout": "part2", "exitCode": 3},
            "isError": True,
        },
    )
    result = client.invoke(MagicMock(identifier="id", session_id="s"), "executeCommand", {"command": "x"})
    assert result.stdout == "part1part2"
    assert result.exit_code == 3
    assert result.is_error is True
    assert len(result.content) == 2


def test_invoke_empty_stream():
    client, boto_client = make_client()
    boto_client.invoke_code_interpreter.return_value = _stream()
    result = client.invoke(MagicMock(identifier="id", session_id="s"), "listFiles", {})
    assert result.content == []
    assert result.exit_code == 0
    assert result.is_error is False


def test_invoke_translates_mid_stream_throttling():
    """A throttling EventStreamError raised while iterating the stream becomes AgentCoreThrottlingError."""
    client, boto_client = make_client()

    def raising_stream():
        raise EventStreamError({"Error": {"Code": "throttlingException", "Message": "x"}}, "InvokeCodeInterpreter")
        yield  # pragma: no cover

    boto_client.invoke_code_interpreter.return_value = {"stream": raising_stream()}
    with pytest.raises(AgentCoreThrottlingError):
        client.invoke(MagicMock(identifier="id", session_id="s"), "executeCode", {"code": "x"})


# --- extract_file_bytes ------------------------------------------------------


def test_extract_file_bytes_blob_resource():
    result = AgentCoreInvocationResult(
        content=[{"type": "resource", "resource": {"type": "blob", "blob": b"raw bytes"}}]
    )
    assert AgentCoreCodeInterpreterClient.extract_file_bytes(result) == b"raw bytes"


def test_extract_file_bytes_text_resource():
    result = AgentCoreInvocationResult(content=[{"type": "resource", "resource": {"type": "text", "text": "plain"}}])
    assert AgentCoreCodeInterpreterClient.extract_file_bytes(result) == b"plain"


def test_extract_file_bytes_text_block_fallback():
    result = AgentCoreInvocationResult(content=[{"type": "text", "text": "fallback"}])
    assert AgentCoreCodeInterpreterClient.extract_file_bytes(result) == b"fallback"


def test_extract_file_bytes_str_blob_coercion():
    result = AgentCoreInvocationResult(
        content=[{"type": "resource", "resource": {"type": "blob", "blob": "already-str"}}]
    )
    assert AgentCoreCodeInterpreterClient.extract_file_bytes(result) == b"already-str"


def test_extract_file_bytes_empty_raises():
    with pytest.raises(FileNotFoundError):
        AgentCoreCodeInterpreterClient.extract_file_bytes(AgentCoreInvocationResult(content=[]))


# --- invocation result properties -------------------------------------------


def test_result_stdout_falls_back_to_text_blocks():
    result = AgentCoreInvocationResult(content=[{"type": "text", "text": "from-block"}], structured={})
    assert result.stdout == "from-block"


def test_result_error_text_prefers_stderr():
    result = AgentCoreInvocationResult(
        content=[{"type": "text", "text": "ignored"}], structured={"stderr": "boom"}, is_error=True
    )
    assert result.error_text == "boom"


def test_result_exit_code_fallback_on_error_without_structured():
    result = AgentCoreInvocationResult(content=[], structured={}, is_error=True)
    assert result.exit_code == 1
