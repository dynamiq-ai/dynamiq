"""Unit tests for the AgentCore Runtime data-plane client.

The event-stream envelope and the command-wrapping rules are the things worth pinning:
both were established empirically against the live API and
both fail silently rather than loudly when they regress.
"""

import shlex
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EventStreamError, ReadTimeoutError

from dynamiq.connections import AWS as AWSConnection
from dynamiq.sandboxes.bedrock_agentcore_runtime_client import (
    MAX_COMMAND_BYTES,
    MAX_COMMAND_TIMEOUT_SECONDS,
    MIN_SESSION_ID_LENGTH,
    AgentCoreRuntimeClient,
    AgentCoreRuntimeCommandResult,
    AgentCoreRuntimeConflictError,
    AgentCoreRuntimeThrottlingError,
    CommandTooLargeError,
    derive_session_id,
    normalize_sandbox_path,
    wrap_shell_command,
)

ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-abc123"
SESSION_ID = "dq-test-session-0000000000000000000"


def make_client():
    """Build a client whose boto3 client is a MagicMock."""
    boto_client = MagicMock()
    boto3_session = MagicMock()
    boto3_session.client.return_value = boto_client
    with patch.object(AWSConnection, "get_boto3_session", return_value=boto3_session):
        client = AgentCoreRuntimeClient(AWSConnection(region="us-east-1"))
    return client, boto_client


def stream(*events):
    return {"stream": list(events)}


def delta(stdout=None, stderr=None):
    payload = {}
    if stdout is not None:
        payload["stdout"] = stdout
    if stderr is not None:
        payload["stderr"] = stderr
    return {"chunk": {"contentDelta": payload}}


def stop(exit_code=0, status="COMPLETED"):
    return {"chunk": {"contentStop": {"exitCode": exit_code, "status": status}}}


# --- command wrapping ----------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "echo one; echo two",
        "echo hello | tr a-z A-Z",
        "cd /mnt/workspace && export FOO=bar && echo $FOO",
        "ls /etc/hostn*",
        "echo 'it'\"'\"'s quoted'",
        'echo "a b   c"',
        "cat <<'EOF'\nhello\nEOF",
    ],
)
def test_wrap_shell_command_produces_a_single_bash_c_invocation(command):
    """Commands are exec'd, not shell-interpreted, so every command must go through bash -c."""
    wrapped = wrap_shell_command(command)
    assert shlex.split(wrapped) == ["bash", "-c", command]


def test_wrap_shell_command_overhead_is_small():
    """The wrapper must not meaningfully eat into the 64 KB command budget."""
    command = "echo " + "y" * 60_000
    assert len(wrap_shell_command(command)) - len(command) < 100


# --- stream parsing ----------------------------------------


def test_invoke_command_accumulates_interleaved_stdout_and_stderr():
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(
        {"chunk": {"contentStart": {}}},
        delta(stdout="first "),
        delta(stderr="warn "),
        delta(stdout="second"),
        delta(stderr="danger"),
        stop(exit_code=0),
    )

    result = client.invoke_command(ARN, SESSION_ID, "echo hi")

    assert result.stdout == "first second"
    assert result.stderr == "warn danger"
    assert result.exit_code == 0
    assert result.status == "COMPLETED"
    assert result.is_success is True


def test_invoke_command_reports_nonzero_exit_code():
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(delta(stderr="boom"), stop(exit_code=42))

    result = client.invoke_command(ARN, SESSION_ID, "exit 42")

    assert result.exit_code == 42
    assert result.is_success is False


def test_invoke_command_maps_timed_out_status():
    """A killed command reports exitCode -1 (not 124) with status TIMED_OUT."""
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(stop(exit_code=-1, status="TIMED_OUT"))

    result = client.invoke_command(ARN, SESSION_ID, "sleep 30", timeout=2)

    assert result.status == "TIMED_OUT"
    assert result.exit_code == -1
    assert result.timed_out is True
    assert result.is_success is False


def test_invoke_command_handles_missing_content_stop():
    """A stream that ends without contentStop must not crash or claim success."""
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(delta(stdout="partial"))

    result = client.invoke_command(ARN, SESSION_ID, "echo partial")

    assert result.stdout == "partial"
    assert result.exit_code is None
    assert result.is_success is False


def test_invoke_command_surfaces_in_stream_error_events():
    """Errors arrive as modeled stream members, not only as raised exceptions."""
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(
        {"chunk": {"validationException": {"message": "bad command", "reason": "invalid"}}}
    )

    with pytest.raises(Exception) as excinfo:
        client.invoke_command(ARN, SESSION_ID, "echo hi")
    assert "bad command" in str(excinfo.value)


def test_invoke_command_translates_in_stream_throttling_to_retryable():
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(
        {"chunk": {"throttlingException": {"message": "slow down"}}}
    )

    with pytest.raises(AgentCoreRuntimeThrottlingError):
        client.invoke_command(ARN, SESSION_ID, "echo hi")


# --- request shaping ---------------------------------------------------------


def test_invoke_command_threads_qualifier_and_wraps_command():
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(stop())

    client.invoke_command(ARN, SESSION_ID, "echo one; echo two", timeout=30, qualifier="v7")

    kwargs = boto_client.invoke_agent_runtime_command.call_args.kwargs
    assert kwargs["agentRuntimeArn"] == ARN
    assert kwargs["runtimeSessionId"] == SESSION_ID
    assert kwargs["qualifier"] == "v7"
    assert kwargs["body"]["timeout"] == 30
    assert shlex.split(kwargs["body"]["command"]) == ["bash", "-c", "echo one; echo two"]


def test_invoke_command_omits_qualifier_when_not_set():
    """qualifier is optional; omitting it makes the service use the default endpoint."""
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(stop())

    client.invoke_command(ARN, SESSION_ID, "echo hi", qualifier=None)

    assert "qualifier" not in boto_client.invoke_agent_runtime_command.call_args.kwargs


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        (None, 300),
        (0, 1),
        (-5, 1),
        (30, 30),
        (3600, 3600),
        (99999, MAX_COMMAND_TIMEOUT_SECONDS),
    ],
)
def test_invoke_command_clamps_timeout(supplied, expected):
    """Out-of-range timeouts are a hard ValidationException server-side, so clamp locally."""
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(stop())

    client.invoke_command(ARN, SESSION_ID, "echo hi", timeout=supplied)

    assert boto_client.invoke_agent_runtime_command.call_args.kwargs["body"]["timeout"] == expected


def test_invoke_command_rejects_oversized_command_before_sending():
    """The 64 KB limit applies to the wrapped payload, and must fail clearly, not as a 400."""
    client, boto_client = make_client()

    with pytest.raises(CommandTooLargeError) as excinfo:
        client.invoke_command(ARN, SESSION_ID, "echo " + "y" * MAX_COMMAND_BYTES)

    assert str(MAX_COMMAND_BYTES) in str(excinfo.value)
    boto_client.invoke_agent_runtime_command.assert_not_called()


def test_invoke_command_size_guard_measures_bytes_not_characters():
    """Multi-byte content must be measured in encoded bytes."""
    client, boto_client = make_client()
    command = "echo " + ("é" * (MAX_COMMAND_BYTES // 2))

    with pytest.raises(CommandTooLargeError):
        client.invoke_command(ARN, SESSION_ID, command)
    boto_client.invoke_agent_runtime_command.assert_not_called()


def test_invoke_command_allows_a_command_just_under_the_limit():
    client, boto_client = make_client()
    boto_client.invoke_agent_runtime_command.return_value = stream(stop())

    client.invoke_command(ARN, SESSION_ID, "echo " + "y" * (MAX_COMMAND_BYTES - 200))

    boto_client.invoke_agent_runtime_command.assert_called_once()


# --- stop_session ------------------------------------------------------------


def test_stop_session_sends_a_conforming_client_token():
    """clientToken has a 33-character minimum, same as runtimeSessionId."""
    client, boto_client = make_client()

    client.stop_session(ARN, SESSION_ID, qualifier="v3")

    kwargs = boto_client.stop_runtime_session.call_args.kwargs
    assert kwargs["agentRuntimeArn"] == ARN
    assert kwargs["runtimeSessionId"] == SESSION_ID
    assert kwargs["qualifier"] == "v3"
    assert len(kwargs["clientToken"]) >= MIN_SESSION_ID_LENGTH


def test_stop_session_swallows_resource_not_found():
    """Stopping an already-gone session is not an error."""
    client, boto_client = make_client()
    boto_client.stop_runtime_session.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "gone"}}, "StopRuntimeSession"
    )

    client.stop_session(ARN, SESSION_ID)


def test_stop_session_translates_conflict_to_retryable():
    client, boto_client = make_client()
    boto_client.stop_runtime_session.side_effect = ClientError(
        {"Error": {"Code": "RetryableConflictException", "Message": "busy"}}, "StopRuntimeSession"
    )

    with pytest.raises(AgentCoreRuntimeConflictError):
        client.stop_session(ARN, SESSION_ID)


# --- error translation -------------------------------------------------------


def test_translate_error_none_response_does_not_crash():
    err = ReadTimeoutError(endpoint_url="https://x")
    assert AgentCoreRuntimeClient._translate_error(err) is err


@pytest.mark.parametrize(
    "code",
    ["ThrottlingException", "throttlingException", "TooManyRequestsException", "ServiceQuotaExceededException"],
)
def test_translate_error_retryable_codes(code):
    err = ClientError({"Error": {"Code": code, "Message": "x"}}, "op")
    assert isinstance(AgentCoreRuntimeClient._translate_error(err), AgentCoreRuntimeThrottlingError)


@pytest.mark.parametrize("code", ["RetryableConflictException", "retryableConflictException"])
def test_translate_error_conflict_codes(code):
    """409 fires while a session provisions or tears down; it must be retried, not surfaced."""
    err = ClientError({"Error": {"Code": code, "Message": "x"}}, "op")
    assert isinstance(AgentCoreRuntimeClient._translate_error(err), AgentCoreRuntimeConflictError)


@pytest.mark.parametrize("code", ["ValidationException", "AccessDeniedException", "ResourceNotFoundException"])
def test_translate_error_leaves_non_retryable_codes_alone(code):
    err = ClientError({"Error": {"Code": code, "Message": "x"}}, "op")
    translated = AgentCoreRuntimeClient._translate_error(err)
    assert translated is err
    assert not isinstance(translated, (AgentCoreRuntimeThrottlingError, AgentCoreRuntimeConflictError))


def test_translate_error_mid_stream_event_stream_error():
    err = EventStreamError({"Error": {"Code": "throttlingException", "Message": "x"}}, "InvokeAgentRuntimeCommand")
    assert isinstance(AgentCoreRuntimeClient._translate_error(err), AgentCoreRuntimeThrottlingError)


def test_client_disables_botocore_retries():
    """InvokeAgentRuntimeCommand is not idempotent: botocore must not silently re-send it."""
    boto_client = MagicMock()
    boto3_session = MagicMock()
    boto3_session.client.return_value = boto_client
    with patch.object(AWSConnection, "get_boto3_session", return_value=boto3_session):
        AgentCoreRuntimeClient(AWSConnection(region="us-east-1"))

    config = boto3_session.client.call_args.kwargs["config"]
    assert config.retries["total_max_attempts"] == 1


# --- session identity --------------------------------------


def test_derive_session_id_meets_the_minimum_length():
    assert len(derive_session_id("c1")) >= MIN_SESSION_ID_LENGTH


def test_derive_session_id_is_deterministic():
    """Reconnect recomputes the id rather than storing it, so it must be stable."""
    assert derive_session_id("conversation-42") == derive_session_id("conversation-42")


def test_derive_session_id_is_unique_per_seed():
    assert derive_session_id("a") != derive_session_id("b")


def test_derive_session_id_stays_within_bounds_for_long_seeds():
    derived = derive_session_id("x" * 500)
    assert MIN_SESSION_ID_LENGTH <= len(derived) <= 256


@pytest.mark.parametrize("seed", ["with space", "with/slash", "with:colon", "üñíçø∂é", "with.dot"])
def test_derive_session_id_sanitises_unsafe_seeds(seed):
    """Session ids must survive being put in a URL path; keep them alphanumeric/dash."""
    derived = derive_session_id(seed)
    assert derived.replace("-", "").replace("_", "").isalnum()
    assert len(derived) >= MIN_SESSION_ID_LENGTH


# --- path handling -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("input/data.csv", "input/data.csv"),
        ("./input/data.csv", "input/data.csv"),
        ("/mnt/workspace/file.csv", "/mnt/workspace/file.csv"),
        (".", "."),
        ("/", "/"),
        ("input/../output/x", "output/x"),
    ],
)
def test_normalize_sandbox_path(raw, expected):
    assert normalize_sandbox_path(raw) == expected


# --- result helpers ----------------------------------------------------------


def test_command_result_error_text_prefers_stderr():
    result = AgentCoreRuntimeCommandResult(stdout="out", stderr="the failure", exit_code=1, status="COMPLETED")
    assert result.error_text == "the failure"


def test_command_result_error_text_falls_back_to_stdout():
    result = AgentCoreRuntimeCommandResult(stdout="only this", stderr="", exit_code=1, status="COMPLETED")
    assert result.error_text == "only this"


def test_command_result_error_text_mentions_timeout_when_timed_out():
    result = AgentCoreRuntimeCommandResult(stdout="", stderr="", exit_code=-1, status="TIMED_OUT")
    assert "timed out" in result.error_text.lower()
