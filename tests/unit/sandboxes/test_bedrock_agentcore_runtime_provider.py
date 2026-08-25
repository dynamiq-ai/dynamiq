"""Unit tests for BedrockAgentCoreRuntimeSandbox."""

import base64
import shlex
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.sandboxes import BedrockAgentCoreRuntimeSandbox
from dynamiq.sandboxes.bedrock_agentcore_runtime_client import (
    MAX_COMMAND_BYTES,
    AgentCoreRuntimeCommandResult,
    AgentCoreRuntimeConflictError,
    AgentCoreRuntimeThrottlingError,
    CommandTooLargeError,
)

ARN = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-abc123"


def ok(stdout="", stderr="", exit_code=0, status="COMPLETED"):
    return AgentCoreRuntimeCommandResult(stdout=stdout, stderr=stderr, exit_code=exit_code, status=status)


def make_sandbox(**kwargs):
    """Sandbox with a mocked runtime client; returns (sandbox, mock_client)."""
    kwargs.setdefault("connection", AWSConnection(region="us-east-1"))
    kwargs.setdefault("agent_runtime_arn", ARN)
    sandbox = BedrockAgentCoreRuntimeSandbox(**kwargs)
    client = MagicMock()
    client.invoke_command.return_value = ok()
    sandbox._client = client
    return sandbox, client


def make_started_sandbox(**kwargs):
    """Sandbox already started, with the mock reset so tests see only their own calls.

    ``ensure_started`` issues a ``mkdir -p`` for base_path, which would otherwise consume
    the first entry of every ``side_effect`` sequence.
    """
    sandbox, client = make_sandbox(**kwargs)
    sandbox.ensure_started()
    client.reset_mock(return_value=False, side_effect=True)
    client.invoke_command.return_value = ok()
    return sandbox, client


def commands_sent(client):
    return [call.kwargs["command"] for call in client.invoke_command.call_args_list]


# --- session identity --------------------------------------------------------


def test_ensure_started_derives_a_session_id():
    sandbox, _ = make_sandbox()
    assert sandbox.session_id is None

    session_id = sandbox.ensure_started()

    assert session_id and len(session_id) >= 33
    assert sandbox.session_id == session_id


def test_ensure_started_is_deterministic_for_a_seed():
    """Reconnect recomputes the id from the conversation rather than storing it."""
    first, _ = make_sandbox(session_seed="conversation-99")
    second, _ = make_sandbox(session_seed="conversation-99")

    assert first.ensure_started() == second.ensure_started()


def test_ensure_started_respects_an_explicit_session_id():
    explicit = "dq-explicit-session-000000000000000"
    sandbox, _ = make_sandbox(session_id=explicit)

    assert sandbox.ensure_started() == explicit


def test_ensure_started_is_idempotent():
    sandbox, client = make_sandbox()

    first = sandbox.ensure_started()
    calls_after_first = client.invoke_command.call_count
    second = sandbox.ensure_started()

    assert first == second
    assert client.invoke_command.call_count == calls_after_first


def test_ensure_started_creates_the_base_directory():
    sandbox, client = make_sandbox(base_path="/mnt/workspace/project")

    sandbox.ensure_started()

    assert any("mkdir -p" in cmd and "/mnt/workspace/project" in cmd for cmd in commands_sent(client))


def test_current_sandbox_id_exposes_the_session():
    sandbox, _ = make_sandbox()
    sandbox.ensure_started()
    assert sandbox.current_sandbox_id == sandbox.session_id


# --- path resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("data.csv", "/mnt/workspace/data.csv"),
        ("sub/data.csv", "/mnt/workspace/sub/data.csv"),
        ("./data.csv", "/mnt/workspace/data.csv"),
        ("/mnt/workspace/data.csv", "/mnt/workspace/data.csv"),
        ("/etc/hosts", "/etc/hosts"),
    ],
)
def test_resolve_path(raw, expected):
    sandbox, _ = make_sandbox()
    assert sandbox._resolve_path(raw) == expected


def test_resolve_path_is_idempotent():
    """collect_files/delete_file pre-resolve, so a second pass must not double the prefix."""
    sandbox, _ = make_sandbox()
    once = sandbox._resolve_path("data.csv")
    assert sandbox._resolve_path(once) == once


# --- views -------------------------------------------------------------------


def test_supports_views():
    sandbox, _ = make_sandbox()
    assert sandbox.supports_views is True


def test_create_view_shares_the_session_and_rebases_paths():
    sandbox, _ = make_sandbox()
    session_id = sandbox.ensure_started()

    view = sandbox.create_view(base_path="/mnt/workspace/agent-2")

    assert view.session_id == session_id
    assert view.agent_runtime_arn == sandbox.agent_runtime_arn
    assert view.base_path == "/mnt/workspace/agent-2"
    assert view._resolve_path("out.txt") == "/mnt/workspace/agent-2/out.txt"


def test_create_view_carries_transfer_and_qualifier_config():
    sandbox, _ = make_sandbox(qualifier="v7", s3_relay_bucket="relay", inline_transfer_max_bytes=123)

    view = sandbox.create_view(base_path="/mnt/workspace/x")

    assert view.qualifier == "v7"
    assert view.s3_relay_bucket == "relay"
    assert view.inline_transfer_max_bytes == 123


def test_create_view_accepts_an_explicit_sandbox_id():
    sandbox, _ = make_sandbox()
    view = sandbox.create_view(base_path="/mnt/workspace/x", sandbox_id="dq-other-session-00000000000000000")
    assert view.session_id == "dq-other-session-00000000000000000"


def test_create_view_carries_the_session_env():
    """Views share one session, so they must share the file that session's env lives in."""
    sandbox, _ = make_sandbox(session_env={"TOKEN": "abc"}, session_env_param="/dq/conv-1")

    view = sandbox.create_view(base_path="/mnt/workspace/x")

    assert view.session_env == {"TOKEN": "abc"}
    assert view.session_env_param == "/dq/conv-1"
    assert view.session_env_path == sandbox.session_env_path


# --- session env -------------------------------------------------------------


def bootstrap_commands(client):
    return [cmd for cmd in commands_sent(client) if ".dqenv" in cmd]


def bootstrap_command(client) -> str:
    sent = bootstrap_commands(client)
    assert sent, "no bootstrap command was sent"
    return sent[0]


def only_bootstrap_fails(result_or_error):
    """Side effect that fails the env-file write while leaving mkdir and friends healthy."""

    def side_effect(**kwargs):
        if ".dqenv" not in kwargs["command"]:
            return ok()
        if isinstance(result_or_error, Exception):
            raise result_or_error
        return result_or_error

    return side_effect


def test_no_env_means_no_bootstrap_and_no_prefix():
    sandbox, client = make_sandbox()

    sandbox.run_command_shell("echo hi")

    assert bootstrap_commands(client) == []
    assert commands_sent(client)[-1] == "echo hi"


def test_bootstrap_writes_the_env_file_atomically_and_locks_it_down():
    sandbox, client = make_sandbox(session_env={"TOKEN": "abc"})

    sandbox.ensure_started()

    command = bootstrap_command(client)
    assert command is not None
    assert "TOKEN=abc" in command
    assert "umask 077" in command and "chmod 600" in command
    # Written to a temp path and moved, so a concurrent view never reads a partial file.
    assert ".tmp.$$" in command and "mv " in command


def test_bootstrap_runs_once_per_ensure_started():
    sandbox, client = make_sandbox(session_env={"TOKEN": "abc"})

    sandbox.ensure_started()
    sandbox.ensure_started()

    assert len(bootstrap_commands(client)) == 1


def test_bootstrap_overwrites_rather_than_skipping():
    """Session storage outlives the token in it; a write-once bootstrap serves a stale one."""
    sandbox, client = make_sandbox(session_env={"TOKEN": "fresh"})

    sandbox.ensure_started()

    command = bootstrap_command(client)
    assert "TOKEN=fresh" in command
    # No existence guard anywhere in the write path.
    assert "-f " not in command and "-e " not in command


@pytest.mark.parametrize("value", ["a'b", "a$b", "a;b", "a\nb", 'a"b', "a b"])
def test_bootstrap_quotes_values_that_would_otherwise_break_out(value):
    sandbox, client = make_sandbox(session_env={"TOKEN": value})

    sandbox.ensure_started()

    command = bootstrap_command(client)
    # The value survives shlex round-tripping rather than terminating the assignment.
    assert shlex.split(shlex.split(command[command.index("printf") :])[2])[0] == f"TOKEN={value}"


def test_bootstrap_from_a_parameter_never_sends_the_value():
    sandbox, client = make_sandbox(session_env_param="/dynamiq/prod/conv-42")

    sandbox.ensure_started()

    command = bootstrap_command(client)
    assert "aws ssm get-parameter" in command
    assert "/dynamiq/prod/conv-42" in command
    assert "--with-decryption" in command
    assert "--region us-east-1" in command


def test_a_parameter_takes_precedence_over_inline_values():
    sandbox, client = make_sandbox(session_env={"TOKEN": "inline-secret"}, session_env_param="/dq/conv-1")

    sandbox.ensure_started()

    command = bootstrap_command(client)
    assert "aws ssm get-parameter" in command
    assert "inline-secret" not in command


def test_bootstrap_failure_is_not_fatal():
    sandbox, client = make_sandbox(session_env={"TOKEN": "abc"})
    client.invoke_command.side_effect = only_bootstrap_fails(RuntimeError("boom"))

    assert sandbox.ensure_started() is not None


def test_a_failed_bootstrap_does_not_log_its_output(caplog):
    """``error_text`` is the command's stderr, which on the inline path can echo the values."""
    sandbox, client = make_sandbox(session_env={"TOKEN": "super-secret"})
    client.invoke_command.side_effect = only_bootstrap_fails(
        ok(stderr="bash: TOKEN=super-secret: command not found", exit_code=127)
    )

    sandbox.ensure_started()

    assert "super-secret" not in caplog.text


def test_no_env_value_reaches_the_logs(caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    sandbox, _ = make_sandbox(session_env={"TOKEN": "super-secret"})

    sandbox.ensure_started()
    sandbox.run_command_shell("echo hi")

    assert "super-secret" not in caplog.text


def test_commands_source_the_env_file():
    sandbox, client = make_started_sandbox(session_env={"TOKEN": "abc"})

    sandbox.run_command_shell("echo hi")

    sent = commands_sent(client)[-1]
    assert sent.startswith("set -a; ")
    assert "/mnt/workspace/.dqenv" in sent
    assert sent.endswith("echo hi")


def test_the_prefix_does_not_carry_the_value():
    """The whole point: after bootstrap the secret is absent from every command string."""
    sandbox, client = make_started_sandbox(session_env={"TOKEN": "super-secret"})

    sandbox.run_command_shell("echo hi")

    assert all("super-secret" not in cmd for cmd in commands_sent(client))


def test_background_commands_source_the_env_outside_nohup():
    sandbox, client = make_started_sandbox(session_env={"TOKEN": "abc"})

    sandbox.run_command_shell("sleep 100", run_in_background_enabled=True)

    sent = commands_sent(client)[-1]
    # `&` must detach only the user's command, not the sourcing.
    assert sent.index("set -a;") < sent.index("nohup")


def test_internal_plumbing_is_not_prefixed():
    """Uploads and listings need no user env and are tight against the 64 KB budget."""
    sandbox, client = make_started_sandbox(session_env={"TOKEN": "abc"})

    sandbox.upload_file("f.txt", b"data")
    sandbox.list_files()
    sandbox.exists("f.txt")

    assert all(not cmd.startswith("set -a;") for cmd in commands_sent(client))


def test_the_env_file_is_pinned_to_the_session_mount_not_base_path():
    """Views differ only in base_path, so a base_path-relative file would fork per view."""
    sandbox, client = make_started_sandbox(base_path="/mnt/workspace/agent-2", session_env={"TOKEN": "abc"})

    sandbox.run_command_shell("echo hi")

    assert "/mnt/workspace/.dqenv" in commands_sent(client)[-1]


def test_an_oversized_env_degrades_instead_of_crashing_the_session():
    """`CommandTooLargeError` is raised client-side, before the wire; bootstrap absorbs it.

    Asserted here rather than through ``run_command_shell``, which swallows every exception
    into ``ShellCommandResult(error=...)`` and so can never surface the type.
    """
    sandbox, client = make_sandbox(session_env={"BIG": "x" * (MAX_COMMAND_BYTES + 1)})
    client.invoke_command.side_effect = only_bootstrap_fails(CommandTooLargeError("too large"))

    assert sandbox.ensure_started() is not None
    assert sandbox._started is True


def test_session_env_is_hidden_from_tracing_but_present_for_execution():
    sandbox, _ = make_sandbox(session_env={"TOKEN": "abc"})

    assert sandbox.to_dict(for_tracing=True).get("session_env") is None
    assert sandbox.to_dict()["session_env"] == {"TOKEN": "abc"}


def test_a_parameter_name_is_safe_to_serialize_either_way():
    sandbox, _ = make_sandbox(session_env_param="/dq/conv-1")

    assert sandbox.to_dict(for_tracing=True)["session_env_param"] == "/dq/conv-1"


# --- command execution -------------------------------------------------------


def test_run_command_shell_returns_stdout_and_exit_code():
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(stdout="hello\n", exit_code=0)

    result = sandbox.run_command_shell("echo hello")

    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert result.is_success is True


def test_run_command_shell_threads_qualifier_and_timeout():
    sandbox, client = make_sandbox(qualifier="v3")

    sandbox.run_command_shell("echo hi", timeout=45)

    kwargs = client.invoke_command.call_args.kwargs
    assert kwargs["qualifier"] == "v3"
    assert kwargs["timeout"] == 45
    assert kwargs["agent_runtime_arn"] == ARN


def test_run_command_shell_surfaces_failure():
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(stderr="boom", exit_code=1)

    result = sandbox.run_command_shell("false")

    assert result.exit_code == 1
    assert result.stderr == "boom"
    assert result.is_success is False


def test_run_command_shell_marks_timeouts_as_errors():
    """A killed command reports exitCode -1; it must not read as a plain non-zero exit."""
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(exit_code=-1, status="TIMED_OUT")

    result = sandbox.run_command_shell("sleep 30", timeout=2)

    assert result.is_success is False
    assert "timed out" in result.error.lower()


def test_run_command_shell_reports_transport_errors_without_raising():
    sandbox, client = make_sandbox()
    client.invoke_command.side_effect = RuntimeError("network gone")

    result = sandbox.run_command_shell("echo hi")

    assert result.is_success is False
    assert "network gone" in result.error


def test_run_command_shell_background_detaches():
    sandbox, client = make_sandbox()

    result = sandbox.run_command_shell("sleep 100", run_in_background_enabled=True)

    assert result.background is True
    assert any("nohup" in cmd and "&" in cmd for cmd in commands_sent(client))


# --- retries -----------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [AgentCoreRuntimeConflictError("provisioning"), AgentCoreRuntimeThrottlingError("slow down")]
)
def test_transient_errors_are_retried(error):
    """409 fires while a session provisions — the most likely first-call flake."""
    sandbox, client = make_started_sandbox()
    sandbox.creation_error_handling.initial_wait_seconds = 0
    sandbox.creation_error_handling.max_wait_seconds = 0
    sandbox.creation_error_handling.jitter = 0
    client.invoke_command.side_effect = [error, ok(stdout="recovered")]

    result = sandbox.run_command_shell("echo hi")

    assert result.stdout == "recovered"
    assert client.invoke_command.call_count == 2


def test_retries_are_bounded():
    sandbox, client = make_started_sandbox()
    sandbox.creation_error_handling.max_retries = 3
    sandbox.creation_error_handling.initial_wait_seconds = 0
    sandbox.creation_error_handling.max_wait_seconds = 0
    sandbox.creation_error_handling.jitter = 0
    client.invoke_command.side_effect = AgentCoreRuntimeConflictError("still provisioning")

    result = sandbox.run_command_shell("echo hi")

    assert result.is_success is False
    assert client.invoke_command.call_count == 3


# --- file upload -------------------------------------------------------------


def test_upload_file_inline_encodes_base64():
    sandbox, client = make_sandbox()

    path = sandbox.upload_file("notes.txt", b"hello world")

    assert path == "/mnt/workspace/notes.txt"
    upload_cmd = next(cmd for cmd in commands_sent(client) if "base64 -d" in cmd)
    assert base64.b64encode(b"hello world").decode() in upload_cmd
    assert "/mnt/workspace/notes.txt" in upload_cmd


def test_upload_file_creates_parent_directories():
    sandbox, client = make_sandbox()

    sandbox.upload_file("f.txt", b"x", destination_path="/mnt/workspace/deep/nested/f.txt")

    assert any("mkdir -p" in cmd and "/mnt/workspace/deep/nested" in cmd for cmd in commands_sent(client))


def test_upload_file_raises_on_command_failure():
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(stderr="No space left on device", exit_code=1)

    with pytest.raises(RuntimeError, match="No space left on device"):
        sandbox.upload_file("f.txt", b"x")


def test_upload_file_switches_to_s3_above_the_threshold():
    sandbox, client = make_sandbox(inline_transfer_max_bytes=10, s3_relay_bucket="relay-bucket")
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://relay-bucket.s3/presigned-get"
    sandbox._s3_client = s3

    sandbox.upload_file("big.bin", b"y" * 50)

    s3.put_object.assert_called_once()
    assert s3.put_object.call_args.kwargs["Bucket"] == "relay-bucket"
    assert any("curl" in cmd and "presigned-get" in cmd for cmd in commands_sent(client))
    s3.delete_object.assert_called_once()


def test_upload_file_large_without_relay_bucket_explains_why():
    sandbox, _ = make_sandbox(inline_transfer_max_bytes=10)

    with pytest.raises(ValueError, match="s3_relay_bucket"):
        sandbox.upload_file("big.bin", b"y" * 50)


def test_upload_file_cleans_up_the_relay_object_on_failure():
    sandbox, client = make_started_sandbox(inline_transfer_max_bytes=10, s3_relay_bucket="relay-bucket")
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://relay/get"
    sandbox._s3_client = s3
    client.invoke_command.side_effect = [ok(), ok(stderr="curl failed", exit_code=22)]

    with pytest.raises(RuntimeError):
        sandbox.upload_file("big.bin", b"y" * 50)

    s3.delete_object.assert_called_once()


def test_store_routes_through_upload():
    """FileStore-compatible write interface must work against this backend."""
    sandbox, client = make_sandbox()

    info = sandbox.store("report.md", "# title")

    assert info.path == "/mnt/workspace/report.md"
    assert info.size == len(b"# title")
    assert any("base64 -d" in cmd for cmd in commands_sent(client))


# --- file retrieval ----------------------------------------------------------


def test_retrieve_inline_decodes_base64():
    sandbox, client = make_started_sandbox()
    payload = base64.b64encode(b"file contents").decode()
    client.invoke_command.side_effect = [ok(stdout="13\n"), ok(stdout=payload)]

    assert sandbox.retrieve("notes.txt") == b"file contents"


def test_retrieve_raises_file_not_found_when_stat_fails():
    sandbox, client = make_started_sandbox()
    client.invoke_command.return_value = ok(stderr="No such file", exit_code=1)

    with pytest.raises(FileNotFoundError):
        sandbox.retrieve("missing.txt")


def test_retrieve_raises_on_malformed_base64():
    sandbox, client = make_started_sandbox()
    client.invoke_command.side_effect = [ok(stdout="10\n"), ok(stdout="not base64!!")]

    with pytest.raises(FileNotFoundError):
        sandbox.retrieve("corrupt.bin")


def test_retrieve_switches_to_s3_above_the_threshold():
    sandbox, client = make_started_sandbox(inline_transfer_max_bytes=10, s3_relay_bucket="relay-bucket")
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://relay-bucket.s3/presigned-put"
    body = MagicMock()
    body.read.return_value = b"z" * 50
    s3.get_object.return_value = {"Body": body}
    sandbox._s3_client = s3
    client.invoke_command.side_effect = [ok(stdout="50\n"), ok()]

    assert sandbox.retrieve("big.bin") == b"z" * 50
    assert any("curl" in cmd and "upload-file" in cmd for cmd in commands_sent(client))
    s3.delete_object.assert_called_once()


def test_retrieve_large_without_relay_bucket_explains_why():
    sandbox, client = make_started_sandbox(inline_transfer_max_bytes=10)
    client.invoke_command.return_value = ok(stdout="5000\n")

    with pytest.raises(ValueError, match="s3_relay_bucket"):
        sandbox.retrieve("big.bin")


# --- listing / existence -----------------------------------------------------


def test_list_files_parses_find_output():
    sandbox, client = make_started_sandbox()
    client.invoke_command.return_value = ok(stdout="/mnt/workspace/a.txt\n/mnt/workspace/b.txt\n\n")

    assert sandbox.list_files() == ["/mnt/workspace/a.txt", "/mnt/workspace/b.txt"]


def test_list_files_respects_max_output_files():
    sandbox, client = make_started_sandbox(max_output_files=2)
    client.invoke_command.return_value = ok(stdout="a\nb\nc\nd\n")

    assert len(sandbox.list_files()) == 2


def test_list_files_returns_empty_on_error():
    sandbox, client = make_started_sandbox()
    client.invoke_command.side_effect = RuntimeError("boom")

    assert sandbox.list_files() == []


def test_exists_true_on_zero_exit():
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(exit_code=0)
    assert sandbox.exists("there.txt") is True


def test_exists_false_on_nonzero_exit():
    sandbox, client = make_sandbox()
    client.invoke_command.return_value = ok(exit_code=1)
    assert sandbox.exists("missing.txt") is False


def test_exists_false_on_transport_error():
    sandbox, client = make_sandbox()
    client.invoke_command.side_effect = RuntimeError("network gone")
    assert sandbox.exists("unknown.txt") is False


def test_delete_file_uses_rm_and_reports_success():
    sandbox, client = make_sandbox()

    assert sandbox.delete_file("stale.txt") is True
    assert any("rm -f" in cmd and "/mnt/workspace/stale.txt" in cmd for cmd in commands_sent(client))


# --- metadata / lifecycle ----------------------------------------------------


def test_get_sandbox_info_reports_no_public_url_for_a_port():
    sandbox, _ = make_sandbox()
    sandbox.ensure_started()

    info = sandbox.get_sandbox_info(port=8080)

    assert info.public_url is None
    assert "no public ports" in info.public_url_error
    assert info.sandbox_id == sandbox.session_id


def test_get_sandbox_info_without_a_port_has_no_url_error():
    sandbox, _ = make_sandbox()
    assert sandbox.get_sandbox_info().public_url_error is None


def test_close_without_kill_keeps_the_session():
    sandbox, client = make_sandbox()
    session_id = sandbox.ensure_started()

    sandbox.close()

    client.stop_session.assert_not_called()
    assert sandbox.session_id == session_id


def test_close_with_kill_stops_the_session():
    sandbox, client = make_sandbox(qualifier="v2")
    session_id = sandbox.ensure_started()

    sandbox.close(kill=True)

    client.stop_session.assert_called_once_with(ARN, session_id, qualifier="v2")
    assert sandbox.session_id is None


def test_close_is_safe_before_start():
    sandbox, client = make_sandbox()
    sandbox.close(kill=True)
    client.stop_session.assert_not_called()


def test_close_swallows_stop_errors():
    sandbox, client = make_sandbox()
    sandbox.ensure_started()
    client.stop_session.side_effect = RuntimeError("already gone")

    sandbox.close(kill=True)


def test_context_manager_starts_and_closes():
    sandbox, client = make_sandbox()

    with sandbox as entered:
        assert entered.session_id is not None

    client.stop_session.assert_not_called()


# --- serialization -----------------------------------------------------------


def test_to_dict_excludes_connection_secrets():
    sandbox, _ = make_sandbox()
    data = sandbox.to_dict()

    assert data["agent_runtime_arn"] == ARN
    assert data["type"].endswith("BedrockAgentCoreRuntimeSandbox")


def test_get_tools_returns_shell_and_file_tools():
    sandbox, _ = make_sandbox()
    names = {type(tool).__name__ for tool in sandbox.get_tools()}
    assert "SandboxShellTool" in names
    assert "FileWriteTool" in names
