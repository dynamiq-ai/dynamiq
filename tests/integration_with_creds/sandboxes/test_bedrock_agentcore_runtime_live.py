"""Live integration tests for ``BedrockAgentCoreRuntimeSandbox`` against real AWS.

Requires:
- AWS credentials resolvable by boto3 (env vars, ``AWS_DEFAULT_PROFILE``, SSO, IMDS…) with
  ``bedrock-agentcore:InvokeAgentRuntimeCommand`` and ``bedrock-agentcore:StopRuntimeSession``.
- ``AWS_DEFAULT_REGION`` (or a region on the resolved profile).
- ``BEDROCK_AGENTCORE_RUNTIME_ARN`` — ARN of a provisioned agent runtime. Its container
  must serve the AgentCore HTTP contract or every invocation returns a 500 — the control
  plane reports READY even when nothing answers.
- ``BEDROCK_AGENTCORE_TESTS=1`` opt-in — AWS credentials alone do not imply AgentCore is
  enabled for the account/region.

Optional:
- ``BEDROCK_AGENTCORE_RELAY_BUCKET`` — an S3 bucket for the large-file relay. Without it the
  large-transfer tests skip (they cannot be faked: the point is the >64 KB path).
- ``BEDROCK_AGENTCORE_QUALIFIER`` — endpoint alias; defaults to the runtime's default endpoint.

Sessions cost nothing while idle and self-terminate at the runtime's idle timeout, so a
crashed test leaves no billable residue.
"""

import concurrent.futures
import os
import uuid

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.sandboxes.bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox
from dynamiq.sandboxes.bedrock_agentcore_runtime_client import MAX_COMMAND_BYTES, CommandTooLargeError

BASE_PATH = "/mnt/workspace/dynamiq-e2e"
COMMAND_TIMEOUT = 120


def _require_agentcore_runtime() -> tuple[AWSConnection, str]:
    """Skip unless opted in and credentials, region and a runtime ARN all resolve."""
    if not os.getenv("BEDROCK_AGENTCORE_TESTS"):
        pytest.skip("BEDROCK_AGENTCORE_TESTS is not set")

    runtime_arn = os.getenv("BEDROCK_AGENTCORE_RUNTIME_ARN")
    if not runtime_arn:
        pytest.skip("BEDROCK_AGENTCORE_RUNTIME_ARN is not set (needs a provisioned agent runtime)")

    connection = AWSConnection()
    session = connection.get_boto3_session()
    # Resolve rather than reading env vars, so SSO / profiles / IMDS are all supported.
    if session.get_credentials() is None:
        pytest.skip("no AWS credentials could be resolved")
    if not (connection.region or session.region_name):
        pytest.skip("no AWS region configured (set AWS_DEFAULT_REGION)")
    return connection, runtime_arn


@pytest.fixture(scope="module")
def runtime() -> tuple[AWSConnection, str]:
    return _require_agentcore_runtime()


@pytest.fixture(scope="module")
def relay_bucket() -> str | None:
    return os.getenv("BEDROCK_AGENTCORE_RELAY_BUCKET")


def make_sandbox(runtime, base_path=BASE_PATH, **kwargs) -> BedrockAgentCoreRuntimeSandbox:
    connection, runtime_arn = runtime
    return BedrockAgentCoreRuntimeSandbox(
        connection=connection,
        agent_runtime_arn=runtime_arn,
        qualifier=os.getenv("BEDROCK_AGENTCORE_QUALIFIER") or None,
        base_path=base_path,
        timeout=COMMAND_TIMEOUT,
        **kwargs,
    )


@pytest.fixture(scope="module")
def sandbox(runtime):
    """One shared session for the read-mostly tests; stopped at the end."""
    sb = make_sandbox(runtime)
    try:
        yield sb
    finally:
        sb.close(kill=True)


# --- command execution -------------------------------------------------------


def test_session_is_created_by_the_first_command(runtime):
    """No InvokeAgentRuntime bootstrap: the first command creates the session (spike #1)."""
    sb = make_sandbox(runtime)
    try:
        result = sb.run_command_shell("echo created")
        assert result.is_success
        assert result.stdout.strip() == "created"
        assert sb.session_id and len(sb.session_id) >= 33
    finally:
        sb.close(kill=True)


def test_compound_command_is_shell_interpreted(sandbox):
    """Regression guard: unwrapped, this returns the literal 'one; echo two'."""
    result = sandbox.run_command_shell("echo one; echo two")
    assert result.is_success
    assert result.stdout.split() == ["one", "two"]


def test_pipes_globs_and_expansion_work(sandbox):
    assert sandbox.run_command_shell("echo hello | tr a-z A-Z").stdout.strip() == "HELLO"
    assert sandbox.run_command_shell("echo $HOME").stdout.strip().startswith("/")
    assert "hostname" in sandbox.run_command_shell("ls /etc/hostna*").stdout


def test_exit_code_is_reported(sandbox):
    result = sandbox.run_command_shell("exit 42")
    assert result.exit_code == 42
    assert result.is_success is False


def test_missing_binary_returns_127(sandbox):
    """Unwrapped this is an opaque 'ctr: OCI runtime exec failed' instead of 127."""
    result = sandbox.run_command_shell("definitely-not-a-binary")
    assert result.exit_code == 127


def test_stdout_and_stderr_are_separated(sandbox):
    result = sandbox.run_command_shell("echo to-out; echo to-err >&2")
    assert result.stdout.strip() == "to-out"
    assert (result.stderr or "").strip() == "to-err"


def test_command_timeout_is_enforced(sandbox):
    result = sandbox.run_command_shell("sleep 30", timeout=2)
    assert result.is_success is False
    assert "timed out" in (result.error or "").lower()


def test_runs_as_root_on_arm64(sandbox):
    assert sandbox.run_command_shell("id -u").stdout.strip() == "0"
    assert sandbox.run_command_shell("uname -m").stdout.strip() == "aarch64"


def test_shell_state_does_not_leak_between_commands(sandbox):
    """Each command is a fresh process, which is why env must be composed per command."""
    sandbox.run_command_shell("export LEAKED=yes")
    assert sandbox.run_command_shell("echo [$LEAKED]").stdout.strip() == "[]"


def test_oversized_command_is_rejected_locally(sandbox):
    result = sandbox.run_command_shell("echo " + "y" * MAX_COMMAND_BYTES)
    assert result.is_success is False
    assert "65536" in (result.error or "")


def test_oversized_command_raises_from_the_client(runtime):
    connection, runtime_arn = runtime
    from dynamiq.sandboxes.bedrock_agentcore_runtime_client import AgentCoreRuntimeClient, derive_session_id

    client = AgentCoreRuntimeClient(connection)
    with pytest.raises(CommandTooLargeError):
        client.invoke_command(runtime_arn, derive_session_id("oversized"), "echo " + "y" * MAX_COMMAND_BYTES)


def test_network_egress_is_available(sandbox):
    """networkMode PUBLIC — the reason a custom interpreter was needed on the CI backend."""
    result = sandbox.run_command_shell("curl -sS -o /dev/null -w '%{http_code}' https://pypi.org/simple/", timeout=60)
    assert result.stdout.strip() == "200"


def test_background_command_detaches(sandbox):
    marker = f"{BASE_PATH}/bg-{uuid.uuid4().hex}.txt"
    result = sandbox.run_command_shell(f"sleep 1 && echo done > {marker}", run_in_background_enabled=True)
    assert result.background is True


# --- file transfer -----------------------------------------------------------


def test_upload_and_retrieve_roundtrip(sandbox):
    content = b"hello from the runtime backend\n"
    path = sandbox.upload_file(f"round-{uuid.uuid4().hex}.txt", content)

    assert path.startswith(BASE_PATH)
    assert sandbox.retrieve(path) == content


def test_upload_creates_parent_directories(sandbox):
    nested = f"{BASE_PATH}/deep/{uuid.uuid4().hex}/nested.txt"
    sandbox.upload_file("nested.txt", b"nested", destination_path=nested)

    assert sandbox.exists(nested)
    assert sandbox.retrieve(nested) == b"nested"


def test_binary_roundtrip_is_byte_exact(sandbox):
    """Transfer goes through base64, so non-UTF-8 bytes must survive intact."""
    content = bytes(range(256)) * 20
    path = sandbox.upload_file(f"binary-{uuid.uuid4().hex}.bin", content)

    assert sandbox.retrieve(path) == content


def test_exists_reports_missing_files(sandbox):
    assert sandbox.exists(f"{BASE_PATH}/definitely-absent-{uuid.uuid4().hex}") is False


def test_list_files_finds_uploads(sandbox):
    name = f"listed-{uuid.uuid4().hex}.txt"
    path = sandbox.upload_file(name, b"x")

    assert path in sandbox.list_files(target_dir=BASE_PATH)


def test_delete_file_removes_it(sandbox):
    path = sandbox.upload_file(f"doomed-{uuid.uuid4().hex}.txt", b"x")
    assert sandbox.exists(path)

    assert sandbox.delete_file(path) is True
    assert sandbox.exists(path) is False


def test_store_and_collect_files(sandbox):
    """FileStore-compatible helpers inherited from the ABC must work over this transport."""
    name = f"stored-{uuid.uuid4().hex}.md"
    info = sandbox.store(name, "# heading")

    collected = sandbox.collect_files(file_paths=[info.path])
    assert len(collected) == 1
    assert collected[0].read() == b"# heading"


def test_retrieve_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        sandbox.retrieve(f"{BASE_PATH}/never-written-{uuid.uuid4().hex}")


def test_large_file_roundtrip_through_s3_relay(runtime, relay_bucket):
    """The >64 KB path: inline base64 cannot carry this, so it must use the relay."""
    if not relay_bucket:
        pytest.skip("BEDROCK_AGENTCORE_RELAY_BUCKET is not set")

    sb = make_sandbox(runtime, s3_relay_bucket=relay_bucket)
    try:
        content = os.urandom(5 * 1024 * 1024)
        path = sb.upload_file(f"large-{uuid.uuid4().hex}.bin", content)
        assert sb.retrieve(path) == content
    finally:
        sb.close(kill=True)


def test_large_upload_without_relay_bucket_explains_itself(sandbox):
    with pytest.raises(ValueError, match="s3_relay_bucket"):
        sandbox.upload_file("too-big.bin", os.urandom(sandbox.inline_transfer_max_bytes + 1))


# --- persistence, views, reconnection ----------------------------------------


def test_workspace_survives_stop_and_resume(runtime):
    """Session storage is flushed by StopRuntimeSession and restored on the next invoke."""
    connection, runtime_arn = runtime
    seed = f"persist-{uuid.uuid4().hex}"
    marker = f"{BASE_PATH}/persisted.txt"

    first = make_sandbox(runtime, session_seed=seed)
    first.upload_file("persisted.txt", b"survived", destination_path=marker)
    session_id = first.session_id
    first.close(kill=True)

    second = make_sandbox(runtime, session_seed=seed)
    try:
        assert second.ensure_started() == session_id
        assert second.retrieve(marker) == b"survived"
    finally:
        second.close(kill=True)


def test_session_id_is_deterministic_across_instances(runtime):
    seed = f"determinism-{uuid.uuid4().hex}"
    assert make_sandbox(runtime, session_seed=seed).ensure_started() == (
        make_sandbox(runtime, session_seed=seed).ensure_started()
    )


def test_views_share_one_session(sandbox):
    """Two views of one session must see each other's writes."""
    assert sandbox.supports_views is True
    view_a = sandbox.create_view(base_path=f"{BASE_PATH}/view-a")
    view_b = sandbox.create_view(base_path=f"{BASE_PATH}/view-a")

    name = f"shared-{uuid.uuid4().hex}.txt"
    view_a.upload_file(name, b"written by a")

    assert view_b.retrieve(name) == b"written by a"
    assert view_a.session_id == view_b.session_id == sandbox.session_id


def test_views_isolate_relative_paths(sandbox):
    """Same session, different base_path: a relative name resolves to different files."""
    view_a = sandbox.create_view(base_path=f"{BASE_PATH}/iso-a")
    view_b = sandbox.create_view(base_path=f"{BASE_PATH}/iso-b")

    view_a.upload_file("same-name.txt", b"from a")
    view_b.upload_file("same-name.txt", b"from b")

    assert view_a.retrieve("same-name.txt") == b"from a"
    assert view_b.retrieve("same-name.txt") == b"from b"


def test_concurrent_commands_on_one_session(sandbox):
    """The platform documents concurrent invocations against a single session."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda i: sandbox.run_command_shell(f"echo concurrent-{i}"), range(5)))

    assert [r.stdout.strip() for r in results] == [f"concurrent-{i}" for i in range(5)]
    assert all(r.is_success for r in results)


# --- metadata ----------------------------------------------------------------


def test_get_sandbox_info_reports_no_public_url(sandbox):
    sandbox.ensure_started()
    info = sandbox.get_sandbox_info(port=8080)

    assert info.public_url is None
    assert info.public_url_error
    assert info.sandbox_id == sandbox.session_id


def test_close_without_kill_keeps_the_session_reusable(runtime):
    sb = make_sandbox(runtime)
    try:
        sb.upload_file("kept.txt", b"kept")
        session_id = sb.session_id
        sb.close()

        assert sb.session_id == session_id
        assert sb.retrieve("kept.txt") == b"kept"
    finally:
        sb.close(kill=True)
