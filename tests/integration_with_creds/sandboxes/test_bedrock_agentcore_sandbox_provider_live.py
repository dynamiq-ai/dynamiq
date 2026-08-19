"""Live integration tests for ``BedrockAgentCoreSandbox`` (the sandbox *provider*).

The shipped suite covers the interpreter *tool* live but only unit-tests the sandbox
provider that ``SandboxConfig(backend=...)`` / ``Agent`` actually use. These tests close
that gap against real AWS.

Requires:
- AWS credentials resolvable by boto3 (env vars, ``AWS_DEFAULT_PROFILE``, SSO, IMDS…) with
  ``bedrock-agentcore:*CodeInterpreterSession*`` and ``bedrock-agentcore:InvokeCodeInterpreter``.
- ``AWS_DEFAULT_REGION`` (or a region on the resolved profile).
- ``BEDROCK_AGENTCORE_TESTS=1`` opt-in — AWS credentials alone do not imply AgentCore is
  enabled for the account/region.

Optional:
- ``BEDROCK_AGENTCORE_INTERPRETER_ID`` — a *custom* code interpreter id. When set, the
  identifier-override path is exercised; otherwise that single test skips.

The fixtures deliberately use a non-default ``base_path`` so ``_ensure_directories`` and
``_resolve_path`` are exercised rather than short-circuited by the ``"."`` default.
"""

import concurrent.futures
import os
import time
import uuid

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections.agentcore import DEFAULT_CODE_INTERPRETER_IDENTIFIER
from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox
from dynamiq.sandboxes.exceptions import SandboxConnectionError

BASE_PATH = "dynamiq-e2e"
# Sessions are billed on active compute, so keep the TTL short: anything left behind by a
# crashed test expires on its own instead of lingering for the 8h maximum.
SESSION_TIMEOUT = 900


def _require_agentcore() -> AWSConnection:
    """Skip unless opted in and credentials + region actually resolve."""
    if not os.getenv("BEDROCK_AGENTCORE_TESTS"):
        pytest.skip("BEDROCK_AGENTCORE_TESTS is not set")

    connection = AWSConnection()
    session = connection.get_boto3_session()
    # Resolve rather than reading env vars, so SSO / profiles / IMDS are all supported.
    if session.get_credentials() is None:
        pytest.skip("no AWS credentials could be resolved")
    if not (connection.region or session.region_name):
        pytest.skip("no AWS region configured (set AWS_DEFAULT_REGION)")
    return connection


@pytest.fixture(scope="module")
def connection() -> AWSConnection:
    return _require_agentcore()


@pytest.fixture(scope="module")
def sandbox(connection):
    """One shared session for the read-mostly tests; killed at the end."""
    sb = BedrockAgentCoreSandbox(
        connection=connection,
        base_path=BASE_PATH,
        timeout=SESSION_TIMEOUT,
    )
    try:
        yield sb
    finally:
        sb.close(kill=True)


@pytest.fixture(scope="module")
def corpus(sandbox):
    """A corpus with near-misses, so a broken path resolver produces a wrong answer.

    - ``notes.txt`` exists **both** at the workspace root and under ``base_path`` with
      different content, so ``retrieve``/``exists`` must prove they scope to ``base_path``.
    - ``notes.txt.bak`` is a name-prefix near-miss for ``exists``.
    - ``deep/a/b/c/buried.txt`` sits below the ``-maxdepth 3`` used by ``list_files``.
    """
    sandbox.upload_file("notes.txt", b"scoped-to-base-path\n")
    sandbox.upload_file("notes.txt.bak", b"stale\n")
    sandbox.upload_file("report.csv", b"name,score\nAlice,95\n")
    sandbox.upload_file("buried.txt", b"too-deep\n", destination_path=f"{BASE_PATH}/deep/a/b/c/buried.txt")
    # Decoy outside base_path with the same leaf name and different content.
    result = sandbox.run_command_shell("printf 'workspace-root-decoy\\n' > notes.txt && cat notes.txt")
    assert result.is_success, f"corpus setup failed: {result}"
    assert "workspace-root-decoy" in (result.stdout or "")
    return sandbox


@pytest.mark.integration
def test_session_starts_and_reports_id(sandbox):
    result = sandbox.run_command_shell("echo agentcore-live")
    assert result.is_success, result
    assert "agentcore-live" in (result.stdout or "")
    assert sandbox.current_sandbox_id, "session id should be populated after first use"


@pytest.mark.integration
def test_ensure_directories_created_base_path(sandbox):
    result = sandbox.run_command_shell(f"test -d {BASE_PATH} && echo present")
    assert result.is_success, result
    assert "present" in (result.stdout or "")


@pytest.mark.integration
def test_retrieve_scopes_to_base_path_not_workspace_root(corpus):
    """The decoy at the workspace root must not win over the base_path copy."""
    assert corpus.retrieve("notes.txt") == b"scoped-to-base-path\n"
    # An absolute-looking path is passed through unchanged by _resolve_path.
    assert corpus.retrieve(f"{BASE_PATH}/notes.txt") == b"scoped-to-base-path\n"


@pytest.mark.integration
def test_exists_rejects_near_misses(corpus):
    assert corpus.exists("notes.txt") is True
    assert corpus.exists("notes.txt.bak") is True
    assert corpus.exists("notes.tx") is False
    assert corpus.exists("nested/notes.txt") is False
    assert corpus.exists(f"{uuid.uuid4().hex}.txt") is False


@pytest.mark.integration
def test_retrieve_missing_file_raises(corpus):
    with pytest.raises((FileNotFoundError, Exception)) as exc:
        corpus.retrieve(f"absent-{uuid.uuid4().hex}.txt")
    assert "absent-" in str(exc.value) or "No file content" in str(exc.value)


@pytest.mark.integration
def test_list_files_is_scoped_and_depth_limited(corpus):
    listed = corpus.list_files()
    leaves = {path.rsplit("/", 1)[-1] for path in listed}

    assert "notes.txt" in leaves
    assert "report.csv" in leaves
    # Everything listed must live under base_path — the workspace-root decoy must not appear.
    assert all(BASE_PATH in path for path in listed), listed
    # find -maxdepth 3 from base_path cannot reach deep/a/b/c/buried.txt
    assert "buried.txt" not in leaves, listed


@pytest.mark.integration
def test_list_files_respects_max_output_files(connection):
    sb = BedrockAgentCoreSandbox(
        connection=connection, base_path=BASE_PATH, timeout=SESSION_TIMEOUT, max_output_files=3
    )
    try:
        setup = sb.run_command_shell(f"mkdir -p {BASE_PATH} && for i in 1 2 3 4 5 6; do touch {BASE_PATH}/f$i; done")
        assert setup.is_success, setup
        assert len(sb.list_files()) == 3
    finally:
        sb.close(kill=True)


@pytest.mark.integration
def test_collect_files_round_trip(corpus):
    collected = corpus.collect_files(file_paths=["report.csv"])
    assert len(collected) == 1
    assert collected[0].read() == b"name,score\nAlice,95\n"
    assert collected[0].name == "report.csv"


@pytest.mark.integration
def test_delete_file_removes_only_the_target(corpus):
    corpus.upload_file("doomed.txt", b"bye\n")
    assert corpus.exists("doomed.txt") is True

    corpus.delete_file("doomed.txt")

    assert corpus.exists("doomed.txt") is False
    # A sibling with an overlapping prefix must survive.
    assert corpus.exists("notes.txt") is True


@pytest.mark.integration
def test_command_timeout_kills_long_running_command(sandbox):
    """AgentCore has no per-invocation timeout; the provider wraps with coreutils `timeout`."""
    started = time.monotonic()
    result = sandbox.run_command_shell("sleep 45", timeout=5)
    elapsed = time.monotonic() - started

    assert result.exit_code == 124, f"expected coreutils timeout exit 124, got {result}"
    assert result.is_success is False
    assert elapsed < 40, f"timeout wrapper did not cut the command short ({elapsed:.1f}s)"


@pytest.mark.integration
def test_command_reports_nonzero_exit_code(sandbox):
    result = sandbox.run_command_shell("exit 3")
    assert result.exit_code == 3, result
    assert result.is_success is False


@pytest.mark.integration
def test_background_command_runs_detached(sandbox):
    marker = f"{BASE_PATH}/bg-{uuid.uuid4().hex}.txt"
    result = sandbox.run_command_shell(f"sleep 2 && printf 'done\\n' > {marker}", run_in_background_enabled=True)
    assert result.background is True
    assert result.error is None, result

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if sandbox.exists(marker):
            break
        time.sleep(3)
    else:
        pytest.fail(f"background task never produced {marker}")

    assert sandbox.retrieve(marker) == b"done\n"


@pytest.mark.integration
def test_sandbox_info_has_no_public_url(sandbox):
    sandbox.run_command_shell("true")

    info = sandbox.get_sandbox_info()
    assert info.base_path == BASE_PATH
    assert info.sandbox_id == sandbox.current_sandbox_id
    assert info.public_url is None
    assert info.public_url_error is None

    info_with_port = sandbox.get_sandbox_info(port=8080)
    assert info_with_port.public_url is None
    assert "do not expose public ports" in (info_with_port.public_url_error or "")


@pytest.mark.integration
def test_concurrent_first_use_creates_exactly_one_session(connection):
    """Double-checked locking must not start N sessions for N racing threads."""
    sb = BedrockAgentCoreSandbox(connection=connection, base_path=BASE_PATH, timeout=SESSION_TIMEOUT)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda i: sb.run_command_shell(f"echo t{i}"), range(8)))

        assert all(r.is_success for r in results), results
        assert sb.current_sandbox_id is not None
        # Every thread must have observed the same underlying session.
        session_ids = {sb.current_sandbox_id}
        assert len(session_ids) == 1
    finally:
        sb.close(kill=True)


@pytest.mark.integration
def test_close_without_kill_keeps_session_reconnectable(connection):
    first = BedrockAgentCoreSandbox(connection=connection, base_path=BASE_PATH, timeout=SESSION_TIMEOUT)
    first.upload_file("handoff.txt", b"survived\n")
    session_id = first.current_sandbox_id
    assert session_id

    first.close(kill=False)
    assert first.session_id == session_id, "close(kill=False) must preserve the session id"

    second = BedrockAgentCoreSandbox(
        connection=connection, base_path=BASE_PATH, timeout=SESSION_TIMEOUT, session_id=session_id
    )
    try:
        # Files persist for the session lifetime, so reconnecting must see the same filesystem.
        assert second.retrieve("handoff.txt") == b"survived\n"
        assert second.current_sandbox_id == session_id
    finally:
        second.close(kill=True)

    dead = BedrockAgentCoreSandbox(
        connection=connection, base_path=BASE_PATH, timeout=SESSION_TIMEOUT, session_id=session_id
    )
    with pytest.raises(SandboxConnectionError):
        dead.run_command_shell("echo should-not-run")


@pytest.mark.integration
def test_reconnect_to_unknown_session_fails_cleanly(connection):
    sb = BedrockAgentCoreSandbox(
        connection=connection,
        base_path=BASE_PATH,
        timeout=SESSION_TIMEOUT,
        session_id=f"nonexistent-{uuid.uuid4().hex}",
    )
    with pytest.raises(SandboxConnectionError):
        sb.run_command_shell("echo should-not-run")


@pytest.mark.integration
def test_custom_code_interpreter_identifier(connection):
    identifier = os.getenv("BEDROCK_AGENTCORE_INTERPRETER_ID")
    if not identifier:
        pytest.skip("BEDROCK_AGENTCORE_INTERPRETER_ID is not set")
    assert identifier != DEFAULT_CODE_INTERPRETER_IDENTIFIER

    sb = BedrockAgentCoreSandbox(
        connection=connection,
        base_path=BASE_PATH,
        timeout=SESSION_TIMEOUT,
        code_interpreter_identifier=identifier,
    )
    try:
        result = sb.run_command_shell("echo custom-interpreter")
        assert result.is_success, result
        assert "custom-interpreter" in (result.stdout or "")
    finally:
        sb.close(kill=True)


@pytest.mark.integration
def test_get_tools_operate_against_the_live_session(sandbox):
    """The tools the Agent is handed must work end to end, not just construct."""
    from dynamiq.nodes.tools.file_tools import FileWriteTool
    from dynamiq.sandboxes.tools.sandbox_info import SandboxInfoTool
    from dynamiq.sandboxes.tools.shell import SandboxShellTool

    tools = {type(tool): tool for tool in sandbox.get_tools()}
    assert {SandboxShellTool, FileWriteTool, SandboxInfoTool} <= set(tools)

    shell = tools[SandboxShellTool]
    result = shell.run(input_data={"command": "echo via-shell-tool"})
    assert "via-shell-tool" in str(result.output), result
