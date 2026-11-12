"""Live tests for ``BoxdSandbox`` and ``BoxdInterpreterTool`` against real boxd machines.

Requires:
- ``BOXD_API_KEY``. ``BOXD_ORG`` and ``BOXD_REGION`` are optional.
- ``pip install "dynamiq[boxd]"``.

Optional:
- ``BOXD_TEST_NAME_PREFIX``: prefix of the machine names the tests create (default ``dynamiq-e2e-``).

The module's machine is deleted at the end, and the interpreter tool deletes its own machine after each call.
"""

import hashlib
import os
import time
import uuid

import pytest

from dynamiq.connections import Boxd as BoxdConnection
from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema

pytestmark = pytest.mark.integration

NAME_PREFIX = os.getenv("BOXD_TEST_NAME_PREFIX", "dynamiq-e2e-")


@pytest.fixture(scope="module")
def connection() -> BoxdConnection:
    if not os.getenv("BOXD_API_KEY"):
        pytest.skip("BOXD_API_KEY is not set; skipping live boxd tests.")
    pytest.importorskip("boxd")
    return BoxdConnection()


@pytest.fixture(scope="module")
def sandbox(connection):
    from dynamiq.sandboxes.boxd import BoxdSandbox

    sandbox = BoxdSandbox(connection=connection, name_prefix=NAME_PREFIX, envs={"DYNAMIQ_E2E": "on"})
    sandbox.ensure_started()
    try:
        yield sandbox
    finally:
        machine_id = sandbox.current_sandbox_id
        sandbox.close(kill=True)
        from boxd import NotFoundError

        with pytest.raises(NotFoundError):
            connection.get_client().machines.get(machine_id)


def test_shell_reports_exit_code_streams_and_env(sandbox):
    result = sandbox.run_command_shell("pwd; echo out; echo err >&2; sh -c 'echo $DYNAMIQ_E2E'; exit 3")

    assert result.exit_code == 3
    assert result.stdout.splitlines() == [sandbox.base_path, "out", "on"]
    assert result.stderr == "err\n"


def test_the_machine_is_isolated(connection, sandbox):
    machine = connection.get_client().machines.get(sandbox.current_sandbox_id)

    assert machine.networking.isolated is True


def test_the_machine_never_goes_to_standby_and_has_no_delete_timer(connection, sandbox):
    machine = connection.get_client().machines.get(sandbox.current_sandbox_id)

    assert machine.idle.suspend_after == 0
    assert machine.delete_at is None


def test_pip_installs_into_the_python_env(sandbox):
    result = sandbox.run_command_shell("pip install -q six && python -c 'import six; print(six.__version__)'", 180)

    assert result.exit_code == 0, result.stderr
    assert result.stdout.strip()


def test_a_large_file_round_trips(sandbox):
    content = os.urandom(5 * 1024 * 1024)

    path = sandbox.upload_file("big.bin", content, destination_path=f"{sandbox.base_path}/nested/dir/big.bin")

    assert sandbox.exists("nested/dir/big.bin")
    assert not sandbox.exists("nested/dir/missing.bin")
    assert path in sandbox.list_files()
    assert hashlib.sha256(sandbox.retrieve(path)).hexdigest() == hashlib.sha256(content).hexdigest()


def test_a_background_process_outlives_the_call(sandbox):
    result = sandbox.run_command_shell("python -m http.server 8765", run_in_background_enabled=True)
    assert result.background is True

    for _ in range(20):
        probe = sandbox.run_command_shell("curl -s -o /dev/null -w '%{http_code}' localhost:8765")
        if probe.stdout == "200":
            break
        time.sleep(0.5)
    assert probe.stdout == "200"


def test_a_command_past_its_timeout_returns_an_error(sandbox):
    started = time.monotonic()

    result = sandbox.run_command_shell("sleep 60", timeout=3)

    assert result.error and "timed out" in result.error
    assert time.monotonic() - started < 30


def test_a_port_gets_a_public_url(sandbox):
    info = sandbox.get_sandbox_info(port=8765)

    assert info.public_url and info.public_url.startswith("https://p8765.")
    assert info.public_url_error is None


def test_a_view_shares_the_machine_with_its_own_directory(sandbox):
    view = sandbox.create_view(f"{sandbox.base_path}/work/agent-{uuid.uuid4().hex[:6]}")

    pwd = view.run_command_shell("pwd")
    view.close(kill=False)

    assert view.current_sandbox_id == sandbox.current_sandbox_id
    assert pwd.stdout.strip() == view.base_path
    assert sandbox.run_command_shell("true").exit_code == 0


def test_reconnecting_starts_a_stopped_machine(connection, sandbox):
    from dynamiq.sandboxes.boxd import BoxdSandbox

    sandbox.upload_file("marker.txt", b"kept")
    connection.get_client().machines.stop(sandbox.current_sandbox_id)

    reconnected = BoxdSandbox(connection=connection, sandbox_id=sandbox.current_sandbox_id)
    result = reconnected.run_command_shell("cat marker.txt")

    assert result.stdout == "kept"


def test_the_interpreter_runs_python_with_packages_and_deletes_its_machine(connection):
    from dynamiq.nodes.tools.boxd_sandbox import BoxdInterpreterTool

    run_label = {"dynamiq-e2e-run": uuid.uuid4().hex[:12]}
    tool = BoxdInterpreterTool(
        connection=connection,
        name_prefix=NAME_PREFIX,
        labels=run_label,
        persistent_sandbox=False,
        is_optimized_for_agents=False,
    )
    code = (
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, 2, 3]})\n"
        "df.to_csv('/home/boxd/output/sum.csv')\n"
        "print(int(df['a'].sum()))\n"
    )

    result = tool.execute(CodeInterpreterInputSchema(packages="pandas", python=code))

    assert result["content"]["code_execution"].strip() == "6"
    assert "/home/boxd/output/sum.csv" in result["content"]["files"]
    assert list(connection.get_client().machines.list(org=connection.org, labels=run_label)) == []
