"""BoxdSandbox against an in-memory boxd client."""

import shutil
import subprocess  # nosec B404
import sys

import pytest

from dynamiq.connections import Boxd as BoxdConnection
from tests.helpers.boxd import ExecResult, FakeBoxd, FakeBoxdClient, boxd_sdk_module


@pytest.fixture
def fake_boxd(monkeypatch):
    monkeypatch.setitem(sys.modules, "boxd", boxd_sdk_module())
    fake = FakeBoxd()
    monkeypatch.setattr(BoxdConnection, "get_client", lambda self: FakeBoxdClient(fake))
    return fake


@pytest.fixture
def connection():
    return BoxdConnection(api_key="test-key", org="acme")


@pytest.fixture
def make_sandbox(fake_boxd, connection):
    from dynamiq.sandboxes.boxd import BoxdSandbox

    def make(**kwargs):
        kwargs.setdefault("creation_error_handling", {"initial_wait_seconds": 0, "max_wait_seconds": 0, "jitter": 0})
        return BoxdSandbox(connection=connection, **kwargs)

    return make


def test_a_new_machine_is_isolated_ready_and_has_the_python_env_and_base_path(fake_boxd, make_sandbox):
    sandbox = make_sandbox(labels={"team": "data"}, vcpu=2, memory="8G", auto_suspend_timeout=0, name_prefix="t-")

    machine_id = sandbox.ensure_started()

    (machine,) = fake_boxd.created
    assert machine_id == machine.id == sandbox.current_sandbox_id
    assert machine.name.startswith("t-")
    assert machine.create_kwargs["isolated"] is True
    assert machine.create_kwargs["org"] == "acme"
    assert machine.create_kwargs["labels"] == {"team": "data"}
    assert (machine.create_kwargs["vcpu"], machine.create_kwargs["memory"]) == (2, "8G")
    assert machine.create_kwargs["auto_suspend_timeout"] == 0
    assert ("wait_until_ready", machine.id) in fake_boxd.calls
    scripts = fake_boxd.scripts()
    assert "uv venv" in scripts[0]
    assert fake_boxd.execs[1].command == ["mkdir", "-p", "/home/boxd/workspace"]


def test_machine_names_are_unique(fake_boxd, make_sandbox):
    make_sandbox().ensure_started()
    make_sandbox().ensure_started()

    assert len({machine.name for machine in fake_boxd.created}) == 2


def test_creation_is_retried_on_rate_limit(fake_boxd, make_sandbox):
    fake_boxd.create_failures = [fake_boxd.sdk.RateLimitError("slow down")]

    make_sandbox().ensure_started()

    assert len(fake_boxd.created) == 1


def test_a_machine_that_never_gets_ready_is_deleted(fake_boxd, make_sandbox):
    fake_boxd.ready_error = fake_boxd.sdk.BoxdError("not ready in time")

    with pytest.raises(fake_boxd.sdk.BoxdError):
        make_sandbox().ensure_started()

    assert fake_boxd.deleted == [fake_boxd.created[0].id]


def test_commands_run_in_the_base_path_with_separate_streams(fake_boxd, make_sandbox):
    sandbox = make_sandbox(envs={"FOO": "bar"})
    sandbox.ensure_started()
    fake_boxd.exec_handler = lambda call: ExecResult(stdout="out\n", stderr="err\n", exit_code=3)

    result = sandbox.run_command_shell("make test", timeout=30)

    assert (result.stdout, result.stderr, result.exit_code) == ("out\n", "err\n", 3)
    assert not result.is_success
    call = fake_boxd.execs[-1]
    assert call.cwd == "/home/boxd/workspace"
    assert call.env == {"FOO": "bar"}
    assert call.timeout == 30
    assert call.script.endswith("make test")


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_env_reaches_every_part_of_a_compound_command(fake_boxd, make_sandbox, tmp_path):
    make_sandbox(envs={"FOO": "bar baz"}).run_command_shell("echo A=$FOO; sh -c 'echo B=$FOO'")

    call = fake_boxd.execs[-1]
    # The SDK puts env in front of the command as assignments, which apply to the first program it starts.
    result = subprocess.run(  # nosec B603
        ["env", *(f"{key}={value}" for key, value in call.env.items()), *call.command],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.stdout.splitlines() == ["A=bar baz", "B=bar baz"]


def test_commands_see_the_python_env_first_on_path(fake_boxd, make_sandbox):
    make_sandbox().run_command_shell("pip install six")

    script = fake_boxd.execs[-1].script
    assert 'PATH="/opt/dynamiq/venv/bin:$PATH"' in script.splitlines()[0]


def test_a_failed_call_is_reported_as_an_error(fake_boxd, make_sandbox):
    sandbox = make_sandbox()
    sandbox.ensure_started()

    def fail(call):
        raise fake_boxd.sdk.BoxdError("exec timed out after 3s")

    fake_boxd.exec_handler = fail

    result = sandbox.run_command_shell("sleep 30", timeout=3)

    assert not result.is_success
    assert "timed out" in result.error


def test_a_background_command_is_detached_from_the_call(fake_boxd, make_sandbox):
    result = make_sandbox().run_command_shell("python -m http.server 8000", run_in_background_enabled=True)

    assert result.background is True
    script = fake_boxd.execs[-1].script
    assert "setsid nohup bash -c 'python -m http.server 8000'" in script
    assert script.rstrip().endswith("&")


def test_upload_creates_the_parent_directory_and_retrieve_reads_it_back(fake_boxd, make_sandbox):
    sandbox = make_sandbox()

    path = sandbox.upload_file("data.csv", b"a,b\n1,2\n", destination_path="/home/boxd/workspace/in/data.csv")

    assert path == "/home/boxd/workspace/in/data.csv"
    assert ["mkdir", "-p", "/home/boxd/workspace/in"] in [call.command for call in fake_boxd.execs]
    assert sandbox.retrieve("in/data.csv") == b"a,b\n1,2\n"


def test_upload_defaults_to_the_base_path(make_sandbox):
    assert make_sandbox().upload_file("notes.txt", b"hi") == "/home/boxd/workspace/notes.txt"


def test_list_files_returns_the_paths_find_prints(fake_boxd, make_sandbox):
    sandbox = make_sandbox(max_output_files=7)
    sandbox.ensure_started()
    fake_boxd.exec_handler = lambda call: ExecResult(stdout="/home/boxd/workspace/a.txt\n/home/boxd/workspace/b/c.py\n")

    files = sandbox.list_files()

    assert files == ["/home/boxd/workspace/a.txt", "/home/boxd/workspace/b/c.py"]
    script = fake_boxd.execs[-1].script
    assert "find /home/boxd/workspace -type f" in script and "head -7" in script


def test_exists_follows_the_test_exit_code(fake_boxd, make_sandbox):
    sandbox = make_sandbox()
    sandbox.ensure_started()

    fake_boxd.exec_handler = lambda call: ExecResult(exit_code=0 if call.command[-1].endswith("here.txt") else 1)

    assert sandbox.exists("here.txt") is True
    assert sandbox.exists("missing.txt") is False
    assert fake_boxd.execs[-1].command == ["test", "-f", "/home/boxd/workspace/missing.txt"]


@pytest.mark.parametrize(
    "status, action",
    [("stopped", "start"), ("hibernated", "wake"), ("suspended", "resume"), ("running", None)],
)
def test_reconnecting_brings_the_machine_back_before_the_first_command(fake_boxd, make_sandbox, status, action):
    fake_boxd.add_machine("existing-id", status)
    sandbox = make_sandbox(sandbox_id="existing-id")

    result = sandbox.run_command_shell("ls")

    assert result.exit_code == 0
    actions = [name for name, _ in fake_boxd.calls if name != "wait_until_ready"]
    assert actions == ([action] if action else [])
    assert fake_boxd.created == []


def test_reconnecting_to_a_deleted_machine_raises_a_connection_error(fake_boxd, make_sandbox):
    from dynamiq.sandboxes.exceptions import SandboxConnectionError

    with pytest.raises(SandboxConnectionError, match="boxd sandbox gone-id"):
        make_sandbox(sandbox_id="gone-id").ensure_started()


def test_close_keeps_the_machine_and_close_kill_deletes_it(fake_boxd, make_sandbox):
    sandbox = make_sandbox()
    machine_id = sandbox.ensure_started()

    sandbox.close()
    assert fake_boxd.deleted == []
    assert sandbox.sandbox_id == machine_id

    sandbox.ensure_started()
    sandbox.close(kill=True)
    assert fake_boxd.deleted == [machine_id]
    assert sandbox.sandbox_id is None


def test_a_port_gets_one_public_proxy(fake_boxd, make_sandbox):
    sandbox = make_sandbox()

    first = sandbox.get_sandbox_info(port=3000)
    second = sandbox.get_sandbox_info(port=3000)

    machine = fake_boxd.created[0]
    assert first.public_host == f"p3000.{machine.name}.boxd.sh"
    assert first.public_url == f"https://p3000.{machine.name}.boxd.sh"
    assert second.public_url == first.public_url
    assert [(proxy.name, proxy.port) for proxy in fake_boxd.machines_api.proxies.list(machine.id)] == [("p3000", 3000)]


def test_sandbox_info_without_a_port_has_no_public_url(make_sandbox):
    info = make_sandbox().get_sandbox_info()

    assert info.base_path == "/home/boxd/workspace"
    assert info.public_url is None


def test_views_share_one_machine_and_keep_their_own_working_directory(fake_boxd, make_sandbox):
    owner = make_sandbox()
    owner.ensure_started()
    assert owner.supports_views is True

    view = owner.create_view("/home/boxd/workspace/work/agent-1")
    view.run_command_shell("pwd")
    view.close(kill=False)

    assert len(fake_boxd.created) == 1
    assert view.current_sandbox_id == owner.current_sandbox_id
    assert fake_boxd.execs[-1].cwd == "/home/boxd/workspace/work/agent-1"
    assert fake_boxd.deleted == []


def test_tracing_output_leaves_out_env_and_credentials(make_sandbox):
    sandbox = make_sandbox(envs={"TOKEN": "secret-value"})

    traced = sandbox.to_dict(for_tracing=True)
    saved = sandbox.to_dict()

    assert "envs" not in traced
    assert "secret-value" not in str(traced)
    assert "test-key" not in str(traced)
    assert saved["envs"] == {"TOKEN": "secret-value"}
    assert saved["type"] == "dynamiq.sandboxes.BoxdSandbox"


def test_get_tools_gives_the_agent_a_shell_and_file_tools(make_sandbox):
    tool_names = [tool.name for tool in make_sandbox().get_tools()]

    assert "sandbox-shell" in tool_names
    assert "sandbox-info" in tool_names


def test_a_connection_without_an_api_key_fails_before_calling_boxd(monkeypatch):
    monkeypatch.delenv("BOXD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="BOXD_API_KEY"):
        BoxdConnection().get_client()


def test_a_machine_whose_base_path_cannot_be_created_is_deleted(fake_boxd, make_sandbox):
    fake_boxd.exec_handler = lambda call: (
        ExecResult(stderr="Permission denied", exit_code=1) if call.command[0] == "mkdir" else ExecResult()
    )
    sandbox = make_sandbox(base_path="/root/workspace")

    with pytest.raises(RuntimeError, match="Permission denied"):
        sandbox.ensure_started()

    assert fake_boxd.deleted == [fake_boxd.created[0].id]
    assert sandbox.sandbox_id is None


def test_a_new_machine_never_goes_to_standby_and_has_no_delete_timer_by_default(fake_boxd, make_sandbox):
    make_sandbox().ensure_started()

    (machine,) = fake_boxd.created
    assert machine.create_kwargs["auto_suspend_timeout"] == 0
    assert machine.create_kwargs["delete_after"] is None


@pytest.mark.parametrize(("delete_after", "expected"), [(7200, 7200), (0, None)])
def test_delete_after_reaches_the_new_machine_and_views(fake_boxd, make_sandbox, delete_after, expected):
    sandbox = make_sandbox(delete_after=delete_after)
    sandbox.ensure_started()
    view = sandbox.create_view("/home/boxd/workspace/work/agent-1")

    assert fake_boxd.created[0].create_kwargs["delete_after"] == expected
    assert view.delete_after == delete_after
