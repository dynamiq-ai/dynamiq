"""BoxdInterpreterTool against an in-memory boxd client."""

import base64
import io
import sys

import pytest

from dynamiq.connections import Boxd as BoxdConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema
from tests.helpers.boxd import ExecResult, FakeBoxd, FakeBoxdClient, boxd_sdk_module


@pytest.fixture
def fake_boxd(monkeypatch):
    monkeypatch.setitem(sys.modules, "boxd", boxd_sdk_module())
    fake = FakeBoxd()
    monkeypatch.setattr(BoxdConnection, "get_client", lambda self: FakeBoxdClient(fake))
    return fake


@pytest.fixture
def make_tool(fake_boxd):
    from dynamiq.nodes.tools.boxd_sandbox import BoxdInterpreterTool

    def make(**kwargs):
        kwargs.setdefault("persistent_sandbox", True)
        kwargs.setdefault("is_optimized_for_agents", False)
        return BoxdInterpreterTool(connection=BoxdConnection(api_key="test-key"), **kwargs)

    return make


def answer(handlers: dict[str, ExecResult]):
    """Answer a command with the result of the first handler whose key is in its script."""

    def handler(call):
        for needle, result in handlers.items():
            if needle in call.script:
                return result
        return ExecResult()

    return handler


def test_a_persistent_tool_creates_one_isolated_machine_up_front(fake_boxd, make_tool):
    make_tool(labels={"team": "data"}, name_prefix="t-")

    (machine,) = fake_boxd.created
    assert machine.create_kwargs["isolated"] is True
    assert machine.create_kwargs["labels"] == {"team": "data"}
    assert machine.name.startswith("t-")


def test_python_runs_as_a_script_in_the_base_path_and_returns_stdout(fake_boxd, make_tool):
    tool = make_tool()
    fake_boxd.exec_handler = answer({"python /tmp/dynamiq-": ExecResult(stdout="42\n")})

    result = tool.execute(CodeInterpreterInputSchema(python="print(6 * 7)", params={"factor": 7}))

    assert result["content"]["code_execution"] == "42\n"
    ((machine_id, script_path), code) = [(key, value) for key, value in fake_boxd.files.items()][0]
    assert script_path.startswith("/tmp/dynamiq-") and script_path.endswith(".py")
    assert code.decode().startswith("\n# Tool params variables\nfactor = 7\n")
    run = next(call for call in fake_boxd.execs if script_path in call.script)
    assert run.cwd == "/home/boxd"
    assert f"rm -f {script_path}" in run.script


def test_a_python_error_is_recoverable_and_carries_the_traceback_tail(fake_boxd, make_tool):
    tool = make_tool()
    traceback = "Traceback (most recent call last):\n  ...\nNameError: name 'x' is not defined\n"
    fake_boxd.exec_handler = answer({"python /tmp/dynamiq-": ExecResult(stderr=traceback, exit_code=1)})

    with pytest.raises(ToolExecutionException, match="NameError: name 'x' is not defined") as error:
        tool.execute(CodeInterpreterInputSchema(python="print(x)"))

    assert error.value.recoverable is True


def test_python_stderr_of_a_successful_run_is_labelled(fake_boxd, make_tool):
    tool = make_tool()
    fake_boxd.exec_handler = answer({"python /tmp/dynamiq-": ExecResult(stdout="done", stderr="a warning")})

    result = tool.execute(CodeInterpreterInputSchema(python="print('done')"))

    assert result["content"]["code_execution"] == "done\n[stderr] a warning"


def test_shell_commands_get_env_cwd_and_timeout(fake_boxd, make_tool):
    tool = make_tool()
    fake_boxd.exec_handler = answer({"echo $GREETING": ExecResult(stdout="hello\n")})

    result = tool.execute(
        CodeInterpreterInputSchema(shell_command="echo $GREETING", env={"GREETING": "hello"}, cwd="/srv", timeout=90)
    )

    assert result["content"]["shell_command_execution"] == "hello\n"
    call = next(call for call in fake_boxd.execs if "echo $GREETING" in call.script)
    assert (call.env, call.cwd, call.timeout) == ({"GREETING": "hello"}, "/srv", 90)
    assert call.command[:2] == ["bash", "-c"]


def test_shell_commands_run_in_the_output_directory_by_default(fake_boxd, make_tool):
    tool = make_tool()

    tool.execute(CodeInterpreterInputSchema(shell_command="ls"))

    assert fake_boxd.execs[-2].cwd == "/home/boxd/output"


def test_a_failing_shell_command_reports_its_exit_code_and_streams(fake_boxd, make_tool):
    tool = make_tool()
    fake_boxd.exec_handler = answer({"pytest": ExecResult(stdout="1 failed", stderr="boom", exit_code=2)})

    with pytest.raises(ToolExecutionException, match="exit code 2. stderr: boom stdout: 1 failed"):
        tool.execute(CodeInterpreterInputSchema(shell_command="pytest"))


def test_packages_install_into_the_python_env_with_uv(fake_boxd, make_tool):
    tool = make_tool()

    result = tool.execute(CodeInterpreterInputSchema(packages="pandas>=2.0, numpy", shell_command="ls"))

    assert result["content"]["packages_installation"] == "Installed packages: pandas>=2.0, numpy"
    install = next(script for script in fake_boxd.scripts() if "pip install" in script)
    assert install.splitlines()[-1] == 'uv pip install -q "pandas>=2.0" "numpy"'
    assert 'PATH="/opt/dynamiq/venv/bin:$PATH"' in install


def test_a_failed_install_is_recoverable(fake_boxd, make_tool):
    tool = make_tool()
    fake_boxd.exec_handler = answer({"uv pip install": ExecResult(stderr="No solution found", exit_code=1)})

    with pytest.raises(ToolExecutionException, match="No solution found"):
        tool.execute(CodeInterpreterInputSchema(packages="no-such-package", shell_command="ls"))


def test_files_are_uploaded_to_the_input_directory_and_downloaded_as_base64(fake_boxd, make_tool):
    tool = make_tool()
    upload = io.BytesIO(b"a,b\n1,2\n")
    upload.name = "data/table.csv"
    machine_id = fake_boxd.created[0].id
    fake_boxd.files[(machine_id, "/home/boxd/output/result.txt")] = b"result"

    result = tool.execute(
        CodeInterpreterInputSchema(
            files=[upload], shell_command="cat input/data/table.csv", download_files=["/home/boxd/output/result.txt"]
        )
    )

    assert fake_boxd.files[(machine_id, "/home/boxd/input/data/table.csv")] == b"a,b\n1,2\n"
    assert result["content"]["files"]["/home/boxd/output/result.txt"] == base64.b64encode(b"result").decode()


def test_a_non_persistent_tool_deletes_its_machine_after_each_call(fake_boxd, make_tool):
    tool = make_tool(persistent_sandbox=False)
    assert fake_boxd.created == []

    tool.execute(CodeInterpreterInputSchema(shell_command="ls"))
    tool.execute(CodeInterpreterInputSchema(shell_command="ls"))

    assert len(fake_boxd.created) == 2
    assert fake_boxd.deleted == [machine.id for machine in fake_boxd.created]


def test_the_machine_is_deleted_when_the_command_fails(fake_boxd, make_tool):
    tool = make_tool(persistent_sandbox=False)
    fake_boxd.exec_handler = answer({"false": ExecResult(exit_code=1)})

    with pytest.raises(ToolExecutionException):
        tool.execute(CodeInterpreterInputSchema(shell_command="false"))

    assert fake_boxd.deleted == [fake_boxd.created[0].id]


def test_close_deletes_the_persistent_machine(fake_boxd, make_tool):
    tool = make_tool()

    tool.close()

    assert fake_boxd.deleted == [fake_boxd.created[0].id]


def test_restoring_a_checkpoint_reconnects_to_the_stopped_machine(fake_boxd, make_tool):
    tool = make_tool()
    state = tool.to_checkpoint_state()
    machine_id = state.sandbox_id
    fake_boxd.machines[machine_id].status = "stopped"

    restored = make_tool(persistent_sandbox=True)
    restored.from_checkpoint_state(state)
    restored.execute(CodeInterpreterInputSchema(shell_command="ls"))

    assert ("start", machine_id) in fake_boxd.calls
    assert fake_boxd.execs[-1].machine_id == machine_id


def test_the_description_does_not_promise_state_between_python_runs(make_tool):
    description = make_tool(persistent_sandbox=False).description

    assert "persist between" not in description
    assert "stateful" not in description
    assert "new Python process" in description
    assert "/home/boxd/input" in description and "/home/user" not in description


def test_the_description_follows_a_custom_base_path(make_tool):
    description = make_tool(persistent_sandbox=False, base_path="/srv/agent").description

    assert "/srv/agent/output" in description
    assert "/home/boxd" not in description


def test_tracing_output_leaves_out_the_api_key(make_tool):
    traced = make_tool(persistent_sandbox=False).to_dict(for_tracing=True)

    assert "test-key" not in str(traced)


def test_the_tool_survives_a_yaml_round_trip(fake_boxd, make_tool, tmp_path):
    from dynamiq import Workflow
    from dynamiq.flows import Flow
    from dynamiq.nodes.tools.boxd_sandbox import BoxdInterpreterTool

    tool = make_tool(
        id="boxd-tool",
        persistent_sandbox=False,
        snapshot="python-data",
        labels={"team": "data"},
        name_prefix="agent-",
        vcpu=4,
        memory="16G",
        auto_suspend_timeout=300,
        delete_after=1800,
    )
    path = tmp_path / "workflow.yaml"
    Workflow(id="wf", flow=Flow(id="flow", nodes=[tool])).to_yaml_file(str(path))

    loaded = Workflow.from_yaml_file(str(path), init_components=True)
    roundtrip_path = tmp_path / "roundtrip.yaml"
    loaded.to_yaml_file(str(roundtrip_path))
    (restored,) = Workflow.from_yaml_file(str(roundtrip_path), init_components=True).flow.nodes

    assert isinstance(restored, BoxdInterpreterTool)
    assert (restored.snapshot, restored.labels, restored.name_prefix) == ("python-data", {"team": "data"}, "agent-")
    assert (restored.vcpu, restored.memory, restored.auto_suspend_timeout) == (4, "16G", 300)
    assert restored.delete_after == 1800
    assert isinstance(restored.connection, BoxdConnection)
    restored.execute(CodeInterpreterInputSchema(shell_command="ls"))
    assert fake_boxd.created[-1].create_kwargs["from_snapshot"] == "python-data"
    assert fake_boxd.created[-1].create_kwargs["delete_after"] == 1800


@pytest.mark.parametrize("persistent", [False, True])
def test_a_machine_whose_default_packages_fail_to_install_is_deleted(fake_boxd, make_tool, persistent):
    fake_boxd.exec_handler = answer({"uv pip install": ExecResult(stderr="No solution found", exit_code=1)})

    with pytest.raises(ToolExecutionException):
        tool = make_tool(persistent_sandbox=persistent, installed_packages=["no-such-package"])
        tool.execute(CodeInterpreterInputSchema(shell_command="ls"))

    assert fake_boxd.created
    assert fake_boxd.deleted == [machine.id for machine in fake_boxd.created]


def test_no_command_runs_without_a_time_limit(fake_boxd, make_tool):
    tool = make_tool(timeout=120)
    fake_boxd.exec_handler = answer({"python /tmp/dynamiq-": ExecResult(stdout="1")})

    tool.execute(CodeInterpreterInputSchema(packages="six", python="print(1)", shell_command="ls"))
    tool.execute(CodeInterpreterInputSchema(shell_command="make", timeout=900))

    assert all(call.timeout is not None for call in fake_boxd.execs)
    assert {call.timeout for call in fake_boxd.execs if call.script.endswith("\nmake")} == {900}
    assert {call.timeout for call in fake_boxd.execs if "pip install" in call.script} == {120}


def test_new_machines_never_go_to_standby_by_default(fake_boxd, make_tool):
    make_tool()

    assert fake_boxd.created[0].create_kwargs["auto_suspend_timeout"] == 0


@pytest.mark.parametrize(
    ("persistent", "timeout", "delete_after", "expected"),
    [
        (False, 600, None, 3600),
        (False, 1200, None, 4800),
        (True, 600, None, None),
        (False, 600, 900, 900),
        (True, 600, 900, 900),
        (False, 600, 0, None),
    ],
)
def test_a_machine_for_a_single_call_gets_a_delete_timer(
    fake_boxd, make_tool, persistent, timeout, delete_after, expected
):
    tool = make_tool(persistent_sandbox=persistent, timeout=timeout, delete_after=delete_after)
    tool.execute(CodeInterpreterInputSchema(shell_command="ls"))

    assert fake_boxd.created[0].create_kwargs["delete_after"] == expected
