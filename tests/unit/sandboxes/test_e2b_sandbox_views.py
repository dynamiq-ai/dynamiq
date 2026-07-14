from unittest.mock import MagicMock, patch

from dynamiq.connections import E2B
from dynamiq.sandboxes.e2b import E2BSandbox


def _sandbox(**kwargs):
    return E2BSandbox(connection=E2B(api_key="test-key"), **kwargs)


def test_create_view_reconnects_same_id_with_new_base_path():
    sb = _sandbox(sandbox_id="sbx-1", base_path="/home/user")
    view = sb.create_view(base_path="/home/user/work/a", sandbox_id="sbx-1")

    assert isinstance(view, E2BSandbox)
    assert view.sandbox_id == "sbx-1"
    assert view.base_path == "/home/user/work/a"
    assert view.connection is sb.connection
    assert view._sandbox is None  # lazy: not connected yet


def test_create_view_defaults_sandbox_id_to_current():
    sb = _sandbox(sandbox_id="sbx-2", base_path="/home/user")
    view = sb.create_view(base_path="/home/user/work/b")
    assert view.sandbox_id == "sbx-2"


def test_ensure_started_materializes_and_returns_id():
    sb = _sandbox()
    fake = MagicMock()
    fake.sandbox_id = "sbx-created"
    with patch.object(sb, "_create_with_retry", return_value=fake):
        sid = sb.ensure_started()
    assert sid == "sbx-created"
    assert sb.current_sandbox_id == "sbx-created"


def test_e2b_supports_views():
    assert _sandbox(sandbox_id="s").supports_views is True


def test_run_command_shell_runs_in_base_path_cwd():
    sb = _sandbox(base_path="/home/user/work/x")
    fake = MagicMock()
    result = MagicMock()
    result.stdout, result.stderr, result.exit_code = "ok", "", 0
    fake.commands.run.return_value = result
    sb._sandbox = fake  # bypass creation

    out = sb.run_command_shell("ls")

    assert out.stdout == "ok"
    assert fake.commands.run.call_args.kwargs.get("cwd") == "/home/user/work/x"
