import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.memory import memory

MEMORY_ID = "11111111-1111-1111-1111-111111111111"


def test_items_sends_user_id(cli_ctx, recording_api):
    result = CliRunner().invoke(memory, ["items", MEMORY_ID, "--user-id", "u-1"], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[0]
    assert (method, path) == ("GET", f"/v1/memories/{MEMORY_ID}/items")
    assert kwargs["params"] == {"user_id": "u-1"}


def test_items_sends_session_id_when_given(cli_ctx, recording_api):
    result = CliRunner().invoke(memory, ["items", MEMORY_ID, "--user-id", "u-1", "--session-id", "s-1"], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    assert recording_api.calls[0][2]["params"] == {"user_id": "u-1", "session_id": "s-1"}


def test_items_requires_user_id(cli_ctx, recording_api):
    result = CliRunner().invoke(memory, ["items", MEMORY_ID], obj=cli_ctx)

    assert result.exit_code != 0
    assert "--user-id" in result.output
    assert recording_api.calls == []


def test_clear_sends_user_and_session_id(cli_ctx, recording_api):
    result = CliRunner().invoke(
        memory, ["clear", MEMORY_ID, "--user-id", "u-1", "--session-id", "s-1", "--yes"], obj=cli_ctx
    )

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[0]
    assert (method, path) == ("DELETE", f"/v1/memories/{MEMORY_ID}/items")
    assert kwargs["params"] == {"user_id": "u-1", "session_id": "s-1"}


def test_clear_requires_user_id(cli_ctx, recording_api):
    result = CliRunner().invoke(memory, ["clear", MEMORY_ID, "--yes"], obj=cli_ctx)

    assert result.exit_code != 0
    assert recording_api.calls == []


@pytest.mark.parametrize("option", ["--session-id", "--user-id"])
@pytest.mark.parametrize("blank", ["", "   "])
def test_clear_rejects_blank_scope_id(cli_ctx, recording_api, option, blank):
    args = {"--user-id": "u-1", "--session-id": "s-1", option: blank}
    result = CliRunner().invoke(
        memory,
        ["clear", MEMORY_ID, "--user-id", args["--user-id"], "--session-id", args["--session-id"], "--yes"],
        obj=cli_ctx,
    )

    assert result.exit_code != 0
    assert "must not be empty" in result.output
    assert recording_api.calls == []


def test_items_rejects_blank_session_id(cli_ctx, recording_api):
    result = CliRunner().invoke(memory, ["items", MEMORY_ID, "--user-id", "u-1", "--session-id", ""], obj=cli_ctx)

    assert result.exit_code != 0
    assert recording_api.calls == []
