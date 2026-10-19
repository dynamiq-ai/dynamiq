import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.memory_store import memory_store

STORE_ID = "22222222-2222-2222-2222-222222222222"
STORE_PATH = f"/v1/memory-stores/{STORE_ID}"
FILES_PATH = f"{STORE_PATH}/files"


def invoke(cli_ctx, *args, **kwargs):
    return CliRunner().invoke(memory_store, list(args), obj=cli_ctx, **kwargs)


def test_list_scopes_to_current_project(cli_ctx, recording_api):
    result = invoke(cli_ctx, "list")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [("GET", "/v1/memory-stores", {"params": {"project_id": "proj-1"}})]


def test_list_all_walks_pages(cli_ctx, recording_api):
    recording_api.responses[("GET", "/v1/memory-stores")] = {
        "data": [{"id": "s-1"}],
        "pagination": {"page_count": 1, "total_count": 1},
    }

    result = invoke(cli_ctx, "list", "--all")

    assert result.exit_code == 0, result.output
    assert recording_api.calls[0][2]["params"] == {"project_id": "proj-1", "page": 1, "page_size": 500}


def test_create_fills_in_project_id(cli_ctx, recording_api):
    result = invoke(cli_ctx, "create", '{"name": "notes", "description": "d"}')

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [
        ("POST", "/v1/memory-stores", {"json": {"name": "notes", "description": "d", "project_id": "proj-1"}})
    ]


def test_get(cli_ctx, recording_api):
    result = invoke(cli_ctx, "get", STORE_ID)

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [("GET", STORE_PATH, {})]


def test_update_keeps_fields_not_given(cli_ctx, recording_api):
    recording_api.responses[("GET", STORE_PATH)] = {"data": {"id": STORE_ID, "name": "old", "description": "keep"}}

    result = invoke(cli_ctx, "update", STORE_ID, '{"name": "new"}')

    assert result.exit_code == 0, result.output
    assert recording_api.calls[1] == ("PUT", STORE_PATH, {"json": {"name": "new", "description": "keep"}})


def test_update_rejects_unknown_fields(cli_ctx, recording_api):
    result = invoke(cli_ctx, "update", STORE_ID, '{"project_id": "p"}')

    assert result.exit_code != 0
    assert "cannot update project_id" in result.output
    assert recording_api.calls == []


def test_delete(cli_ctx, recording_api):
    result = invoke(cli_ctx, "delete", STORE_ID, "--yes")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [("DELETE", STORE_PATH, {})]


def test_files_sends_user_id(cli_ctx, recording_api):
    result = invoke(cli_ctx, "files", STORE_ID, "--user-id", "u-1")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [("GET", FILES_PATH, {"params": {"user_id": "u-1"}})]


def test_files_sends_path_prefix(cli_ctx, recording_api):
    result = invoke(cli_ctx, "files", STORE_ID, "--user-id", "u-1", "--path", "notes/")

    assert result.exit_code == 0, result.output
    assert recording_api.calls[0][2]["params"] == {"user_id": "u-1", "path": "notes/"}


def test_read_sends_user_and_path(cli_ctx, recording_api):
    result = invoke(cli_ctx, "read", STORE_ID, "notes/a.md", "--user-id", "u-1")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [
        ("GET", f"{FILES_PATH}/content", {"params": {"user_id": "u-1", "path": "notes/a.md"}})
    ]


def test_read_raw_prints_only_content(cli_ctx, recording_api):
    recording_api.responses[("GET", f"{FILES_PATH}/content")] = {"data": {"path": "a.md", "content": "hello"}}

    result = invoke(cli_ctx, "read", STORE_ID, "a.md", "--user-id", "u-1", "--raw")

    assert result.exit_code == 0, result.output
    assert result.output == "hello"


def test_write_sends_content(cli_ctx, recording_api):
    result = invoke(cli_ctx, "write", STORE_ID, "a.md", "--user-id", "u-1", "--content", "hello")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [
        ("PUT", FILES_PATH, {"json": {"path": "a.md", "content": "hello", "user_id": "u-1"}})
    ]


def test_write_reads_content_from_stdin(cli_ctx, recording_api):
    result = invoke(cli_ctx, "write", STORE_ID, "a.md", "--user-id", "u-1", "--from-file", "-", input="from stdin")

    assert result.exit_code == 0, result.output
    assert recording_api.calls[0][2]["json"]["content"] == "from stdin"


@pytest.mark.parametrize(
    "extra",
    [[], ["--content", "x", "--from-file", "-"], ["--content", ""]],
    ids=["neither", "both", "empty"],
)
def test_write_requires_exactly_one_non_empty_content(cli_ctx, recording_api, extra):
    result = invoke(cli_ctx, "write", STORE_ID, "a.md", "--user-id", "u-1", *extra)

    assert result.exit_code != 0
    assert recording_api.calls == []


def test_delete_file_sends_user_and_path(cli_ctx, recording_api):
    result = invoke(cli_ctx, "delete-file", STORE_ID, "a.md", "--user-id", "u-1", "--yes")

    assert result.exit_code == 0, result.output
    assert recording_api.calls == [("DELETE", FILES_PATH, {"params": {"user_id": "u-1", "path": "a.md"}})]


@pytest.mark.parametrize(
    "args",
    [
        ["files", STORE_ID],
        ["read", STORE_ID, "a.md"],
        ["write", STORE_ID, "a.md", "--content", "x"],
        ["delete-file", STORE_ID, "a.md", "--yes"],
    ],
    ids=["files", "read", "write", "delete-file"],
)
def test_file_commands_require_user_id(cli_ctx, recording_api, args):
    result = invoke(cli_ctx, *args)

    assert result.exit_code != 0
    assert "--user-id" in result.output
    assert recording_api.calls == []


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize(
    "args",
    [
        ["files", STORE_ID, "--user-id", "{blank}"],
        ["files", STORE_ID, "--user-id", "u-1", "--path", "{blank}"],
        ["read", STORE_ID, "{blank}", "--user-id", "u-1"],
        ["write", STORE_ID, "{blank}", "--user-id", "u-1", "--content", "x"],
        ["delete-file", STORE_ID, "a.md", "--user-id", "{blank}", "--yes"],
    ],
    ids=["files-user", "files-prefix", "read-path", "write-path", "delete-user"],
)
def test_rejects_blank_user_id_and_path(cli_ctx, recording_api, args, blank):
    result = invoke(cli_ctx, *[blank if a == "{blank}" else a for a in args])

    assert result.exit_code != 0
    assert "must not be empty" in result.output
    assert recording_api.calls == []
