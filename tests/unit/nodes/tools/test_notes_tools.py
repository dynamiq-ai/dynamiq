import importlib

import pytest

from dynamiq.memory.notes import NotesStorageError, NoteValidationError
from dynamiq.memory.notes.backends import InMemoryNotesBackend
from dynamiq.nodes.tools.notes import DeleteNoteTool, ReadNoteTool, WriteNoteTool, build_notes_tools
from dynamiq.runnables import RunnableStatus

USER_ID = "user-test-123"
OTHER_USER_ID = "user-other-456"


@pytest.fixture
def backend() -> InMemoryNotesBackend:
    return InMemoryNotesBackend()


@pytest.fixture
def tools(backend) -> dict:
    return {tool.name: tool for tool in build_notes_tools(backend=backend, user_id=USER_ID)}


@pytest.fixture
def seeded_tools(tools) -> dict:
    write = tools["write_note"]
    write.execute(write.input_schema(title="Deploy runbook", description="ship steps", content="1. merge"))
    write.execute(write.input_schema(title="API conventions", description="error envelope", content="envelope"))
    return tools


# --- schema contract -------------------------------------------------------------------


@pytest.mark.parametrize("tool_cls", [WriteNoteTool, ReadNoteTool, DeleteNoteTool])
def test_user_id_is_never_exposed_to_the_llm(tool_cls):
    assert "user_id" not in tool_cls.input_schema.model_fields


def test_input_schemas_expose_the_expected_fields():
    assert {"title", "description", "content"} == set(WriteNoteTool.input_schema.model_fields)
    assert {"titles"} == set(ReadNoteTool.input_schema.model_fields)
    assert {"title"} == set(DeleteNoteTool.input_schema.model_fields)


@pytest.mark.parametrize("titles", [[], ["  "], ["ok", ""]])
def test_read_schema_rejects_unusable_titles(titles):
    with pytest.raises(ValueError):
        ReadNoteTool.input_schema(titles=titles)


def test_read_schema_deduplicates_titles():
    schema = ReadNoteTool.input_schema(titles=["a", " a ", "b"])

    assert schema.titles == ["a", "b"]


@pytest.mark.parametrize("field", ["title", "description"])
def test_write_schema_rejects_multiline_single_line_fields(field):
    kwargs = {"title": "t", "description": "d", "content": "c"} | {field: "two\nlines"}

    with pytest.raises(ValueError):
        WriteNoteTool.input_schema(**kwargs)


def test_delete_schema_rejects_a_blank_title():
    with pytest.raises(ValueError):
        DeleteNoteTool.input_schema(title="   ")


# --- write -----------------------------------------------------------------------------


def test_write_persists_under_the_bound_user_id(tools, backend):
    write = tools["write_note"]

    write.execute(write.input_schema(title="Deploy runbook", description="ship steps", content="1. merge"))

    found, missing = backend.read(user_id=USER_ID, titles=["Deploy runbook"])
    assert found[0].content == "1. merge"
    assert missing == []
    assert backend.list_all(user_id=OTHER_USER_ID) == []


def test_write_echoes_the_refreshed_index(seeded_tools):
    write = seeded_tools["write_note"]

    content = write.execute(write.input_schema(title="Third note", description="a third", content="body"))["content"]

    assert isinstance(content, str)
    assert content.startswith("Created note 'Third note'.")
    for title in ("API conventions", "Deploy runbook", "Third note"):
        assert title in content


def test_rewriting_a_title_overwrites_in_place(seeded_tools, backend):
    write = seeded_tools["write_note"]

    content = write.execute(
        write.input_schema(title="Deploy runbook", description="ship steps v2", content="2. deploy")
    )["content"]

    assert content.startswith("Overwrote note 'Deploy runbook'.")
    assert len(backend.list_all(user_id=USER_ID)) == 2
    found, _ = backend.read(user_id=USER_ID, titles=["Deploy runbook"])
    assert found[0].content == "2. deploy"


# --- read ------------------------------------------------------------------------------


def test_read_returns_bodies_under_title_headings_in_request_order(seeded_tools):
    read = seeded_tools["read_note"]

    content = read.execute(read.input_schema(titles=["API conventions", "Deploy runbook"]))["content"]

    assert content == "## API conventions\n\nenvelope\n\n---\n\n## Deploy runbook\n\n1. merge"


def test_read_does_not_echo_the_index_on_a_clean_hit(seeded_tools):
    read = seeded_tools["read_note"]

    content = read.execute(read.input_schema(titles=["Deploy runbook"]))["content"]

    assert "Your notes:" not in content


def test_read_miss_does_not_raise_and_suggests_a_near_match(seeded_tools):
    read = seeded_tools["read_note"]

    content = read.execute(read.input_schema(titles=["deploy runbok"]))["content"]

    assert "Not found: 'deploy runbok'" in content
    assert "Did you mean 'Deploy runbook'?" in content
    assert "Your notes:" in content


def test_read_partial_miss_returns_the_hits_too(seeded_tools):
    read = seeded_tools["read_note"]

    content = read.execute(read.input_schema(titles=["Deploy runbook", "ghost"]))["content"]

    assert "## Deploy runbook" in content
    assert "Not found: 'ghost'." in content


def test_read_on_an_empty_store(tools):
    read = tools["read_note"]

    content = read.execute(read.input_schema(titles=["anything"]))["content"]

    assert "You have no notes yet." in content


def test_read_cannot_reach_another_users_note(backend):
    backend.write(user_id=OTHER_USER_ID, title="Theirs", description="d", content="secret")
    read = {t.name: t for t in build_notes_tools(backend=backend, user_id=USER_ID)}["read_note"]

    content = read.execute(read.input_schema(titles=["Theirs"]))["content"]

    assert "secret" not in content
    assert "Not found: 'Theirs'" in content


# --- delete ----------------------------------------------------------------------------


def test_delete_removes_the_note_and_echoes_the_index(seeded_tools, backend):
    delete = seeded_tools["delete_note"]

    content = delete.execute(delete.input_schema(title="API conventions"))["content"]

    assert content.startswith("Deleted note 'API conventions'.")
    assert "Deploy runbook" in content
    assert "API conventions" not in content.split("Your notes:")[1]
    assert len(backend.list_all(user_id=USER_ID)) == 1


def test_delete_of_a_missing_title_is_soft(seeded_tools, backend):
    delete = seeded_tools["delete_note"]

    content = delete.execute(delete.input_schema(title="ghost"))["content"]

    assert "nothing was deleted" in content
    assert len(backend.list_all(user_id=USER_ID)) == 2


def test_delete_cannot_reach_another_users_note(backend):
    backend.write(user_id=OTHER_USER_ID, title="Theirs", description="d", content="secret")
    delete = {t.name: t for t in build_notes_tools(backend=backend, user_id=USER_ID)}["delete_note"]

    content = delete.execute(delete.input_schema(title="Theirs"))["content"]

    assert "nothing was deleted" in content
    assert len(backend.list_all(user_id=OTHER_USER_ID)) == 1


# --- failure handling ------------------------------------------------------------------


def test_storage_failure_is_not_retryable(tools, mocker):
    """An unreachable database is not something the model can fix by rewording its call —
    marking it recoverable makes the agent retry until it exhausts its loop limit."""
    read = tools["read_note"]
    mocker.patch.object(InMemoryNotesBackend, "read", side_effect=NotesStorageError("connection refused"))

    result = read.run(input_data={"titles": ["anything"]})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is False
    assert "connection refused" in str(result.error)


def test_validation_failure_from_the_backend_is_retryable(tools, mocker):
    read = tools["read_note"]
    mocker.patch.object(InMemoryNotesBackend, "read", side_effect=NoteValidationError("bad title"))

    result = read.run(input_data={"titles": ["anything"]})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable is True


# --- plumbing --------------------------------------------------------------------------


def test_every_tool_returns_plain_text(seeded_tools):
    write, read, delete = (seeded_tools[name] for name in ("write_note", "read_note", "delete_note"))

    outputs = [
        write.execute(write.input_schema(title="t", description="d", content="c"))["content"],
        read.execute(read.input_schema(titles=["t"]))["content"],
        delete.execute(delete.input_schema(title="t"))["content"],
    ]

    assert all(isinstance(output, str) for output in outputs)


def test_run_path_succeeds_and_returns_content(seeded_tools):
    result = seeded_tools["read_note"].run(input_data={"titles": ["Deploy runbook"]})

    assert result.status == RunnableStatus.SUCCESS
    assert "## Deploy runbook" in result.output["content"]


@pytest.mark.parametrize("tool_name", ["write_note", "read_note", "delete_note"])
def test_to_dict_serializes_the_backend(tools, tool_name):
    data = tools[tool_name].to_dict()

    assert isinstance(data["backend"], dict)
    assert data["backend"]["type"] == "dynamiq.memory.notes.backends.InMemoryNotesBackend"
    assert data["user_id"] == USER_ID


def test_to_dict_with_secure_params(tools):
    data = tools["write_note"].to_dict(include_secure_params=True)

    assert isinstance(data["backend"], dict)


@pytest.mark.parametrize("class_name", ["WriteNoteTool", "ReadNoteTool", "DeleteNoteTool"])
def test_tools_are_resolvable_from_the_node_registry(class_name):
    """`NodeManager.get_node_by_type` resolves `dynamiq.nodes.tools.<ClassName>`, so the
    tools must be exported from the package __init__."""
    module = importlib.import_module("dynamiq.nodes.tools")

    assert getattr(module, class_name).__name__ == class_name


def test_factory_builds_all_three_tools_with_user_id_baked_in(backend):
    built = build_notes_tools(backend=backend, user_id=USER_ID)

    assert {tool.name for tool in built} == {"write_note", "read_note", "delete_note"}
    assert all(tool.user_id == USER_ID for tool in built)
    assert all(tool.backend is backend for tool in built)
