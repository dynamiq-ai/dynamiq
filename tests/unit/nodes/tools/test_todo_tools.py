import json

from dynamiq.nodes.tools.todo_tools import TODOS_FILE_PATH, TodoWriteTool
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore


def _stored_todos(file_store: InMemoryFileStore) -> list[dict]:
    return json.loads(file_store.retrieve(TODOS_FILE_PATH).decode("utf-8"))["todos"]


def _plan(*statuses: str) -> list[dict]:
    return [{"id": str(i), "content": f"step {i}", "status": status} for i, status in enumerate(statuses, start=1)]


def test_merge_on_empty_store_creates_the_list():
    """The first call with merge=true (the default) used to fail with "Todo ids not found ... Existing ids: []"."""
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)

    result = tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": True})

    assert result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == _plan("in_progress", "pending", "pending")


def test_merge_omitted_on_empty_store_creates_the_list():
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)

    result = tool.run({"todos": _plan("in_progress", "pending")})

    assert result.status == RunnableStatus.SUCCESS
    assert [t["id"] for t in _stored_todos(file_store)] == ["1", "2"]


def test_merge_updates_status_of_existing_ids_and_keeps_content():
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending"), "merge": False})

    result = tool.run(
        {
            "todos": [
                {"id": "1", "content": "ignored", "status": "completed"},
                {"id": "2", "content": "ignored", "status": "in_progress"},
            ],
            "merge": True,
        }
    )

    assert result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == [
        {"id": "1", "content": "step 1", "status": "completed"},
        {"id": "2", "content": "step 2", "status": "in_progress"},
    ]


def test_merge_on_non_empty_store_with_unknown_id_fails_and_leaves_store_unchanged():
    """A renumbered plan, a hallucinated id, or a typo must not be silently inserted as a
    placeholder ("ignored") todo — the model needs a recoverable error naming the valid ids."""
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress"), "merge": False})

    result = tool.run(
        {
            "todos": [
                {"id": "1", "content": "ignored", "status": "completed"},
                {"id": "2", "content": "ignored", "status": "in_progress"},
            ],
            "merge": True,
        }
    )

    assert result.status == RunnableStatus.FAILURE
    assert "Todo ids not found: ['2']" in result.error.message
    assert "Existing ids: ['1']" in result.error.message
    assert _stored_todos(file_store) == _plan("in_progress")


def test_state_persists_across_calls_on_the_same_store():
    """Separate tool instances over one store (as a re-created tool within a run would be) see the same list."""
    file_store = InMemoryFileStore()
    TodoWriteTool(file_store=file_store).run({"todos": _plan("in_progress", "pending"), "merge": False})

    result = TodoWriteTool(file_store=file_store).run(
        {"todos": [{"id": "2", "content": "ignored", "status": "completed"}], "merge": True}
    )

    assert result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == [
        {"id": "1", "content": "step 1", "status": "in_progress"},
        {"id": "2", "content": "step 2", "status": "completed"},
    ]


def test_replace_mode_overwrites_the_list():
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": False})

    tool.run({"todos": [{"id": "a", "content": "new plan", "status": "pending"}], "merge": False})

    assert _stored_todos(file_store) == [{"id": "a", "content": "new plan", "status": "pending"}]


def test_listing_shows_status_values_not_enum_reprs():
    tool = TodoWriteTool(file_store=InMemoryFileStore())

    result = tool.run({"todos": _plan("in_progress", "pending"), "merge": False})

    content = result.output["content"]
    assert "(in_progress)" in content
    assert "(pending)" in content
    assert "TodoStatus." not in content
