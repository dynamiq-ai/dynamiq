import json

from pydantic import ConfigDict, PrivateAttr

from dynamiq.nodes.tools.todo_tools import TODOS_FILE_PATH, TodoWriteTool
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore


class FlakyFileStore(InMemoryFileStore):
    """An InMemoryFileStore that can simulate the two ways a network-backed sandbox lies:
    retrieve() raising (a transient RPC error) and exists() returning False for a file that is
    actually still there (the "fail open" behavior of E2B/Daytona/BedrockAgentCore exists())."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _fail_retrieve: bool = PrivateAttr(default=False)
    _force_missing: bool = PrivateAttr(default=False)

    def retrieve(self, file_path):
        if self._fail_retrieve:
            raise RuntimeError("simulated transient read failure")
        return super().retrieve(file_path)

    def exists(self, file_path):
        if self._force_missing:
            return False
        return super().exists(file_path)


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


def test_merge_on_unreadable_store_fails_without_overwriting():
    """retrieve() raising (e.g. a transient sandbox RPC error) must not be treated as an empty
    store — that would make final_todos just the placeholder items a status update sends
    ({"id": "3", "content": "ignored", ...}) and overwrite the real plan with them."""
    file_store = FlakyFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending", "pending", "pending"), "merge": False})

    file_store._fail_retrieve = True
    result = tool.run({"todos": [{"id": "3", "content": "ignored", "status": "completed"}], "merge": True})

    assert result.status == RunnableStatus.FAILURE
    file_store._fail_retrieve = False
    assert _stored_todos(file_store) == _plan("in_progress", "pending", "pending", "pending", "pending")


def test_merge_on_corrupt_store_fails_without_overwriting():
    """A non-list `todos` key (truncated/corrupt JSON payload) must fail the call instead of
    silently being treated as an empty store."""
    file_store = FlakyFileStore()
    file_store.store(
        file_path=TODOS_FILE_PATH,
        content=json.dumps({"todos": "not-a-list"}),
        content_type="application/json",
        overwrite=True,
    )
    tool = TodoWriteTool(file_store=file_store)

    result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})

    assert result.status == RunnableStatus.FAILURE
    assert _stored_todos(file_store) == "not-a-list"


def test_merge_after_transient_missing_probe_does_not_recreate_over_existing_run_state():
    """exists() on E2B/Daytona/BedrockAgentCore sandboxes fails open (catches every exception
    and returns False) on a transient RPC error. If this tool already created a list earlier in
    the same run, a False from exists() on a later call is not proof the store is actually
    empty — so it must not be treated as "create a new list", which would silently replace the
    real plan with just the placeholder items of a routine status update."""
    file_store = FlakyFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": False})

    file_store._force_missing = True
    result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})
    file_store._force_missing = False

    assert result.status == RunnableStatus.FAILURE
    assert _stored_todos(file_store) == _plan("in_progress", "pending", "pending")


def test_empty_replace_save_clears_the_latch_so_the_next_merge_call_creates():
    """An explicit merge=false save of an empty list (e.g. the agent clearing its own plan) must
    not latch _list_created_this_run — otherwise every later merge=true call in the run would be
    rejected by the "looks empty" guard forever, since the store really is empty."""
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending"), "merge": False})

    empty_result = tool.run({"todos": [], "merge": False})
    assert empty_result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == []

    result = tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": True})

    assert result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == _plan("in_progress", "pending", "pending")


def test_empty_store_guard_error_names_merge_false_as_the_escape_hatch():
    """The recoverable error must not just say "retry" — a model that only ever retries with
    merge=true against a genuinely empty store loops until it hits its iteration limit. It must
    be told the concrete way out."""
    file_store = FlakyFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": False})

    file_store._force_missing = True
    result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})
    file_store._force_missing = False

    assert result.status == RunnableStatus.FAILURE
    assert "merge=false" in result.error.message


def test_after_guard_fires_merge_false_recreates_the_list_and_later_merges_work():
    file_store = FlakyFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": False})

    file_store._force_missing = True
    guard_result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})
    file_store._force_missing = False
    assert guard_result.status == RunnableStatus.FAILURE

    recreate_result = tool.run({"todos": _plan("in_progress", "pending"), "merge": False})
    assert recreate_result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == _plan("in_progress", "pending")

    merge_result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})
    assert merge_result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == [
        {"id": "1", "content": "step 1", "status": "completed"},
        {"id": "2", "content": "step 2", "status": "pending"},
    ]


def test_store_cleared_externally_mid_run_recovers_via_merge_false_without_looping():
    """Simulates a sub-agent sharing this sandbox clearing the shared todos file (base.py's
    _clear_todos_file, called from a sub-agent's own execute() finally) out from under a parent
    agent's TodoWriteTool instance, whose _list_created_this_run stays latched because the two
    are separate tool instances. The parent must be able to recover with merge=false rather than
    being stuck retrying merge=true forever."""
    file_store = InMemoryFileStore()
    tool = TodoWriteTool(file_store=file_store)
    tool.run({"todos": _plan("in_progress", "pending", "pending"), "merge": False})

    # A sub-agent's own TodoWriteTool instance (over the same shared store) clears the file;
    # this tool's own _list_created_this_run is untouched by that.
    file_store.delete(TODOS_FILE_PATH)
    assert tool._list_created_this_run is True

    guard_result = tool.run({"todos": [{"id": "1", "content": "ignored", "status": "completed"}], "merge": True})
    assert guard_result.status == RunnableStatus.FAILURE
    assert "merge=false" in guard_result.error.message

    recreate_result = tool.run({"todos": _plan("in_progress", "pending"), "merge": False})

    assert recreate_result.status == RunnableStatus.SUCCESS
    assert _stored_todos(file_store) == _plan("in_progress", "pending")


def test_listing_shows_status_values_not_enum_reprs():
    tool = TodoWriteTool(file_store=InMemoryFileStore())

    result = tool.run({"todos": _plan("in_progress", "pending"), "merge": False})

    content = result.output["content"]
    assert "(in_progress)" in content
    assert "(pending)" in content
    assert "TodoStatus." not in content
