"""
Todo Management Tools for Agents
"""

import json
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.file_tools import RESERVED_AGENT_PATH_PREFIX
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox
from dynamiq.storages.file.base import FileStore
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


TODOS_FILE_PATH = f"{RESERVED_AGENT_PATH_PREFIX}/todos.json"


class TodoItem(BaseModel):
    """A single todo item."""

    id: str
    content: str
    status: TodoStatus = TodoStatus.PENDING

    model_config = ConfigDict(extra="allow")

    def to_display_string(self) -> str:
        """Format todo item for display with status icon."""
        icon = {
            TodoStatus.PENDING: "[ ]",
            TodoStatus.IN_PROGRESS: "[~]",
            TodoStatus.COMPLETED: "[+]",
        }.get(self.status, "[ ]")
        return f"{icon} {self.id}: {self.content}"


class TodoWriteInputSchema(BaseModel):
    """Input schema for writing todos."""

    todos: list[TodoItem] = Field(
        ...,
        description=(
            "List of todo items. Each item MUST have 'id', 'content', and 'status'. "
            "With merge=true on an empty store, the items are stored as the new list. "
            "With merge=true on a non-empty store, every id must already exist — only status is updated "
            "(the original content is preserved); an unknown id fails the whole call."
        ),
    )
    merge: bool = Field(
        default=True,
        description="If true and no todo list exists yet, store the provided items as the new list. "
        "If true and a list already exists, update the status of existing todos by id (content you send is "
        "ignored, original is preserved) — every id must already exist, or the call fails. "
        "If false, replace all todos with the provided list.",
    )


class TodoWriteTool(Node):
    """
    Write/update the todo list in storage.

    Saves the provided list of todos, either merging with existing or replacing all.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    # Agent machinery, not an outside-world integration: never swept in by MockPolicy.ALL.
    is_mockable: ClassVar[bool] = False
    name: str = "todo-write"
    description: str = """Save or update the todo list. Every item requires 'id', 'content', and 'status'.

Two modes:

CREATE (merge=false, or merge=true when no list exists yet): Build the full todo list.
  {"todos": [{"id": "1", "content": "Implement auth", "status": "in_progress"},
  {"id": "2", "content": "Add tests", "status": "pending"}], "merge": false}

UPDATE (merge=true on an existing list, default): Change status of existing items. Content is required but
ignored for them — the original content is preserved. Every id you send must already exist; an unknown id
fails the whole call — it does NOT create a new item.
  {"todos": [{"id": "1", "content": "ignored", "status": "completed"},
  {"id": "2", "content": "ignored", "status": "in_progress"}], "merge": true}

RULES:
- Use merge=false (or merge=true on the first call) for initial list creation. First task should be
  "in_progress", rest "pending".
- Use merge=true for ALL subsequent updates — only status is applied to existing ids, content stays unchanged.
- Do NOT restructure, reword, or reorder todos when updating status.
- Do NOT invent ids for the update call — send back the exact ids from the last todo-write result.
"""

    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=30))
    file_store: FileStore | Sandbox = Field(..., description="File storage for todos")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[TodoWriteInputSchema]] = TodoWriteInputSchema

    # Set once this tool has written a todos file during the current agent run (reset by
    # clear(), which runs at the end of every run). Used to tell a genuinely empty store apart
    # from a store that merely *looks* empty because exists()/retrieve() failed transiently —
    # see the comment in execute() for why exists() alone can't be trusted for that.
    _list_created_this_run: bool = PrivateAttr(default=False)

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

    def reset_run_state(self):
        self._run_depends = []

    def _load_todos(self) -> list[dict] | None:
        """Load todos from file store.

        Returns:
            The stored todo list (possibly empty) when the store is confirmed to not exist yet,
            or was read and parsed successfully. Returns None when the store's state could not
            be established — the existence check or the read raised, or the content is
            truncated/corrupt/not the expected shape. Callers must not treat None as "empty": on
            network-backed sandboxes a transient RPC error surfaces the same way an absent file
            does, so collapsing that into [] would let a read failure look like a fresh store.
        """
        try:
            exists = self.file_store.exists(TODOS_FILE_PATH)
        except Exception as e:
            logger.warning(f"TodoWriteTool: Failed to check todo store existence: {e}")
            return None

        if not exists:
            return []

        try:
            content = self.file_store.retrieve(TODOS_FILE_PATH)
            data = json.loads(content.decode("utf-8"))
            todos = data.get("todos")
        except Exception as e:
            logger.warning(f"TodoWriteTool: Failed to load todos: {e}")
            return None

        if not isinstance(todos, list):
            logger.warning(f"TodoWriteTool: Invalid todos format (expected list, got {type(todos).__name__})")
            return None

        validated = []
        for t in todos:
            try:
                validated.append(TodoItem.model_validate(t).model_dump(mode="json"))
            except Exception as e:
                logger.warning(f"TodoWriteTool: Skipping invalid todo item: {e}")
        return validated

    def _save_todos(self, todos: list[dict]) -> None:
        """Save todos to file store or sandbox."""
        content = json.dumps({"todos": todos}, indent=2)
        if isinstance(self.file_store, Sandbox):
            self.file_store.upload_file(
                TODOS_FILE_PATH,
                content.encode("utf-8"),
            )
        else:
            self.file_store.store(
                file_path=TODOS_FILE_PATH,
                content=content,
                content_type="application/json",
                overwrite=True,
            )

    def clear(self) -> bool:
        """Delete the todos file from storage.

        Errors are logged and swallowed — cleanup must never fail the agent run.
        Returns True if the file was removed (or absent); False on failure.
        """
        try:
            if isinstance(self.file_store, Sandbox):
                return bool(self.file_store.delete_file(TODOS_FILE_PATH))
            return bool(self.file_store.delete(TODOS_FILE_PATH))
        except Exception as e:
            logger.warning(f"TodoWriteTool: failed to clear todos file: {e}")
            return False
        finally:
            # A new run starts with no known list, whether or not the delete above succeeded.
            self._list_created_this_run = False

    def execute(
        self, input_data: TodoWriteInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # mode="json" stores the status as its value; a plain dump keeps the enum, which
        # renders as "TodoStatus.PENDING" in the listing below.
        new_todos = [todo.model_dump(mode="json") for todo in input_data.todos]

        if input_data.merge:
            existing = self._load_todos()

            if existing is None:
                # The store exists but couldn't be read/parsed (retrieve() raised, or the
                # content is truncated/corrupt). Treating this as "empty" would make
                # final_todos just the placeholder items the model sends on a status update
                # ({"id": "3", "content": "ignored", ...}), and _save_todos would overwrite the
                # real plan with those. Fail without writing instead — the caller can retry.
                raise ToolExecutionException(
                    "Could not read the current todo list (read failed or its content is "
                    "corrupt). Not overwriting it — retry the call.",
                    recoverable=True,
                )

            existing_by_id = {t.get("id"): t for t in existing if t.get("id")}

            if existing_by_id:
                # Store already has a plan: merge=true only updates statuses of known ids.
                # An id the store doesn't have is a renumbered plan, a hallucinated id, or a
                # typo — not a new todo — so the model needs a recoverable error naming the
                # valid ids, not a silent insert of a placeholder ("ignored") item.
                unknown_ids = [t["id"] for t in new_todos if t["id"] not in existing_by_id]
                if unknown_ids:
                    raise ToolExecutionException(
                        f"Todo ids not found: {unknown_ids}. Existing ids: {list(existing_by_id.keys())}",
                        recoverable=True,
                    )

                for todo in new_todos:
                    existing_by_id[todo["id"]]["status"] = todo["status"]
            elif self._list_created_this_run:
                # existing_by_id is empty, but this tool already wrote a list earlier in this
                # run. exists()/retrieve() on network-backed sandboxes fail open (they catch
                # every exception and report "missing") on a transient RPC error, so an "empty
                # store" result here is not trustworthy — it's far more likely a flaky probe
                # than the list we just created having vanished. Refuse rather than silently
                # recreating over it with placeholder content.
                raise ToolExecutionException(
                    "The todo store looks empty, but this run already has a todo list — this "
                    "looks like a transient read failure rather than a genuinely empty store. "
                    "Retry the call.",
                    recoverable=True,
                )
            else:
                # merge=true is the default, so models routinely create the first list with
                # it. There is nothing to conflict with a genuinely empty store, so treat this
                # call as creating the list.
                for todo in new_todos:
                    existing_by_id[todo["id"]] = todo

            final_todos = list(existing_by_id.values())
        else:
            # Replace all
            final_todos = new_todos

        self._save_todos(final_todos)
        self._list_created_this_run = True

        # Calculate stats
        stats = {
            "total": len(final_todos),
            TodoStatus.PENDING.value: sum(1 for t in final_todos if t.get("status") == TodoStatus.PENDING.value),
            TodoStatus.IN_PROGRESS.value: sum(
                1 for t in final_todos if t.get("status") == TodoStatus.IN_PROGRESS.value
            ),
            TodoStatus.COMPLETED.value: sum(1 for t in final_todos if t.get("status") == TodoStatus.COMPLETED.value),
        }

        status_icons = {
            TodoStatus.PENDING.value: "⏳",
            TodoStatus.IN_PROGRESS.value: "🔄",
            TodoStatus.COMPLETED.value: "✅",
        }

        lines = ["✅ Todos saved successfully!"]
        lines.append("")
        lines.append("📋 Current Todo List:")
        for t in final_todos:
            icon = status_icons.get(t.get("status", ""), "❓")
            lines.append(f"  {icon} [{t.get('id')}] {t.get('content')} ({t.get('status')})")

        lines.append("")
        pending = TodoStatus.PENDING.value
        in_progress = TodoStatus.IN_PROGRESS.value
        completed = TodoStatus.COMPLETED.value
        lines.append(
            f"📊 Stats: {stats['total']} total | ⏳ {stats[pending]} pending |"
            f" 🔄 {stats[in_progress]} in progress | ✅ {stats[completed]} completed"
        )

        return {"content": "\n".join(lines)}
