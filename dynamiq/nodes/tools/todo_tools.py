"""
Todo Management Tools for Agents
"""

import json
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


# Path where todos are stored within the file store
TODOS_FILE_PATH = "._agent/todos.json"


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
            "List of todo items. Each item MUST have: "
            "'id' (required string), 'content' (string), 'status' (TodoStatus enum: pending/in_progress/completed)."
        ),
    )
    merge: bool = Field(
        default=True,
        description="If true, merge with existing todos (update by id). If false, replace all.",
    )


class TodoWriteTool(Node):
    """
    Write/update the todo list in storage.

    Saves the provided list of todos, either merging with existing or replacing all.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "todo-write"
    description: str = """Save or update the todo list.

Each todo item needs: 'id', 'content', 'status' (pending/in_progress/completed).

RULES:
- When creating initial list: first task "in_progress", rest "pending"
- After initial creation, ONLY update status via merge=true - do not restructure the plan
- Use merge=true to update existing todos by id
- Only use merge=false for initial list creation
"""

    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=30))
    file_store: FileStore | Sandbox = Field(..., description="File storage for todos")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[TodoWriteInputSchema]] = TodoWriteInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

    def reset_run_state(self):
        self._run_depends = []

    def _load_todos(self) -> list[dict]:
        """Load todos from file store."""
        try:
            if self.file_store.exists(TODOS_FILE_PATH):
                content = self.file_store.retrieve(TODOS_FILE_PATH)
                data = json.loads(content.decode("utf-8"))
                todos = data.get("todos")
                if not isinstance(todos, list):
                    logger.warning(f"TodoWriteTool: Invalid todos format (expected list, got {type(todos).__name__})")
                    return []
                # Validate each item is a valid TodoItem
                validated = []
                for t in todos:
                    try:
                        validated.append(TodoItem.model_validate(t).model_dump())
                    except Exception as e:
                        logger.warning(f"TodoWriteTool: Skipping invalid todo item: {e}")
                return validated
        except Exception as e:
            logger.warning(f"TodoWriteTool: Failed to load todos: {e}")
        return []

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

    def execute(
        self, input_data: TodoWriteInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # Convert TodoItem objects to dicts for storage
        new_todos = [todo.model_dump() for todo in input_data.todos]

        if input_data.merge:
            # Merge with existing todos
            existing = self._load_todos()
            existing_by_id = {t.get("id"): t for t in existing if t.get("id")}

            # id is guaranteed by TodoItem validation
            for todo in new_todos:
                existing_by_id[todo["id"]] = todo

            final_todos = list(existing_by_id.values())
        else:
            # Replace all
            final_todos = new_todos

        self._save_todos(final_todos)

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
