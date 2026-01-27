"""
Todo Management Tools for Agents
"""

import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger

# Path where todos are stored within the file store
TODOS_FILE_PATH = "._agent/todos.json"


class TodoItem(BaseModel):
    """A single todo item."""

    id: str
    content: str
    status: Literal["pending", "in_progress", "completed"] = "pending"

    model_config = ConfigDict(extra="allow")


class TodoReadInputSchema(BaseModel):
    """Input schema for reading todos."""

    filter_status: str | None = Field(
        default=None,
        description="Optional filter: 'pending', 'in_progress', 'completed'. Leave empty for all.",
    )


class TodoWriteInputSchema(BaseModel):
    """Input schema for writing todos."""

    todos: list[TodoItem] = Field(
        ...,
        description=(
            "List of todo items. Each item MUST have: "
            "'id' (required string), 'content' (string), 'status' ('pending'|'in_progress'|'completed')."
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
    file_store: FileStore = Field(..., description="File storage for todos")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[TodoWriteInputSchema]] = TodoWriteInputSchema

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)

    def reset_run_state(self):
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"file_store": True}

    def _load_todos(self) -> list[dict]:
        """Load todos from file store."""
        try:
            if self.file_store.exists(TODOS_FILE_PATH):
                content = self.file_store.retrieve(TODOS_FILE_PATH)
                data = json.loads(content.decode("utf-8"))
                return data.get("todos", [])
        except Exception as e:
            logger.warning(f"TodoWriteTool: Failed to load todos: {e}")
        return []

    def _save_todos(self, todos: list[dict]) -> None:
        """Save todos to file store."""
        content = json.dumps({"todos": todos}, indent=2)
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
            "pending": sum(1 for t in final_todos if t.get("status") == "pending"),
            "in_progress": sum(1 for t in final_todos if t.get("status") == "in_progress"),
            "completed": sum(1 for t in final_todos if t.get("status") == "completed"),
        }

        lines = ["✅ Todos saved successfully!"]
        lines.append("")
        lines.append("📋 Current Todo List:")
        for t in final_todos:
            icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(t.get("status", ""), "❓")
            lines.append(f"  {icon} [{t.get('id')}] {t.get('content')} ({t.get('status')})")

        lines.append("")
        lines.append(
            f"📊 Stats: {stats['total']} total | ⏳ {stats['pending']} pending |\
                  🔄 {stats['in_progress']} in progress | ✅ {stats['completed']} completed"
        )

        return {"content": "\n".join(lines)}
