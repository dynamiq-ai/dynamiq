"""
Todo Management Tools for Agents

Simple todo list management that stores results in the agent's FileStore.
Provides read and write tools for managing a list of tasks.

Usage:
    from dynamiq.nodes.tools.todo_tools import TodoReadTool, TodoWriteTool

    # Todo tools use the agent's file_store
    agent = Agent(
        llm=llm,
        tools=[...],
        todo_config=TodoConfig(enabled=True),
    )
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

    todos: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of todo items to save. Each item should have: "
            "'id' (string), 'content' (string), 'status' ('pending'|'in_progress'|'completed'). "
            "This replaces the entire todo list."
        ),
    )
    merge: bool = Field(
        default=True,
        description="If true, merge with existing todos (update by id). If false, replace all.",
    )


class TodoReadTool(Node):
    """
    Read the current todo list from storage.

    Returns the list of todos with their status and statistics.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "todo-read"
    description: str = """Read the current todo list.

Returns all todos with their id, content, and status (pending/in_progress/completed).
Includes statistics: total count and counts per status.

Usage:
- Get all todos: {}
- Filter by status: {"filter_status": "pending"}
"""

    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=30))
    file_store: FileStore = Field(..., description="File storage for todos")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[TodoReadInputSchema]] = TodoReadInputSchema

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
            logger.warning(f"TodoReadTool: Failed to load todos: {e}")
        return []

    def execute(
        self, input_data: TodoReadInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        todos = self._load_todos()

        # Filter if requested
        if input_data.filter_status:
            todos = [t for t in todos if t.get("status") == input_data.filter_status]

        # Calculate stats
        all_todos = self._load_todos()
        stats = {
            "total": len(all_todos),
            "pending": sum(1 for t in all_todos if t.get("status") == "pending"),
            "in_progress": sum(1 for t in all_todos if t.get("status") == "in_progress"),
            "completed": sum(1 for t in all_todos if t.get("status") == "completed"),
        }

        # Format for agent
        lines = ["📋 Todo List:"]
        if todos:
            for t in todos:
                icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(t.get("status", ""), "❓")
                lines.append(f"  {icon} [{t.get('id')}] {t.get('content')} ({t.get('status')})")
        else:
            lines.append("  (No todos)")

        lines.append("")
        lines.append(
            f"📊 Stats: {stats['total']} total | ⏳ {stats['pending']} pending |\
                  🔄 {stats['in_progress']} in progress | ✅ {stats['completed']} completed"
        )

        return {"content": "\n".join(lines), "todos": all_todos}


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

        new_todos = input_data.todos

        if input_data.merge:
            # Merge with existing todos
            existing = self._load_todos()
            existing_by_id = {t.get("id"): t for t in existing}

            for todo in new_todos:
                todo_id = todo.get("id")
                if todo_id:
                    existing_by_id[todo_id] = todo

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
