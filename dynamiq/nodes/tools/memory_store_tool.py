from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.memory.base import MemoryNotFoundError, MemoryStore, MemoryStoreError, render_namespaces
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION = """Your memory: notes you keep across conversations. Everything else — this chat, \
your working files, your answer — is discarded when the conversation ends.

Actions:
- list: what you already know. Optional 'path' narrows to a prefix.
- read: the content of one memory. Requires 'path'.
- write: create or replace a memory. Requires 'path' and 'content'.
- edit: change part of a memory in place. Requires 'path', 'find' and 'replace'; 'find' must match \
exactly one place unless 'replace_all' is true.
- delete: remove a memory that has become wrong. Requires 'path'.

Record facts that will still matter next time — preferences, standing rules, corrections, anything \
lasting about the user, their team or their setup. Never the deliverable, working state, or secrets.
One topic per memory, descriptive path; edit rather than duplicate.

Usage examples:
- {"action": "list"}
- {"action": "read", "path": "preferences.md"}
- {"action": "write", "path": "preferences.md", "content": "Prefers British English."}
- {"action": "edit", "path": "preferences.md", "find": "British", "replace": "Australian"}"""

READ_ONLY_NOTE = "\n\nThis memory is READ-ONLY: 'write', 'edit' and 'delete' are unavailable."


class MemoryStoreAction(str, Enum):
    """Action for the memory store tool."""

    LIST = "list"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"


MUTATING_ACTIONS = {MemoryStoreAction.WRITE, MemoryStoreAction.EDIT, MemoryStoreAction.DELETE}


class MemoryStoreToolInputSchema(BaseModel):
    """Input schema for the memory store tool."""

    action: MemoryStoreAction = Field(..., description="What to do: list, read, write, edit or delete.")
    path: str | None = Field(
        default=None,
        description="Memory path, e.g. 'preferences.md'. Required for every action except list, "
        "where it optionally narrows to a prefix.",
    )
    content: str | None = Field(default=None, description="The memory's full text. Required for write.")
    find: str | None = Field(default=None, description="Exact text to replace. Required for edit.")
    replace: str | None = Field(default=None, description="Text to put in its place. Required for edit.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence of 'find' instead of requiring it to be unique.",
    )
    brief: str = Field(default="Using memory", description="Short description of what you are doing.")

    @model_validator(mode="after")
    def validate_action_fields(self):
        """Fail fast on the arguments each action needs, rather than part-way through execution."""
        if self.action != MemoryStoreAction.LIST and not self.path:
            raise ValueError(f"'path' is required for action '{self.action.value}'")
        if self.action == MemoryStoreAction.WRITE and self.content is None:
            raise ValueError("'content' is required for action 'write'")
        if self.action == MemoryStoreAction.EDIT and (self.find is None or self.replace is None):
            raise ValueError("'find' and 'replace' are required for action 'edit'")
        return self


class MemoryStoreTool(Node):
    """The agent's memory: list, read, write, edit and delete notes that outlive the conversation.

    One tool with actions rather than several tools, following ``SkillsTool``. Its backend is a
    ``MemoryStore`` — deliberately not a ``FileStore``, so memory never shares a path namespace or a
    tool with the agent's working files.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.FILE_OPERATION
    # Agent machinery, not an outside-world integration: never swept in by MockPolicy.ALL.
    is_mockable: ClassVar[bool] = False
    name: str = "memory-store"
    description: str = DESCRIPTION
    backend: MemoryStore = Field(..., description="Store holding the agent's memories.")
    write_enabled: bool = Field(default=True, description="Whether the agent may change memories.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[MemoryStoreToolInputSchema]] = MemoryStoreToolInputSchema

    @model_validator(mode="after")
    def describe_memories(self):
        """Tell the model what each memory holds, straight from the stores themselves.

        The backend already knows — a composite composes its routes' descriptions — so there is no
        parallel structure to keep in sync and no need to know whether the backend is composite.
        """
        listing = render_namespaces(self.backend.describe_namespaces())
        if listing and "Your memories:" not in self.description:
            self.description = f"{self.description}\n\nYour memories:\n{listing}"
        if not self.write_enabled and READ_ONLY_NOTE not in self.description:
            self.description += READ_ONLY_NOTE
        return self

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"backend": True}

    def execute(
        self, input_data: MemoryStoreToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        action = input_data.action
        if action in MUTATING_ACTIONS and not self.write_enabled:
            raise ToolExecutionException(
                f"This memory is read-only; '{action.value}' is not available. You can list and read it.",
                recoverable=True,
            )

        try:
            if action == MemoryStoreAction.LIST:
                return self._list(input_data.path or "")
            if action == MemoryStoreAction.READ:
                return self._read(input_data.path)
            if action == MemoryStoreAction.WRITE:
                return self._write(input_data.path, input_data.content)
            if action == MemoryStoreAction.EDIT:
                return self._edit(input_data)
            return self._delete(input_data.path)
        except ToolExecutionException:
            raise
        except MemoryStoreError as e:
            # Includes "not in any memory", which names the valid prefixes: worth relaying verbatim.
            raise ToolExecutionException(str(e), recoverable=True) from e
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: {action.value} failed. Error: {e}")
            raise ToolExecutionException(
                f"Memory {action.value} failed: {e}. Please analyze the error and take appropriate action.",
                recoverable=True,
            ) from e

    def _list(self, prefix: str) -> dict[str, Any]:
        """List memories, grouped under what each one holds.

        The grouping matters: a bare path like ``team/naming.md`` gives the model nothing to judge
        relevance by, so it lists and then reads nothing. Naming the memory the path belongs to puts
        that judgement where the decision is made, the same way the tool description does.
        """
        entries = self.backend.list(prefix)
        if not entries:
            where = f" under '{prefix}'" if prefix else ""
            return {"content": f"No memories{where} yet."}

        namespaces = self.backend.describe_namespaces()
        grouped: dict[str, list[str]] = {}
        for entry in entries:
            owner = max(
                (ns for ns in namespaces if ns and entry.path.startswith(ns)),
                key=len,
                default="",
            )
            grouped.setdefault(owner, []).append(f"  - {entry.path} ({entry.size} chars)")

        sections = []
        for owner, lines in grouped.items():
            heading = f"{owner} — {namespaces[owner]}" if owner and namespaces.get(owner) else (owner or "Memories")
            sections.append(f"{heading}\n" + "\n".join(lines))
        return {"content": "\n".join(sections), "paths": [entry.path for entry in entries]}

    def _read(self, path: str) -> dict[str, Any]:
        try:
            return {"content": self.backend.read(path)}
        except MemoryNotFoundError:
            raise ToolExecutionException(
                f"No memory at '{path}'. Use action 'list' to see what exists.", recoverable=True
            ) from None

    def _write(self, path: str, content: str) -> dict[str, Any]:
        entry = self.backend.write(path, content)
        return {"content": f"Remembered in '{entry.path}'."}

    def _edit(self, input_data: MemoryStoreToolInputSchema) -> dict[str, Any]:
        path, find, replace = input_data.path, input_data.find, input_data.replace
        try:
            current = self.backend.read(path)
        except MemoryNotFoundError:
            raise ToolExecutionException(
                f"No memory at '{path}' to edit. Use action 'write' to create it.", recoverable=True
            ) from None

        occurrences = current.count(find)
        if occurrences == 0:
            raise ToolExecutionException(
                f"'{find}' does not appear in '{path}', so nothing was changed.", recoverable=True
            )
        if occurrences > 1 and not input_data.replace_all:
            raise ToolExecutionException(
                f"'{find}' appears {occurrences} times in '{path}'. Include more surrounding text to "
                f"make it unique, or set 'replace_all' to change every occurrence.",
                recoverable=True,
            )

        updated = current.replace(find, replace) if input_data.replace_all else current.replace(find, replace, 1)
        self.backend.write(path, updated)
        changed = occurrences if input_data.replace_all else 1
        return {"content": f"Updated '{path}' ({changed} replacement{'s' if changed > 1 else ''})."}

    def _delete(self, path: str) -> dict[str, Any]:
        deleted = self.backend.delete(path)
        return {"content": f"Deleted '{path}'." if deleted else f"There was no memory at '{path}'."}
