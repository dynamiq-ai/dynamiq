import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.utils import find_positions
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.memory.base import (
    MemoryEntry,
    MemoryNotFoundError,
    MemoryStore,
    MemoryStoreError,
    render_namespaces,
)
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION = """Your memory: notes you keep across conversations. Everything else — this chat, \
your working files, your answer — is discarded when the conversation ends.

Actions:
- list: what you already know, one line per memory saying what it holds. Optional 'path' narrows to a prefix.
- search: memories that mention something, best first, with the passage that matched. Requires 'query': \
the words the answer would use. Optional 'path' narrows to a prefix.
- read: the content of one memory. Requires 'path'.
- write: create or replace a memory. Requires 'path' and 'content'.
- edit: change part of a memory in place. Requires 'path', 'find' and 'replace'; 'find' must match \
exactly one place unless 'replace_all' is true.
- delete: remove a memory that has become wrong. Requires 'path'.

Record facts that will still matter next time — preferences, standing rules, corrections, anything \
lasting about the user, their team or their setup. Never the deliverable, working state, or secrets.
One topic per memory, descriptive path, and a first line saying what it holds: listings show that line.
Edit rather than duplicate: a write names the memories on nearby topics.

Usage examples:
- {"action": "list"}
- {"action": "search", "query": "refund approval limit"}
- {"action": "read", "path": "preferences.md"}
- {"action": "write", "path": "preferences.md", "content": "How the user likes answers written.\\n\\nPrefers \
British English."}
- {"action": "edit", "path": "preferences.md", "find": "British", "replace": "Australian"}"""

READ_ONLY_NOTE = "\n\nThis memory is READ-ONLY: 'write', 'edit' and 'delete' are unavailable."
LIMIT_NOTE_START = "\n\nEach memory holds at most"


class MemoryStoreAction(str, Enum):
    """Action for the memory store tool."""

    LIST = "list"
    SEARCH = "search"
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    DELETE = "delete"


SEARCH_LIMIT = 8
RELATED_LIMIT = 3
# A write names another memory only when it holds at least this share of the new memory's topic words,
# so one common word in both does not pass for a related topic.
RELATED_MIN_SCORE = 0.5
# A listing describes the most recently changed memories until their lines reach this many characters,
# about 4,000 tokens, and lists the rest by path, so its size grows with the store without a jump.
LIST_DESCRIBED_CHARS = 16_000

MUTATING_ACTIONS = {MemoryStoreAction.WRITE, MemoryStoreAction.EDIT, MemoryStoreAction.DELETE}


class MemoryStoreToolInputSchema(BaseModel):
    """Input schema for the memory store tool."""

    action: MemoryStoreAction = Field(..., description="What to do: list, search, read, write, edit or delete.")
    path: str | None = Field(
        default=None,
        description="Memory path, e.g. 'preferences.md'. Required for every action except list and "
        "search, where it optionally narrows to a prefix.",
    )
    query: str | None = Field(default=None, description="What to look for. Required for search.")
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
        if self.action not in (MemoryStoreAction.LIST, MemoryStoreAction.SEARCH) and not self.path:
            raise ValueError(f"'path' is required for action '{self.action.value}'")
        if self.action == MemoryStoreAction.SEARCH and not (self.query or "").strip():
            raise ValueError("'query' is required for action 'search'")
        if self.action == MemoryStoreAction.WRITE and self.content is None:
            raise ValueError("'content' is required for action 'write'")
        # An empty `find` matches between every character; `replace` may be empty.
        if self.action == MemoryStoreAction.EDIT and (not self.find or self.replace is None):
            raise ValueError("'find' (non-empty) and 'replace' are required for action 'edit'")
        return self


class MemoryStoreTool(Node):
    """The agent's memory: notes that outlive the conversation.

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
    user_id: str | None = Field(
        default=None,
        description="End user whose memories these are, bound at construction. An agent rebuilds "
        "the tool per run from the run's user_id, so callers only pass it to `agent.run(...)`.",
    )

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
        limit = self.backend.content_limit()
        if self.write_enabled and limit and LIMIT_NOTE_START not in self.description:
            self.description += (
                f"{LIMIT_NOTE_START} {limit:,} characters: split one that would grow past that by topic."
            )
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
            user_id = self.user_id
            if action == MemoryStoreAction.LIST:
                return self._list(input_data.path or "", user_id)
            if action == MemoryStoreAction.SEARCH:
                return self._search(input_data.query, input_data.path or "", user_id)
            if action == MemoryStoreAction.READ:
                return self._read(input_data.path, user_id)
            if action == MemoryStoreAction.WRITE:
                return self._write(input_data.path, input_data.content, user_id)
            if action == MemoryStoreAction.EDIT:
                return self._edit(input_data, user_id)
            return self._delete(input_data.path, user_id)
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

    def _list(self, prefix: str, user_id: str | None = None) -> dict[str, Any]:
        """List memories, grouped under what each one holds.

        The grouping matters: a bare path like ``team/naming.md`` gives the model nothing to judge
        relevance by, so it lists and then reads nothing. Naming the memory the path belongs to puts
        that judgement where the decision is made, the same way the tool description does.
        """
        entries = self.backend.list(prefix, user_id)
        if not entries:
            where = f" under '{prefix}'" if prefix else ""
            return {"content": f"No memories{where} yet."}

        described = _recently_described(entries, LIST_DESCRIBED_CHARS)

        namespaces = self.backend.describe_namespaces()
        grouped: dict[str, list[str]] = {}
        for entry in entries:
            owner = max(
                (ns for ns in namespaces if ns and entry.path.startswith(ns)),
                key=len,
                default="",
            )
            line = _describe(entry) if entry.path in described else f"{entry.path} ({entry.size} chars)"
            grouped.setdefault(owner, []).append(f"  - {line}")

        sections = []
        for owner, lines in grouped.items():
            heading = f"{owner} — {namespaces[owner]}" if owner and namespaces.get(owner) else (owner or "Memories")
            sections.append(f"{heading}\n" + "\n".join(lines))
        if len(described) < len(entries):
            sections.append(
                f"{len(entries)} memories; the {len(described)} changed most recently are described. 'search' "
                "for a topic, or 'list' a folder to see what each of its memories holds."
            )
        return {"content": "\n".join(sections), "paths": [entry.path for entry in entries]}

    def _read(self, path: str, user_id: str | None = None) -> dict[str, Any]:
        try:
            return {"content": self.backend.read(path, user_id)}
        except MemoryNotFoundError:
            raise ToolExecutionException(
                f"No memory at '{path}'. Use action 'list' to see what exists.", recoverable=True
            ) from None

    def _search(self, query: str, prefix: str, user_id: str | None = None) -> dict[str, Any]:
        """Hits with what each memory holds and the passage that matched: enough to pick what to read."""
        hits = self.backend.search(query, prefix, SEARCH_LIMIT, user_id)
        if not hits:
            where = f" under '{prefix}'" if prefix else ""
            return {"content": f"No memories{where} mention '{query}'. Try other words, or 'list'."}
        lines = []
        for hit in hits:
            lines.append(f"- {_describe(hit)}")
            if hit.snippet:
                lines.append(f"  > {hit.snippet}")
        return {"content": "\n".join(lines), "paths": [hit.path for hit in hits]}

    def _write(self, path: str, content: str, user_id: str | None = None) -> dict[str, Any]:
        """Write, then name memories on the same topic, so the next step merges instead of duplicating.

        Best effort: a failed search must not turn a saved memory into a failed call.
        """
        entry = self.backend.write(path, content, user_id)
        message = f"Remembered in '{entry.path}'."
        # Within the memory that holds it: links resolve inside one memory, and a route's name is not a topic.
        scope = self.backend.scope_of(entry.path)
        try:
            topic = " ".join(filter(None, [entry.description, _path_words(entry.path.removeprefix(scope))]))
            hits = self.backend.search(topic, scope, RELATED_LIMIT + 1, user_id)
        except MemoryStoreError as e:
            logger.warning(f"Tool {self.name} - {self.id}: related-memory search failed. Error: {e}")
            hits = []
        related = [hit.path for hit in hits if hit.path != entry.path and hit.score >= RELATED_MIN_SCORE]
        related = related[:RELATED_LIMIT]
        if related:
            message += (
                " Memories on nearby topics: " + ", ".join(related) + ". If one covers the same topic, merge into it."
            )
        return {"content": message}

    def _edit(self, input_data: MemoryStoreToolInputSchema, user_id: str | None = None) -> dict[str, Any]:
        path, find, replace = input_data.path, input_data.find, input_data.replace
        try:
            current = self.backend.read(path, user_id)
        except MemoryNotFoundError:
            raise ToolExecutionException(
                f"No memory at '{path}' to edit. Use action 'write' to create it.", recoverable=True
            ) from None

        # Candidate positions, not str.count: overlapping matches ("---" in "-----") are what
        # make a find string ambiguous, and counting only non-overlapping ones would let an
        # ambiguous edit through and persist it.
        positions = find_positions(current, find)
        if not positions:
            raise ToolExecutionException(
                f"'{find}' does not appear in '{path}', so nothing was changed.", recoverable=True
            )
        if len(positions) > 1 and not input_data.replace_all:
            raise ToolExecutionException(
                f"'{find}' appears {len(positions)} times in '{path}'. Include more surrounding text to "
                f"make it unique, or set 'replace_all' to change every occurrence.",
                recoverable=True,
            )

        updated = current.replace(find, replace) if input_data.replace_all else current.replace(find, replace, 1)
        self.backend.write(path, updated, user_id)
        # How many a replacement actually consumes is str.count's question, not the candidates'.
        changed = current.count(find) if input_data.replace_all else 1
        return {"content": f"Updated '{path}' ({changed} replacement{'s' if changed > 1 else ''})."}

    def _delete(self, path: str, user_id: str | None = None) -> dict[str, Any]:
        deleted = self.backend.delete(path, user_id)
        return {"content": f"Deleted '{path}'." if deleted else f"There was no memory at '{path}'."}


def _describe(entry: MemoryEntry) -> str:
    """One listing line: path, what it holds, and how old it is, since a newer memory supersedes an older one."""
    when = f", {entry.updated_at:%Y-%m-%d}" if entry.updated_at else ""
    about = f" — {entry.description}" if entry.description else ""
    return f"{entry.path}{about} ({entry.size} chars{when})"


def _path_words(path: str) -> str:
    """'playbook/refund-policy.md' -> 'playbook refund policy'."""
    return " ".join(part for part in re.split(r"[/_.\-]+", path.removesuffix(".md")) if part)


def _recently_described(entries: list[MemoryEntry], budget: int) -> set[str]:
    """Paths of the most recently changed memories whose listing lines fit in ``budget`` characters."""
    described: set[str] = set()
    for entry in sorted(entries, key=_changed_at, reverse=True):
        budget -= len(_describe(entry))
        if budget < 0:
            break
        described.add(entry.path)
    return described


def _changed_at(entry: MemoryEntry) -> datetime:
    """When a memory last changed, comparable across stores: naive times are UTC, unknown ones oldest."""
    if entry.updated_at is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return entry.updated_at if entry.updated_at.tzinfo else entry.updated_at.replace(tzinfo=timezone.utc)
