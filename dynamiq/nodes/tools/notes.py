import difflib
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.memory.notes import (
    MAX_CONTENT_LEN,
    MAX_DESCRIPTION_LEN,
    MAX_TITLE_LEN,
    NoteDeleteStatus,
    NotesBackend,
    NotesError,
    NotesStorageError,
    NoteValidationError,
    NoteWriteOutcome,
    render_notes,
    render_notes_index,
)
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

WRITE_NOTE_DESCRIPTION = """Save a titled note that persists across conversations.

- The title is the key: writing an existing title REPLACES that note entirely.
- To extend a note, read_note first, then write back the merged content.
- Returns a confirmation plus your refreshed note index.

Example: {"title": "Deploy runbook", "description": "staging->prod steps and rollback",
"content": "1. Merge to main\\n2. ./deploy staging"}
"""

READ_NOTE_DESCRIPTION = """Load the full content of saved notes, by exact title from your note index.

- Pass several titles at once rather than one call each.
- An unknown title is reported with a suggestion, not an error.
- Returns each note's content under a `## title` heading.

Example: {"titles": ["Deploy runbook", "API conventions"]}
"""

DELETE_NOTE_DESCRIPTION = """Permanently delete a note by exact title. There is no undo.

- To correct a note instead, use write_note with the same title — that replaces it in place.
- Returns a confirmation plus your refreshed note index.

Example: {"title": "Deploy runbook"}
"""

_SUGGESTION_CUTOFF = 0.6


def _single_line(value: str, field: str, max_length: int) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"`{field}` must not be empty or whitespace-only")
    if "\n" in stripped or "\r" in stripped:
        raise ValueError(f"`{field}` must be a single line")
    if len(stripped) > max_length:
        raise ValueError(f"`{field}` must be at most {max_length} characters")
    return stripped


class WriteNoteInputSchema(BaseModel):
    """LLM-visible input for `write_note`. `user_id` is bound at construction."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TITLE_LEN,
        description="Single-line title; it is the key. An existing title is replaced.",
    )
    description: str = Field(
        ...,
        min_length=1,
        max_length=MAX_DESCRIPTION_LEN,
        description="Single-line summary shown in your note index from now on.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CONTENT_LEN,
        description="Full body of the note; markdown is fine.",
    )

    @field_validator("title", "description", mode="after")
    @classmethod
    def _validate_single_line(cls, value: str, info) -> str:
        max_length = MAX_TITLE_LEN if info.field_name == "title" else MAX_DESCRIPTION_LEN
        return _single_line(value, info.field_name, max_length)


class ReadNoteInputSchema(BaseModel):
    """LLM-visible input for `read_note`. `user_id` is bound at construction."""

    titles: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Exact titles copied from your note index; up to 10 per call.",
    )

    @field_validator("titles", mode="after")
    @classmethod
    def _strip_dedupe_and_require_nonblank(cls, titles: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for title in titles:
            stripped = title.strip()
            if not stripped:
                raise ValueError("`titles` must not contain empty or whitespace-only strings")
            if stripped not in seen:
                seen.add(stripped)
                cleaned.append(stripped)
        return cleaned


class DeleteNoteInputSchema(BaseModel):
    """LLM-visible input for `delete_note`. `user_id` is bound at construction."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TITLE_LEN,
        description="Exact title of the note to delete, from your note index.",
    )

    @field_validator("title", mode="after")
    @classmethod
    def _validate_single_line(cls, value: str) -> str:
        return _single_line(value, "title", MAX_TITLE_LEN)


class _NotesTool(Node):
    """Shared base for the note tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    is_mockable: ClassVar[bool] = False
    backend: NotesBackend
    user_id: str

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"backend": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, **kwargs)
        data["backend"] = self.backend.to_dict(include_secure_params=include_secure_params, **kwargs)
        return data

    def _start_run(self, config: RunnableConfig | None, **kwargs) -> None:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

    def _index_entries(self):
        return self.backend.index(user_id=self.user_id)

    def _index_block(self) -> str:
        """The refreshed index, echoed after every mutation so the model never diverges
        from the snapshot injected into its system prompt at the start of the run."""
        entries, truncated = self._index_entries()
        return render_notes_index(entries, truncated=truncated, empty="(none)")

    def _fail(self, error: NotesError) -> None:
        """Raise a tool error, recoverable only when the model can actually fix it.

        A `NoteValidationError` is about the arguments, so a retry can succeed. Storage
        failures (unreachable database, bad DDL) are not: marking those recoverable makes
        the agent retry an identical call until it burns through its loop limit, and the
        LLM has no way to see the real cause.
        """
        recoverable = isinstance(error, NoteValidationError)
        logger.error(f"Tool {self.name} - {self.id}: failed ({'recoverable' if recoverable else 'fatal'}): {error}")
        if recoverable:
            raise ToolExecutionException(
                f"Tool '{self.name}' failed. Error: {error}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )
        # NOT a ToolExecutionException: `Node._handle_failure` derives recoverability from
        # `isinstance(e, RecoverableAgentException)` and ignores the `recoverable=False`
        # argument, so raising one here would be retried anyway.
        raise NotesStorageError(f"Tool '{self.name}' failed and cannot be retried: {error}") from error


class WriteNoteTool(_NotesTool):
    """Create a note, or replace the note with the same title, for the bound user_id."""

    name: str = "write_note"
    description: str = WRITE_NOTE_DESCRIPTION
    input_schema: ClassVar[type[WriteNoteInputSchema]] = WriteNoteInputSchema

    def execute(
        self, input_data: WriteNoteInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        self._start_run(config, **kwargs)

        try:
            note, outcome = self.backend.write(
                user_id=self.user_id,
                title=input_data.title,
                description=input_data.description,
                content=input_data.content,
            )
            verb = "Created" if outcome is NoteWriteOutcome.CREATED else "Overwrote"
            return {"content": f"{verb} note {note.title!r}.\n\nYour notes:\n{self._index_block()}"}
        except NotesError as e:
            self._fail(e)


class ReadNoteTool(_NotesTool):
    """Load full note bodies by exact title, scoped to the bound user_id."""

    name: str = "read_note"
    description: str = READ_NOTE_DESCRIPTION
    input_schema: ClassVar[type[ReadNoteInputSchema]] = ReadNoteInputSchema

    def execute(
        self, input_data: ReadNoteInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        self._start_run(config, **kwargs)

        try:
            found, missing = self.backend.read(user_id=self.user_id, titles=input_data.titles)
            if not missing:
                return {"content": render_notes(found)}
            # A near-miss on a hand-copied title is the dominant failure here, so spend one
            # index read to let the model self-correct within this same result.
            entries, truncated = self._index_entries()
        except NotesError as e:
            self._fail(e)

        parts = [render_notes(found)] if found else []
        parts.append(self._miss_report(missing, entries))
        if entries:
            parts.append("Your notes:\n" + render_notes_index(entries, truncated=truncated))
        else:
            parts.append("You have no notes yet.")
        return {"content": "\n\n".join(part for part in parts if part)}

    @staticmethod
    def _miss_report(missing: list[str], entries: list) -> str:
        available = [entry.title for entry in entries]
        lowered = {title.lower(): title for title in available}
        lines = []
        for title in missing:
            suggestion = lowered.get(title.lower())
            if suggestion is None:
                close = difflib.get_close_matches(title, available, n=1, cutoff=_SUGGESTION_CUTOFF)
                suggestion = close[0] if close else None
            if suggestion is not None:
                lines.append(f"Not found: {title!r}. Did you mean {suggestion!r}?")
            else:
                lines.append(f"Not found: {title!r}.")
        return "\n".join(lines)


class DeleteNoteTool(_NotesTool):
    """Permanently delete a note by title, scoped to the bound user_id."""

    name: str = "delete_note"
    description: str = DELETE_NOTE_DESCRIPTION
    input_schema: ClassVar[type[DeleteNoteInputSchema]] = DeleteNoteInputSchema

    def execute(
        self, input_data: DeleteNoteInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        self._start_run(config, **kwargs)

        try:
            status = self.backend.delete(user_id=self.user_id, title=input_data.title)
            if status is NoteDeleteStatus.DELETED:
                headline = f"Deleted note {input_data.title!r}."
            else:
                headline = f"No note titled {input_data.title!r} — nothing was deleted."
            return {"content": f"{headline}\n\nYour notes:\n{self._index_block()}"}
        except NotesError as e:
            self._fail(e)


def build_notes_tools(*, backend: NotesBackend, user_id: str) -> list[Node]:
    """Construct the note tools (write + read + delete) with `user_id` baked in."""
    return [
        WriteNoteTool(backend=backend, user_id=user_id),
        ReadNoteTool(backend=backend, user_id=user_id),
        DeleteNoteTool(backend=backend, user_id=user_id),
    ]
