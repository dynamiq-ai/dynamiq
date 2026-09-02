from enum import Enum


class NoteWriteOutcome(str, Enum):
    """Outcome of `NotesBackend.write()` — distinguishes an insert from an overwrite."""

    CREATED = "created"
    UPDATED = "updated"


class NoteDeleteStatus(str, Enum):
    """Outcome of `NotesBackend.delete()`.

    There is no `FORBIDDEN` member: every query is scoped `WHERE user_id = ...`, so a
    cross-user delete is structurally impossible rather than checked at runtime.
    """

    DELETED = "deleted"
    NOT_FOUND = "not_found"
