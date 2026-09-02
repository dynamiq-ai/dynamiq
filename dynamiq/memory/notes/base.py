from abc import ABC, abstractmethod
from datetime import UTC, datetime
from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, computed_field

from dynamiq.memory.notes.schemas import Note, NoteIndexEntry
from dynamiq.memory.notes.types import NoteDeleteStatus, NoteWriteOutcome
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger

MAX_TITLE_LEN = 200
MAX_DESCRIPTION_LEN = 300
MAX_CONTENT_LEN = 100_000

IndexOrder = Literal["title", "updated_at"]
INDEX_ORDERS: tuple[str, ...] = ("title", "updated_at")


class NotesError(Exception):
    """Base exception for note operations."""

    pass


class NoteValidationError(NotesError):
    """Caller supplied an unusable title, description, content or user_id."""

    pass


class NotesStorageError(NotesError):
    """The backing store failed."""

    pass


def _validate_single_line(value: str | None, *, field: str, max_length: int) -> str:
    """Strip and validate a single-line field. Newlines are rejected because they would
    split one entry into two lines in the rendered index."""
    stripped = value.strip() if value else ""
    if not stripped:
        raise NoteValidationError(f"Note {field} cannot be empty")
    if "\n" in stripped or "\r" in stripped:
        raise NoteValidationError(f"Note {field} must be a single line")
    if len(stripped) > max_length:
        raise NoteValidationError(f"Note {field} exceeds {max_length} characters (got {len(stripped)})")
    return stripped


class NotesBackend(ABC, BaseModel):
    """User-scoped storage for titled notes, addressed by `(user_id, title)`.

    Public operations are implemented once here; subclasses supply the storage
    primitives. Unlike `LongTermMemoryBackend` there is no embedder and no similarity
    search — notes are looked up by exact title, and the always-loaded index is what
    makes them discoverable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "notes-backend"
    id: str = Field(default_factory=generate_uuid)
    max_index_entries: int = Field(
        default=200,
        ge=1,
        description="Hard cap on entries returned by `index()`; bounds the injected prompt block.",
    )

    _storage_ensured: bool = PrivateAttr(default=False)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        return self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)

    def _ensure_storage(self) -> None:
        """Provision tables / indexes if absent. No-op by default."""

    def _guarded_ensure(self) -> None:
        """Call `_ensure_storage` at most once per instance; retry on failure."""
        if self._storage_ensured:
            return
        self._ensure_storage()
        self._storage_ensured = True

    def write(
        self,
        *,
        user_id: str,
        title: str,
        description: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Note, NoteWriteOutcome]:
        """Create a note, or replace the existing note with the same title in place."""
        if not user_id:
            raise NoteValidationError("write requires a non-empty user_id")
        title = _validate_single_line(title, field="title", max_length=MAX_TITLE_LEN)
        description = _validate_single_line(description, field="description", max_length=MAX_DESCRIPTION_LEN)
        if not content or not content.strip():
            raise NoteValidationError("Note content cannot be empty")
        if len(content) > MAX_CONTENT_LEN:
            raise NoteValidationError(f"Note content exceeds {MAX_CONTENT_LEN} characters (got {len(content)})")

        try:
            self._guarded_ensure()
            now = datetime.now(UTC)
            candidate = Note(
                user_id=user_id,
                title=title,
                description=description,
                content=content,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
            stored, outcome = self.upsert(candidate)
            logger.debug(f"Notes: {outcome.value} note {title!r} for user={user_id}")
            return stored, outcome
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.write failed for user={user_id}, title={title!r}: {e}")
            raise NotesStorageError(f"Failed to write note: {e}") from e

    def read(self, *, user_id: str, titles: list[str]) -> tuple[list[Note], list[str]]:
        """Fetch notes by exact title. Returns `(found, missing)`; found preserves the
        requested order. A missing title is reported, never raised."""
        if not user_id:
            raise NoteValidationError("read requires a non-empty user_id")

        wanted: list[str] = []
        seen: set[str] = set()
        for raw in titles or []:
            stripped = raw.strip() if raw else ""
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            wanted.append(stripped)
        if not wanted:
            raise NoteValidationError("read requires at least one non-empty title")

        try:
            self._guarded_ensure()
            by_title = {note.title: note for note in self.get_many(user_id=user_id, titles=wanted)}
            found = [by_title[title] for title in wanted if title in by_title]
            missing = [title for title in wanted if title not in by_title]
            logger.debug(f"Notes: read {len(found)}/{len(wanted)} notes for user={user_id}")
            return found, missing
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.read failed for user={user_id}: {e}")
            raise NotesStorageError(f"Failed to read notes: {e}") from e

    def delete(self, *, user_id: str, title: str) -> NoteDeleteStatus:
        """Delete a note by title. Deleting another user's title is a `NOT_FOUND`."""
        if not user_id:
            raise NoteValidationError("delete requires a non-empty user_id")
        title = _validate_single_line(title, field="title", max_length=MAX_TITLE_LEN)
        try:
            self._guarded_ensure()
            deleted = self.delete_by_title(user_id=user_id, title=title)
            logger.debug(f"Notes: delete {title!r} for user={user_id} -> {deleted}")
            return NoteDeleteStatus.DELETED if deleted else NoteDeleteStatus.NOT_FOUND
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.delete failed for user={user_id}, title={title!r}: {e}")
            raise NotesStorageError(f"Failed to delete note: {e}") from e

    def index(
        self,
        *,
        user_id: str,
        limit: int | None = None,
        order_by: IndexOrder = "title",
    ) -> tuple[list[NoteIndexEntry], bool]:
        """Return `(entries, truncated)` for the always-loaded index.

        Never reads note bodies. Truncation is detected by asking for one row more than
        the cap, so there is no second `COUNT(*)` query.
        """
        if not user_id:
            raise NoteValidationError("index requires a non-empty user_id")
        if order_by not in INDEX_ORDERS:
            raise NoteValidationError(f"index order_by must be one of {INDEX_ORDERS}, got {order_by!r}")

        effective = self.max_index_entries if limit is None else min(limit, self.max_index_entries)
        if effective <= 0:
            return [], False

        try:
            self._guarded_ensure()
            rows = self.list_index(user_id=user_id, limit=effective + 1, order_by=order_by)
            truncated = len(rows) > effective
            return rows[:effective], truncated
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.index failed for user={user_id}: {e}")
            raise NotesStorageError(f"Failed to list note index: {e}") from e

    def list_all(self, *, user_id: str, limit: int = 100) -> list[Note]:
        """Return up to `limit` full notes for `user_id` (admin/introspection)."""
        if not user_id:
            raise NoteValidationError("list_all requires a non-empty user_id")
        if limit <= 0:
            return []
        try:
            self._guarded_ensure()
            return self.list_by_user(user_id=user_id, limit=limit)
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.list_all failed for user={user_id}: {e}")
            raise NotesStorageError(f"Failed to list notes: {e}") from e

    def clear_user(self, *, user_id: str) -> int:
        """Hard-delete every note owned by `user_id`; returns the count deleted."""
        if not user_id:
            raise NoteValidationError("clear_user requires a non-empty user_id")
        try:
            self._guarded_ensure()
            deleted = self.delete_user(user_id=user_id)
            logger.debug(f"Notes: cleared {deleted} notes for user={user_id}")
            return deleted
        except NotesError:
            raise
        except Exception as e:
            logger.error(f"Notes.clear_user failed for user={user_id}: {e}")
            raise NotesStorageError(f"Failed to clear notes: {e}") from e

    @abstractmethod
    def upsert(self, note: Note) -> tuple[Note, NoteWriteOutcome]:
        """Atomically create-or-replace by `(user_id, title)`.

        MUST be a single atomic operation — callers hold no lock — and MUST preserve the
        existing row's `created_at` on update. Returns the stored note and the outcome.
        """

    @abstractmethod
    def get_many(self, *, user_id: str, titles: list[str]) -> list[Note]:
        """Return the notes matching any of `titles` for `user_id`, in any order."""

    @abstractmethod
    def delete_by_title(self, *, user_id: str, title: str) -> bool:
        """Delete one note; return True if a row was removed."""

    @abstractmethod
    def list_index(self, *, user_id: str, limit: int, order_by: str) -> list[NoteIndexEntry]:
        """Return up to `limit` index entries. MUST NOT read the `content` column — this
        runs on every agent run."""

    @abstractmethod
    def list_by_user(self, *, user_id: str, limit: int) -> list[Note]:
        """Return up to `limit` full notes for `user_id`."""

    @abstractmethod
    def delete_user(self, *, user_id: str) -> int:
        """Hard-delete every note for `user_id`; return the count deleted."""
