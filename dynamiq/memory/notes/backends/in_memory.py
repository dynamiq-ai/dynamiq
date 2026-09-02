import threading

from pydantic import PrivateAttr

from dynamiq.memory.notes.base import NotesBackend
from dynamiq.memory.notes.schemas import Note, NoteIndexEntry
from dynamiq.memory.notes.types import NoteWriteOutcome


class InMemoryNotesBackend(NotesBackend):
    """Dict-backed notes store. Loses data on restart; intended for tests and demos."""

    name: str = "in-memory-notes-backend"

    # user_id -> title -> Note, mirroring the composite primary key of the SQL backend
    # so every operation is scoped without scanning other users' notes.
    _notes: dict[str, dict[str, Note]] = PrivateAttr(default_factory=dict)
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def upsert(self, note: Note) -> tuple[Note, NoteWriteOutcome]:
        with self._lock:
            user_notes = self._notes.setdefault(note.user_id, {})
            existing = user_notes.get(note.title)
            if existing is None:
                user_notes[note.title] = note
                return note, NoteWriteOutcome.CREATED

            # Mirror the Postgres `DO UPDATE SET` column list exactly: created_at survives.
            updated = existing.model_copy(
                update={
                    "description": note.description,
                    "content": note.content,
                    "metadata": note.metadata,
                    "updated_at": note.updated_at,
                }
            )
            user_notes[note.title] = updated
            return updated, NoteWriteOutcome.UPDATED

    def get_many(self, *, user_id: str, titles: list[str]) -> list[Note]:
        with self._lock:
            user_notes = self._notes.get(user_id, {})
            return [user_notes[title] for title in titles if title in user_notes]

    def delete_by_title(self, *, user_id: str, title: str) -> bool:
        with self._lock:
            user_notes = self._notes.get(user_id)
            if not user_notes:
                return False
            return user_notes.pop(title, None) is not None

    def list_index(self, *, user_id: str, limit: int, order_by: str) -> list[NoteIndexEntry]:
        if limit <= 0:
            return []
        with self._lock:
            notes = list(self._notes.get(user_id, {}).values())
        notes = self._sorted(notes, order_by)
        return [
            NoteIndexEntry(title=note.title, description=note.description, updated_at=note.updated_at)
            for note in notes[:limit]
        ]

    def list_by_user(self, *, user_id: str, limit: int) -> list[Note]:
        if limit <= 0:
            return []
        with self._lock:
            notes = list(self._notes.get(user_id, {}).values())
        return self._sorted(notes, "title")[:limit]

    def delete_user(self, *, user_id: str) -> int:
        with self._lock:
            return len(self._notes.pop(user_id, {}))

    @staticmethod
    def _sorted(notes: list[Note], order_by: str) -> list[Note]:
        if order_by == "updated_at":
            return sorted(notes, key=lambda note: (-note.updated_at.timestamp(), note.title))
        return sorted(notes, key=lambda note: note.title)
