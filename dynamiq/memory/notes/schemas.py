from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Note(BaseModel):
    """A titled note owned by a single user.

    `(user_id, title)` is the natural key — there is no surrogate id. Length limits are
    enforced on the write path (see `dynamiq.memory.notes.base`), never here, so an
    existing row stays deserializable if those limits are ever tightened.
    """

    user_id: str
    title: str
    description: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class NoteIndexEntry(BaseModel):
    """One line of the always-loaded note index.

    Deliberately has no `content` field: this is what gets injected into the agent's
    system prompt on every run, and it must never carry note bodies.
    """

    title: str
    description: str
    updated_at: datetime
