"""Pydantic schemas for long-term memory."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A single long-term memory fact, scoped to a user.

    `hash` is md5(f"{user_id}:{content.strip().lower()}") and is used to
    short-circuit exact duplicates in `LongTermMemory.remember()` before
    any embedder call.
    """

    id: str
    content: str
    hash: str
    user_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
