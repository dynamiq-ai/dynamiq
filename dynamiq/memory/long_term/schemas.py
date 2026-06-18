from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A single long-term memory fact, scoped to a user."""

    id: str
    content: str
    hash: str
    user_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
