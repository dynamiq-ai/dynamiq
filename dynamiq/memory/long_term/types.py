from enum import Enum


class ForgetStatus(str, Enum):
    """Outcome of `LongTermMemoryBackend.forget()` (programmatic API only)."""

    DELETED = "deleted"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"


class RememberOutcome(str, Enum):
    """Outcome of `LongTermMemoryBackend.remember()` — distinguishes insert from upsert."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


class MemoryToolKind(str, Enum):
    """Kinds of long-term-memory tools exposed to an agent."""

    REMEMBER = "remember"
    RECALL = "recall"
