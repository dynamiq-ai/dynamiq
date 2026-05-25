from enum import Enum


class ForgetStatus(str, Enum):
    """Outcome of `LongTermMemory.forget()`."""

    DELETED = "deleted"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"


class MemoryToolKind(str, Enum):
    """Kinds of long-term-memory tools exposed to an agent."""

    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
