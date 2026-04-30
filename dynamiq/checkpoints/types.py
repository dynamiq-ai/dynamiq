from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class CheckpointStatus(str, Enum):
    """Checkpoint execution status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    PENDING_INPUT = "pending_input"
