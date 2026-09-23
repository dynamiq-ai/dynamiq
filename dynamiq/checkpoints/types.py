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


class RunPausedException(Exception):
    """Stops a node that has paused its run to wait, and explains a paused flow's result.

    It is control flow, not a failure: the checkpoint records what the run completed and which nodes
    wait, and resuming from it continues the run. A flow that pauses returns a FAILURE result carrying
    this exception, with its checkpoint in PENDING_INPUT, which is how human-input pauses have always
    been reported to callers.
    """

    def __init__(self, message: str = "Run paused", resume_at: datetime | None = None):
        super().__init__(message)
        self.resume_at = resume_at
