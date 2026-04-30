import threading
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from dynamiq.runnables import RunnableConfig


class CancellationToken:
    """Thread-safe cancellation signal shared across the execution hierarchy.

    A runtime mutable object (like Queue in StreamingConfig or
    CheckpointContext in CheckpointConfig) that allows external callers
    to signal cancellation to a running workflow/flow/node.

    Usage:
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))

        # From another thread (API handler, UI callback, etc.):
        token.cancel()

    Or grab the default token from the auto-created config:
        config = RunnableConfig()
        config.cancellation.token.cancel()
    """

    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        """Signal cancellation. Thread-safe, idempotent."""
        with self._lock:
            self._event.set()

    @property
    def is_canceled(self) -> bool:
        return self._event.is_set()


class CanceledException(Exception):
    """Raised when a CancellationToken is triggered."""

    def __init__(self):
        super().__init__()


class CancellationConfig(BaseModel):
    """Cancellation configuration for a run.

    Always active — cancellation is a core, non-optional feature. Every
    RunnableConfig gets a CancellationConfig with a fresh CancellationToken
    by default, so callers can cancel any workflow without setup.

    To override with your own token (e.g. one shared across multiple runs):
        token = CancellationToken()
        config = RunnableConfig(cancellation=CancellationConfig(token=token))
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    token: CancellationToken = Field(
        default_factory=CancellationToken,
        description="Shared cancellation signal. Set externally, checked internally.",
    )

    def check(self) -> None:
        """Raise CanceledException if the token has been signaled. Otherwise, no-op."""
        if self.token.is_canceled:
            raise CanceledException()

    @property
    def is_canceled(self) -> bool:
        """Check cancellation state without raising."""
        return self.token.is_canceled

    def to_dict(self, for_tracing: bool = False, **kwargs) -> dict:
        return {"canceled": self.token.is_canceled}


def check_cancellation(config: "RunnableConfig | None") -> None:
    """Convenience helper: check cancellation from a RunnableConfig.

    No-op if config is None or cancellation is not configured.
    Raises CanceledException when the token has been signaled.
    """
    if config and config.cancellation:
        config.cancellation.check()
