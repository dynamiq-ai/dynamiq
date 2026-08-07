"""Per-node-run correlation id for log lines.

Set by ``Node.run_sync`` / ``Node._run_async_native`` after ``_prepare_execution``
mints ``run_id``, so helpers like ``log_execution_start`` / ``log_execution_finish``
(and private format helpers that never receive ``kwargs``) can tag matching
started/finished lines without threading the id through every signature.
"""

from contextvars import ContextVar, Token
from uuid import UUID

_current_node_run_id: ContextVar[str | None] = ContextVar("dynamiq_node_run_id", default=None)


def set_node_run_id(run_id: UUID | str) -> Token:
    """Bind the active node run id for the current context. Returns a reset token."""
    return _current_node_run_id.set(str(run_id))


def reset_node_run_id(token: Token) -> None:
    """Restore the previous node run id using the token from ``set_node_run_id``."""
    _current_node_run_id.reset(token)


def current_node_run_id() -> str:
    """Short form of the active node run id, or ``-`` when unset."""
    run_id = _current_node_run_id.get()
    return run_id[:8] if run_id else "-"
