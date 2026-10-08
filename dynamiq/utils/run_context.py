"""Per-run context carried on ContextVars.

Node run id: set by ``Node.run_sync`` / ``Node._run_async_native`` after ``_prepare_execution``
mints ``run_id``, so ``Node._node_run_log`` and related helpers can tag every lifecycle line
(start/finish, retries, errors, skip/cancel) without threading the id through every signature.

Run identity: the ``user_id`` / ``session_id`` a flow run was started with. A node's input
transformer can narrow what it receives (a selector replaces the input), so identity read only
from a node's own input disappears whenever a selector leaves it out. ``Flow`` records it here
at run start; agents fall back to it for memory scoping. ContextVars reach nested nodes,
sub-agents and ``ContextAwareThreadPoolExecutor`` / asyncio workers, but not other processes.
"""

from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any
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


@dataclass(frozen=True)
class RunIdentity:
    """Who a run is for: the end user and the conversation it belongs to."""

    user_id: str | None = None
    session_id: str | None = None

    def __bool__(self) -> bool:
        return bool(self.user_id or self.session_id)


_run_identity: ContextVar[RunIdentity | None] = ContextVar("dynamiq_run_identity", default=None)


def _as_id(value: Any) -> str | None:
    if value is None or isinstance(value, (Mapping, list, tuple, set, bytes)):
        return None
    value = str(value)
    return value or None


def run_identity_from_input(input_data: Any) -> RunIdentity | None:
    """Read ``user_id`` / ``session_id`` off a run's input mapping, or None when it has neither."""
    if not isinstance(input_data, Mapping):
        return None
    identity = RunIdentity(
        user_id=_as_id(input_data.get("user_id")),
        session_id=_as_id(input_data.get("session_id")),
    )
    return identity or None


def set_run_identity_from_input(input_data: Any) -> Token | None:
    """Bind the identity carried by ``input_data`` for the current context.

    Returns a reset token, or None when the input carries no identity -- the context then keeps
    whatever an enclosing run set, so a nested flow started without ids stays on its parent's.
    """
    identity = run_identity_from_input(input_data)
    if identity is None:
        return None
    return _run_identity.set(identity)


def reset_run_identity(token: Token | None) -> None:
    """Restore the previous run identity using the token from ``set_run_identity_from_input``."""
    if token is not None:
        _run_identity.reset(token)


def current_run_identity() -> RunIdentity | None:
    """The identity of the run executing in this context, or None outside a run that has one."""
    return _run_identity.get()
