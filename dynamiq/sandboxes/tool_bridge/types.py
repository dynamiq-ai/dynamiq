from contextvars import ContextVar
from typing import Any, Callable, Protocol

from pydantic import BaseModel, Field


class ToolBridgeConfig(BaseModel):
    """Lets the code a sandbox-shell command runs call the agent's own tools.

    The agent process runs each call through its tool path, as a call of the model, so credentials and per-run
    parameters stay out of the sandbox. Every check runs there: any process in the sandbox can use the bridge of
    the command that is running, since all of them run as one user.
    """

    enabled: bool = Field(default=False, description="Offer the agent's tools to code in sandbox-shell commands.")
    max_calls_per_command: int = Field(default=50, ge=1, description="Requests of any kind one command may make.")
    max_concurrent_calls: int = Field(default=8, ge=1, description="Requests run at the same time.")
    max_request_bytes: int = Field(default=1024 * 1024, ge=1024, description="Largest request accepted.")
    max_reply_bytes: int = Field(default=16 * 1024 * 1024, ge=1024, description="Largest result passed back.")
    ready_timeout_seconds: float = Field(default=30, gt=0, description="How long to wait for the bridge to start.")


class ToolBridgeHandler(Protocol):
    """Answers one request of code in the sandbox. Never raises: a failure is a reply with ``ok`` false."""

    def handle(self, request: dict[str, Any]) -> dict[str, Any]: ...


# Set by an agent around a run of its own sandbox-shell tool: builds the handler for that command, given the
# shell call's run id, the parent of the bridged calls in traces.
current_tool_bridge_factory: ContextVar[Callable[..., ToolBridgeHandler] | None] = ContextVar(
    "current_tool_bridge_factory", default=None
)

# True while an agent runs a call that came from code in the sandbox.
in_bridged_call: ContextVar[bool] = ContextVar("in_bridged_call", default=False)


def refusal(error: str, reason: str | None = None) -> dict[str, Any]:
    """A reply for a request the bridge refused. ``reason`` is fixed text, never tool output."""
    reply: dict[str, Any] = {"ok": False, "error": error}
    if reason:
        reply["reason"] = reason
    return reply
