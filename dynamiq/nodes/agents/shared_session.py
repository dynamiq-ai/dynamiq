import re
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING

from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox

# Set by the owning agent's execute(); read by descendant agents at construction.
_shared_session: ContextVar["SharedSession | None"] = ContextVar("dynamiq_shared_session", default=None)

# Per-agent-invocation id (NOT the workflow run_id), used as the page-control key: reentrant for one
# agent's own repeated/concurrent calls, exclusive across different agents.
_current_agent_run: ContextVar[str | None] = ContextVar("dynamiq_current_agent_run", default=None)


def slugify(value: str) -> str:
    """Filesystem-safe slug for per-agent working directories."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return slug or "agent"


class SandboxSharingScope(str, Enum):
    """Which subagents join the owner's shared sandbox when sharing is enabled."""

    ALL = "all"          # every subagent uses the shared sandbox (overrides a subagent's own)
    AUGMENT = "augment"  # only subagents that bring no sandbox of their own


class SharedSession:
    """Resources shared by an agent and its subagents for one run.

    The owner registers its `sandbox_backend`; subagents get a per-agent *view* (same sandbox_id,
    isolated base_path) via `sandbox_view_for`.
    """

    def __init__(
        self,
        *,
        sandbox: "Sandbox | None" = None,
        share_sandbox: bool = False,
        share_browser: bool = False,
        owner_run_id: str = "",
        owner_agent_id: str = "",
        sharing_scope: SandboxSharingScope = SandboxSharingScope.ALL,
    ):
        self.sandbox = sandbox
        # Only view-capable backends (e.g. E2B) can be shared; others degrade to no-sharing.
        self.share_sandbox = bool(share_sandbox and sandbox is not None and getattr(sandbox, "supports_views", False))
        self.share_browser = bool(share_browser)
        self.owner_run_id = owner_run_id
        self.owner_agent_id = owner_agent_id
        self.sharing_scope = sharing_scope
        self._lock = threading.Lock()

        # Browser sharing, two deliberately separate axes:
        #  * _browser_session_id — one live session all agents in the run drive, so cookies/logins
        #    cross immediately with no close. Ended once, at owner teardown, via end_browser_session.
        #  * _browser_context_id — a persistent Context the session loads/persists on end; carries
        #    state to a LATER run only (cross-run), no part in agent-to-agent sharing.
        # Page control still needs serializing (all agents drive the same page): held for an agent's
        # whole turn, released around delegate calls.
        self._browser_session_id: str | None = None
        self._browser_context_id: str | None = None
        self._browser_identity_lock = threading.Lock()  # guards one-time session/context adoption
        self._browser_live_view_url: str | None = None
        self._browser_end_fn: Callable[[], None] | None = None
        # Condition, not Lock: control is not thread-owned (claimed and released on different
        # threads) and one agent's concurrent calls must pass through freely.
        self._page_control_state = threading.Condition()
        self._page_control_key: str | None = None

    def get_sandbox(self) -> "Sandbox | None":
        return self.sandbox

    def sandbox_view_for(self, key: str) -> "Sandbox | None":
        """Return a per-agent view of the shared sandbox with an isolated working dir.

        Materializes the shared sandbox (assigns a sandbox_id) on first call so the
        view can reconnect to the same underlying sandbox rather than create a new one.
        """
        if self.sandbox is None or not getattr(self.sandbox, "supports_views", False):
            return None
        with self._lock:
            sandbox_id = self.sandbox.current_sandbox_id
            if sandbox_id is None:
                self.sandbox.ensure_started()
                sandbox_id = self.sandbox.current_sandbox_id
            workdir = f"{self.sandbox.base_path.rstrip('/')}/work/{slugify(key)}"
            return self.sandbox.create_view(base_path=workdir, sandbox_id=sandbox_id)

    # --- shared live session (intra-run) + persistent Context (cross-run) ---
    def browser_session_id(self) -> str | None:
        return self._browser_session_id

    def adopt_browser_session_id(self, session_id: str) -> str:
        """Record the run's shared session on first use, first-writer-wins; return the effective id."""
        with self._browser_identity_lock:
            if self._browser_session_id is None:
                self._browser_session_id = session_id
            return self._browser_session_id

    def browser_context_id(self) -> str | None:
        return self._browser_context_id

    def adopt_browser_context_id(self, context_id: str) -> str:
        """Record the persistent Context id, first-writer-wins. Cross-run only (see __init__)."""
        with self._browser_identity_lock:
            if self._browser_context_id is None:
                self._browser_context_id = context_id
            return self._browser_context_id

    def browser_live_view_url(self) -> str | None:
        return self._browser_live_view_url

    def set_browser_live_view_url(self, live_view_url: str | None) -> None:
        """Record the shared session's live-view URL so the owner can surface it."""
        if live_view_url and self._browser_live_view_url is None:
            self._browser_live_view_url = live_view_url

    def register_browser_end(self, end_fn: Callable[[], None]) -> None:
        """Register how to end the shared session (called once at owner teardown).

        Independent of any tool instance: a subagent's tool may be GC'd before the run finishes.
        """
        with self._browser_identity_lock:
            if self._browser_end_fn is None:
                self._browser_end_fn = end_fn

    def end_browser_session(self) -> None:
        """End the shared session — owner teardown only, once per run. Also persists the Context."""
        with self._browser_identity_lock:
            end_fn = self._browser_end_fn
            self._browser_end_fn = None
        if end_fn is None:
            return
        try:
            end_fn()
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("Failed to end the shared browser session: %s", exc)

    # --- page control: agents share one page, so only one may drive it at a time ---
    def acquire_page_control(self, agent_run_key: str, timeout: float = 300.0) -> None:
        """Take control of the shared page, blocking until free.

        Idempotent per turn (released once, in execute()'s finally or on delegate), so one agent's
        repeated/concurrent calls pass through. Orders page access only, not state — state is live
        on the shared session.
        """
        deadline = time.monotonic() + timeout
        with self._page_control_state:
            while True:
                if self._page_control_key == agent_run_key:
                    return  # already ours (or a concurrent call of ours just claimed it)
                if self._page_control_key is None:
                    self._page_control_key = agent_run_key
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._page_control_state.wait(timeout=remaining):
                    raise TimeoutError(
                        f"Timed out after {timeout:.0f}s waiting for control of the shared browser "
                        "page; another agent still holds it. Agents share one page, so only one may "
                        "drive it at a time — do not use the browser and delegate browser work in "
                        "the same parallel batch."
                    )

    def release_page_control(self, agent_run_key: str) -> None:
        """Release the shared page to any waiter. Non-holder call is a no-op; nothing is closed."""
        with self._page_control_state:
            if self._page_control_key != agent_run_key:
                return
            self._page_control_key = None
            self._page_control_state.notify_all()


def active_browser_session() -> "SharedSession | None":
    """The active shared session iff browser sharing is on, else None.

    Single accessor for the "is browser sharing active?" guard, used by the agent and Stagehand paths.
    """
    ss = _shared_session.get()
    if ss is None or not ss.share_browser:
        return None
    return ss
