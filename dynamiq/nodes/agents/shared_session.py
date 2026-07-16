import re
import threading
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dynamiq.browsers.base import BrowserSession
    from dynamiq.sandboxes.base import Sandbox

# Set by the owning agent's execute(); read by descendant agents at construction.
_shared_session: ContextVar["SharedSession | None"] = ContextVar("dynamiq_shared_session", default=None)

# Per-agent-invocation id (NOT the workflow run_id). Set at the top of each
# Agent.execute and used as the browser-lease key so parallel subagents get
# distinct keys and mutual exclusion holds. See spec §10.4.
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
    """Holds resources shared by an agent and its subagents for one run.

    The owning agent registers its own `sandbox_backend`;
    subagents obtain a per-agent *view* (same sandbox_id,
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
        # Only backends that can produce per-agent views (e.g. E2B) can be shared;
        # others degrade to no-sharing so subagents fall back to their own sandbox.
        self.share_sandbox = bool(share_sandbox and sandbox is not None and getattr(sandbox, "supports_views", False))
        self.share_browser = bool(share_browser)
        self.owner_run_id = owner_run_id
        self.owner_agent_id = owner_agent_id
        self.sharing_scope = sharing_scope
        self._lock = threading.Lock()

        # browser (Model A): one shared session + an exclusive, reentrant, per-agent-run lease
        self._browser: "BrowserSession | None" = None
        self._browser_lock = threading.Lock()
        self._lease_cond = threading.Condition()
        self._lease_owner: str | None = None

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

    # --- browser identity (first writer wins) ---
    def browser_session_id(self) -> str | None:
        return self._browser.session_id if self._browser else None

    def browser_live_view_url(self) -> str | None:
        return self._browser.live_view_url if self._browser else None

    def record_browser(
        self,
        *,
        session_id: str,
        provider: Literal["browserbase", "steel"] = "browserbase",
        live_view_url: str | None = None,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        """Record the shared session's identity the first time a tool creates it."""
        from dynamiq.browsers.base import BrowserSession

        with self._browser_lock:
            if self._browser is not None and self._browser.session_id:
                return  # first writer wins
            self._browser = BrowserSession(
                provider=provider,
                session_id=session_id,
                live_view_url=live_view_url,
                close_callback=close_callback,
            )

    # --- Model A lease (exclusive, reentrant per agent-run, baton hand-off) ---
    def acquire_browser(self, agent_run_key: str) -> None:
        """Block until this run holds the browser lease (reentrant for the same run).

        Deadlock invariant: the lease is exclusive and held for the WHOLE agent-run
        (released only in the owning agent's execute() finally). An agent that drives
        the browser directly must NOT hold the lease while awaiting a subagent that also
        needs it — orchestrate/delegate rather than co-drive, or the two will deadlock.
        """
        with self._lease_cond:
            while self._lease_owner is not None and self._lease_owner != agent_run_key:
                self._lease_cond.wait()
            self._lease_owner = agent_run_key

    def release_browser(self, agent_run_key: str) -> None:
        """Release the lease only if this run holds it; wake any waiters."""
        with self._lease_cond:
            if self._lease_owner == agent_run_key:
                self._lease_owner = None
                self._lease_cond.notify_all()

    def close_browser(self) -> None:
        """Owner-only: close the shared live session at run end."""
        with self._browser_lock:
            if self._browser is not None:
                self._browser.close()
