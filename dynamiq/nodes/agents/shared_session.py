import re
import threading
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox

# Set by the owning agent's execute(); read by descendant agents at construction.
_shared_session: ContextVar["SharedSession | None"] = ContextVar("dynamiq_shared_session", default=None)

# Per-agent-invocation id (NOT the workflow run_id). Set at the top of each
# Agent.execute and used as the browser-lease key so parallel subagents get
# distinct keys and mutual exclusion holds. See spec §10.4.
_current_agent_run: ContextVar[str | None] = ContextVar("dynamiq_current_agent_run", default=None)

# Chain of run-keys from the root agent down to self (root->...->self). Set at the
# top of each Agent.execute. Used only when acquiring the browser lease so a nested
# run knows its ancestors and may borrow a lease currently held by an ancestor.
_agent_run_chain: ContextVar[tuple[str, ...] | None] = ContextVar("dynamiq_agent_run_chain", default=None)


def slugify(value: str) -> str:
    """Filesystem-safe slug for per-agent working directories."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return slug or "agent"


class SandboxSharingScope(str, Enum):
    """Which subagents join the owner's shared sandbox when sharing is enabled."""

    ALL = "all"          # every subagent uses the shared sandbox (overrides a subagent's own)
    AUGMENT = "augment"  # only subagents that bring no sandbox of their own


# Thin browser-session abstraction shared by an agent and its subagents.
#
# There is no unified Browser abstraction in the SDK today (only the Stagehand
# tool). BrowserSession is a minimal holder for one live session's identity so a
# SharedSession can hand the same session_id to every agent in a run and close it
# once at the end. Browserbase only in P3 (see spec §10.4).
class BrowserSession:
    """Identity + teardown handle for one shared live browser session."""

    def __init__(
        self,
        *,
        provider: Literal["browserbase", "steel"],
        session_id: str | None = None,
        live_view_url: str | None = None,
        close_callback: Callable[[], None] | None = None,
    ):
        self.provider = provider
        self.session_id = session_id
        self.live_view_url = live_view_url
        self._close_callback = close_callback
        self._closed = False

    def close(self) -> None:
        """Release the live session exactly once (owner-only, at run end)."""
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()


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

        # browser (Model A): one shared session + a lease that is reentrant down the ancestor
        # chain. The lease holders form a LIFO stack (root owner at the bottom, current driver on
        # top); a nested run borrows from an ancestor already on the stack, parallel siblings serialize.
        self._browser: "BrowserSession | None" = None
        self._browser_lock = threading.Lock()
        self._lease_cond = threading.Condition()
        self._lease_stack: list[str] = []

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
        with self._browser_lock:
            if self._browser is not None and self._browser.session_id:
                return  # first writer wins
            self._browser = BrowserSession(
                provider=provider,
                session_id=session_id,
                live_view_url=live_view_url,
                close_callback=close_callback,
            )

    # --- Model A lease (reentrant down the ancestor chain, LIFO stack, baton hand-off) ---
    def acquire_browser(self, agent_run_key: str, ancestor_keys: tuple[str, ...] = ()) -> None:
        """Block until this run may drive the browser, then push it onto the lease stack.

        Reentrant-chain semantics: a run whose ancestor currently holds the lease may borrow
        it (that ancestor is blocked awaiting this nested call, so still only one run drives at
        a time) — this lets an owner co-drive the browser AND delegate to subagents without
        deadlock. Genuinely-parallel siblings (whose holder is neither self nor an ancestor)
        serialize. Reentry by the same run does not push a duplicate.
        """
        ancestors = set(ancestor_keys)
        with self._lease_cond:
            # Block only while the current driver (top of stack) is NEITHER me NOR one of my
            # ancestors. A held-by-ancestor lease is borrowable because that ancestor is blocked
            # awaiting me (nested call), so only one run actually drives at a time.
            while (
                self._lease_stack and self._lease_stack[-1] != agent_run_key and self._lease_stack[-1] not in ancestors
            ):
                self._lease_cond.wait()
            if not self._lease_stack or self._lease_stack[-1] != agent_run_key:
                self._lease_stack.append(agent_run_key)  # reentrant same-run = no duplicate push

    def release_browser(self, agent_run_key: str) -> None:
        """Pop the lease if this run is the current driver (LIFO); wake any waiters."""
        with self._lease_cond:
            if self._lease_stack and self._lease_stack[-1] == agent_run_key:
                self._lease_stack.pop()
                self._lease_cond.notify_all()

    def close_browser(self) -> None:
        """Owner-only: close the shared live session at run end."""
        with self._browser_lock:
            if self._browser is not None:
                self._browser.close()
