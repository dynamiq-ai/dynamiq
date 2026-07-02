import re
import threading
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox

# Set by the owning agent's execute(); read by descendant agents at construction.
_shared_session: ContextVar["SharedSession | None"] = ContextVar("dynamiq_shared_session", default=None)


def slugify(value: str) -> str:
    """Filesystem-safe slug for per-agent working directories."""
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return slug or "agent"


class SharedSession:
    """Holds resources shared by an agent and its subagents for one run.

    The owning agent registers its own `sandbox_backend`;
    subagents obtain a per-agent *view* (same sandbox_id,
    isolated base_path) via `sandbox_view_for`.
    """

    def __init__(self, *, sandbox: "Sandbox | None" = None, share_sandbox: bool = False, owner_run_id: str = ""):
        self.sandbox = sandbox
        # Only backends that can produce per-agent views (e.g. E2B) can be shared;
        # others degrade to no-sharing so subagents fall back to their own sandbox.
        self.share_sandbox = bool(share_sandbox and sandbox is not None and getattr(sandbox, "supports_views", False))
        self.owner_run_id = owner_run_id
        self._lock = threading.Lock()

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
