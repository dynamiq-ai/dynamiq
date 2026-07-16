"""Thin browser-session abstraction shared by an agent and its subagents.

There is no unified Browser abstraction in the SDK today (only the Stagehand
tool). BrowserSession is a minimal holder for one live session's identity so a
SharedSession can hand the same session_id to every agent in a run and close it
once at the end. Browserbase only in P3 (see spec §10.4).
"""

from collections.abc import Callable
from typing import Literal


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
