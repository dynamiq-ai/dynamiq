"""Sandbox-related exceptions."""


class SandboxConnectionError(Exception):
    """Raised when connecting to an existing sandbox fails after retries."""

    def __init__(self, sandbox_id: str, cause: Exception | None = None):
        self.sandbox_id = sandbox_id
        self.cause = cause
        super().__init__(
            f"Failed to connect to E2B sandbox {sandbox_id}. "
            "The sandbox may have been killed or expired." + (f" Cause: {cause}" if cause else "")
        )
