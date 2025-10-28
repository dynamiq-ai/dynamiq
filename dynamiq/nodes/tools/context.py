from typing import Any, Callable

from dynamiq.auth import AuthConfig, AuthRequest


class ToolExecutionContext:
    """
    Lightweight context passed to tools for auth and per-tool state handling.

    The context intentionally keeps a narrow surface:
    - `state` is a mutable dict that tools can reuse across invocations.
    - `get_auth()` exposes any pre-configured auth payload provided by the agent.
    - `request_auth()` lets a tool signal that additional credentials are required.
    """

    def __init__(
        self,
        *,
        auth_payload: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        request_auth_callback: Callable[[AuthRequest], None] | None = None,
    ) -> None:
        self._auth_payload = auth_payload or {}
        self.state = state or {}
        self._request_auth_callback = request_auth_callback
        self._requested_auth_request: AuthRequest | None = None

    def get_auth(self) -> dict[str, Any] | None:
        """Return the auth payload, if one was supplied."""
        return self._auth_payload or None

    def request_auth(
        self,
        auth_payload: AuthConfig | dict[str, Any] | AuthRequest,
        *,
        message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AuthRequest:
        """
        Signal that the tool needs authentication.

        The payload is stored locally and forwarded to the supplied callback.
        """
        if isinstance(auth_payload, AuthRequest):
            request = auth_payload
        else:
            request = AuthRequest(
                required=auth_payload if isinstance(auth_payload, AuthConfig) else None,
                extra={} if extra is None else extra,
                message=message,
            )
            if isinstance(auth_payload, dict):
                request.extra = request.extra | {"payload": auth_payload}
        self._requested_auth_request = request
        if self._request_auth_callback:
            self._request_auth_callback(request)
        return request

    @property
    def requested_auth(self) -> AuthRequest | None:
        """Auth request emitted during this invocation, if any."""
        return self._requested_auth_request
