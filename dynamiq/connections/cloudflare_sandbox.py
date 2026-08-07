"""HTTP client for Cloudflare Sandboxes reached through the official Bridge Worker.

Cloudflare Sandboxes have no public REST API and no Python SDK: sandboxes are only
reachable through a Worker deployed in the user's Cloudflare account. The official
path for external clients is the Bridge Worker (``@cloudflare/sandbox/bridge``),
which exposes a versioned HTTP API under ``/v1`` authenticated with
``Authorization: Bearer <SANDBOX_API_KEY>``.

This module implements that wire contract with plain ``requests`` (no extra
dependencies):

- ``POST /v1/sandbox`` -> ``{"id": "<base32 id>"}`` (container is allocated lazily
  on the first operation).
- ``POST /v1/sandbox/:id/exec`` with ``{"argv": [...], "timeout_ms"?, "cwd"?}``,
  streamed back as SSE: ``event: stdout|stderr`` carry base64 chunks,
  ``event: exit`` carries ``{"exit_code": N}``, ``event: error`` carries
  ``{"error", "code"}``. An optional ``Session-Id`` header scopes the command to a
  previously created execution session.
- ``GET|PUT /v1/sandbox/:id/file/<path>`` for raw file bytes. All paths are
  constrained to ``/workspace`` by the bridge.
- ``GET /v1/sandbox/:id/running``, ``DELETE /v1/sandbox/:id``, session and tunnel
  management routes.
"""

import base64
import json
import re
from typing import Any
from urllib.parse import quote

import requests
from pydantic import BaseModel

from dynamiq.utils.logger import logger

WORKSPACE_ROOT = "/workspace"
SANDBOX_ID_PATTERN = re.compile(r"[a-z2-7]{1,128}")
SANDBOX_MARKER_PATH = f"{WORKSPACE_ROOT}/.dynamiq/created"


class CloudflareSandboxError(Exception):
    """Error returned by the Cloudflare sandbox bridge."""

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None):
        self.code = code
        self.status_code = status_code
        super().__init__(message)


class CloudflareRateLimitError(CloudflareSandboxError):
    """Raised when the bridge reports capacity or rate limiting.

    Maps HTTP 429 responses and 503 responses with the ``capacity_exceeded`` code
    (Cloudflare Containers instance limit reached), both of which are retryable.
    """


class CloudflareExecResult(BaseModel):
    """Result of a command executed through the bridge."""

    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


def resolve_workspace_path(path: str) -> str:
    """POSIX-normalize a path and ensure it stays inside /workspace.

    Mirrors the bridge-side validation so invalid paths fail fast locally with a
    clear error instead of an HTTP 403.

    Args:
        path: Absolute or /workspace-relative path.

    Returns:
        The normalized absolute path.

    Raises:
        ValueError: If the path escapes /workspace.
    """
    abs_path = path if path.startswith("/") else f"{WORKSPACE_ROOT}/{path}"
    parts: list[str] = []
    for segment in abs_path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if parts:
                parts.pop()
        else:
            parts.append(segment)
    resolved = "/" + "/".join(parts)
    if resolved == WORKSPACE_ROOT or resolved.startswith(f"{WORKSPACE_ROOT}/"):
        return resolved
    raise ValueError(
        f"Cloudflare sandbox file paths must stay inside {WORKSPACE_ROOT} " f"(got {path!r}, resolved to {resolved!r})"
    )


class CloudflareSandboxClient:
    """Minimal HTTP client for a deployed Cloudflare sandbox bridge Worker.

    Args:
        url: Base URL of the deployed bridge Worker (e.g. ``https://my-bridge.workers.dev``).
        api_key: The Worker's ``SANDBOX_API_KEY`` secret. Optional only when the
            Worker was deployed without one (auth disabled).
        api_prefix: Route prefix configured in the bridge (default ``/v1``).
        connect_timeout: TCP connect timeout in seconds.
        read_timeout: Default read timeout in seconds for non-exec requests. The
            first operation on a sandbox may block while the container boots
            (up to ~2 minutes inside the Worker), so this needs generous headroom.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        api_prefix: str = "/v1",
        connect_timeout: float = 30.0,
        read_timeout: float = 300.0,
    ):
        if not url:
            raise ValueError(
                "Cloudflare sandbox bridge URL is required. Deploy the bridge Worker "
                "(https://developers.cloudflare.com/sandbox/bridge/) and pass its URL."
            )
        self.base_url = url.rstrip("/")
        self.api_prefix = f"/{api_prefix.strip('/')}" if api_prefix else ""
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _url(self, path: str) -> str:
        return f"{self.base_url}{self.api_prefix}{path}"

    @staticmethod
    def _validate_sandbox_id(sandbox_id: str) -> str:
        if not SANDBOX_ID_PATTERN.fullmatch(sandbox_id or ""):
            raise ValueError(f"Invalid Cloudflare sandbox id {sandbox_id!r}: expected ^[a-z2-7]{{1,128}}$")
        return sandbox_id

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        if 300 <= response.status_code < 400:
            raise CloudflareSandboxError(
                f"Cloudflare sandbox bridge returned a redirect (HTTP {response.status_code}). "
                "Use the final https:// Worker URL as the connection url.",
                status_code=response.status_code,
            )
        if response.status_code < 400:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        message = payload.get("error") or f"Cloudflare sandbox bridge returned HTTP {response.status_code}"
        code = payload.get("code")
        if response.status_code == 429 or code == "capacity_exceeded":
            raise CloudflareRateLimitError(message, code=code, status_code=response.status_code)
        raise CloudflareSandboxError(message, code=code, status_code=response.status_code)

    def _request(self, method: str, path: str, *, read_timeout: float | None = None, **kwargs) -> requests.Response:
        timeout = (self.connect_timeout, read_timeout or self.read_timeout)
        response = self._session.request(method, self._url(path), timeout=timeout, allow_redirects=False, **kwargs)
        self._raise_for_error(response)
        return response

    def health(self) -> bool:
        """Check bridge health (unauthenticated ``GET /health`` at the Worker root)."""
        try:
            response = self._session.get(f"{self.base_url}/health", timeout=(self.connect_timeout, self.read_timeout))
            return response.status_code == 200 and response.json().get("ok") is True
        except (requests.RequestException, ValueError) as e:
            logger.debug(f"Cloudflare sandbox bridge health check failed: {e}")
            return False

    def create_sandbox(self) -> str:
        """Create a new sandbox and return its id (container is allocated lazily)."""
        response = self._request("POST", "/sandbox")
        sandbox_id = response.json()["id"]
        logger.debug(f"Cloudflare sandbox created: {sandbox_id}")
        return sandbox_id

    def destroy_sandbox(self, sandbox_id: str) -> None:
        """Destroy the sandbox container and release its warm-pool assignment."""
        self._request("DELETE", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}")
        logger.debug(f"Cloudflare sandbox destroyed: {sandbox_id}")

    def is_running(self, sandbox_id: str) -> bool:
        """Check sandbox liveness. Note: this starts the container if it is asleep."""
        response = self._request("GET", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/running")
        return bool(response.json().get("running"))

    def exec(
        self,
        sandbox_id: str,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        session_id: str | None = None,
    ) -> CloudflareExecResult:
        """Execute a command in the sandbox and collect its streamed output.

        Args:
            sandbox_id: Target sandbox id.
            command: Shell command string (run via ``bash -c``) or argv list.
            cwd: Working directory; must resolve inside /workspace.
            env: Environment variables for this command. The bridge exec route has
                no env field, so they are prepended via ``env KEY=VALUE ...``.
            timeout: Command timeout in seconds (mapped to ``timeout_ms``).
            session_id: Optional execution session id (``Session-Id`` header) so
                shell state (cwd, exported vars) persists across commands.

        Returns:
            CloudflareExecResult with exit_code, stdout, and stderr.
        """
        argv = ["bash", "-c", command] if isinstance(command, str) else list(command)
        if env:
            argv = ["env", *[f"{key}={value}" for key, value in env.items()], *argv]

        body: dict[str, Any] = {"argv": argv}
        if timeout:
            body["timeout_ms"] = int(timeout * 1000)
        if cwd:
            body["cwd"] = resolve_workspace_path(cwd)

        headers = {"Session-Id": session_id} if session_id else None
        read_timeout = max((timeout or 0) + 120, self.read_timeout)
        response = self._request(
            "POST",
            f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/exec",
            json=body,
            headers=headers,
            stream=True,
            read_timeout=read_timeout,
        )
        return self._parse_exec_sse(response)

    @staticmethod
    def _parse_exec_sse(response: requests.Response) -> CloudflareExecResult:
        """Parse the exec SSE stream into a CloudflareExecResult."""
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        exit_code: int | None = None
        event: str | None = None
        data_lines: list[str] = []

        def handle_event() -> None:
            nonlocal exit_code
            data = "\n".join(data_lines)
            if event in ("stdout", "stderr"):
                try:
                    decoded = base64.b64decode(data)
                except (ValueError, TypeError):
                    decoded = data.encode("utf-8", errors="replace")
                (stdout_chunks if event == "stdout" else stderr_chunks).append(decoded)
            elif event == "exit":
                try:
                    exit_code = json.loads(data).get("exit_code")
                except (ValueError, AttributeError):
                    exit_code = None
            elif event == "error":
                try:
                    payload = json.loads(data)
                except ValueError:
                    payload = {"error": data}
                raise CloudflareSandboxError(
                    payload.get("error") or "Cloudflare sandbox exec failed", code=payload.get("code")
                )

        # The bridge emits UTF-8; requests would otherwise fall back to ISO-8859-1
        # for text/event-stream responses without an explicit charset.
        response.encoding = "utf-8"
        try:
            with response:
                for line in response.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    if line == "":
                        if event is not None:
                            handle_event()
                        event, data_lines = None, []
                        continue
                    if line.startswith("event:"):
                        event = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].removeprefix(" "))
                if event is not None:
                    handle_event()
        except requests.RequestException as e:
            raise CloudflareSandboxError(f"Cloudflare sandbox exec stream failed: {e}") from e

        # Decode once so multibyte characters split across SSE chunks survive intact.
        return CloudflareExecResult(
            exit_code=exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )

    def _file_url_path(self, sandbox_id: str, path: str) -> str:
        # The bridge routes decode the URL with decodeURI, which leaves URI-reserved
        # characters (: @ & = + $ , ;) and % encoded/untouched asymmetrically - keep
        # them raw so the container sees the literal filename, exactly like a TS
        # client using fetch(). Filenames containing %, ? or # are unrepresentable
        # through the bridge file routes (true for all clients).
        resolved = resolve_workspace_path(path)
        return f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/file{quote(resolved, safe='/:@&=+$,;')}"

    def read_file(self, sandbox_id: str, path: str) -> bytes:
        """Read a file from the sandbox as raw bytes (path must be inside /workspace)."""
        response = self._request("GET", self._file_url_path(sandbox_id, path))
        return response.content

    def write_file(self, sandbox_id: str, path: str, content: bytes) -> None:
        """Write raw bytes to a file in the sandbox (max 32 MiB; parents auto-created)."""
        self._request("PUT", self._file_url_path(sandbox_id, path), data=content)

    def create_session(
        self,
        sandbox_id: str,
        session_id: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Create an execution session with isolated shell state; returns its id."""
        body: dict[str, Any] = {}
        if session_id:
            body["id"] = session_id
        if cwd:
            body["cwd"] = cwd
        if env:
            body["env"] = env
        response = self._request("POST", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/session", json=body)
        return response.json()["id"]

    def delete_session(self, sandbox_id: str, session_id: str) -> None:
        """Delete an execution session."""
        self._request("DELETE", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/session/{quote(session_id)}")

    def create_tunnel(self, sandbox_id: str, port: int, name: str | None = None) -> dict[str, Any]:
        """Expose a sandbox port via a Cloudflare tunnel; returns tunnel info incl. ``url``."""
        kwargs: dict[str, Any] = {"json": {"name": name}} if name else {}
        response = self._request(
            "POST", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/tunnel/{int(port)}", **kwargs
        )
        return response.json()

    def delete_tunnel(self, sandbox_id: str, port: int) -> None:
        """Tear down the tunnel for a sandbox port."""
        self._request("DELETE", f"/sandbox/{self._validate_sandbox_id(sandbox_id)}/tunnel/{int(port)}")


class CloudflareSandboxInstance:
    """Lightweight handle binding a client to one sandbox id (and optional session).

    Used as the live sandbox object by CloudflareInterpreterTool, mirroring the
    role SDK sandbox instances play for the E2B and Daytona tools.
    """

    def __init__(self, client: CloudflareSandboxClient, sandbox_id: str, session_id: str | None = None):
        self.client = client
        self.sandbox_id = sandbox_id
        self.session_id = session_id

    def exec(
        self,
        command: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CloudflareExecResult:
        return self.client.exec(self.sandbox_id, command, cwd=cwd, env=env, timeout=timeout, session_id=self.session_id)

    def read_file(self, path: str) -> bytes:
        return self.client.read_file(self.sandbox_id, path)

    def write_file(self, path: str, content: bytes) -> None:
        self.client.write_file(self.sandbox_id, path, content)

    def destroy(self) -> None:
        self.client.destroy_sandbox(self.sandbox_id)
