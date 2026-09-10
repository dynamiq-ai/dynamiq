"""Memory store backed by the Dynamiq platform API."""

from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.utils.logger import logger

from .base import MemoryEntry, MemoryNotFoundError, MemoryPermissionError, MemoryStore, MemoryStoreError


class DynamiqMemoryStore(MemoryStore):
    """Memories held by the Dynamiq platform API.

    Every operation is an API call to ``{connection.url}/v1/memory-stores/{memory_store_id}/files``;
    this class never touches storage directly. Access control is enforced server-side from the
    connection credentials together with ``user_id``, which is sent on every request and is never
    supplied by the agent.

    ``MemoryStore`` is a plain pydantic model rather than a ``ConnectionNode``, so there is no
    injected client and no async path: requests go through ``connection.connect()`` synchronously.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    memory_store_id: str = Field(min_length=1, description="Identifier of the remote memory store.")
    user_id: str = Field(
        min_length=1,
        description="End user this memory belongs to. Sent with every request; never agent-supplied.",
    )
    timeout: float = Field(default=30, description="Timeout in seconds for API requests.")

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude connection details from the plain dump; ``to_dict`` re-adds them safely."""
        return {"connection": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the store, keeping the API key out of the payload.

        Credentials are emitted only when a caller explicitly asks for secure params; otherwise the
        connection is reduced to its id and type and rebuilds from ``DYNAMIQ_API_KEY`` /
        ``DYNAMIQ_URL`` on load, as connections do everywhere else.
        """
        kwargs.pop("for_tracing", None)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        data["connection"] = self.connection.to_dict(for_tracing=not include_secure_params)
        return data

    def identity(self):
        """Which remote store this addresses: the same triple means the same data."""
        return (self.type, self.connection.url, self.memory_store_id, self.user_id)

    def _base_path(self) -> str:
        return f"/v1/memory-stores/{self.memory_store_id}/files"

    def _request(
        self,
        method: HTTPMethod,
        path: str = "",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        operation: str = "",
        memory_path: str = "",
    ) -> Any:
        """Execute a request against the memory API and unwrap its ``data`` envelope."""
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise MemoryStoreError("Dynamiq API base URL is not configured.", operation=operation, path=memory_path)

        url = f"{base_url}{self._base_path()}{path}"
        headers = {"Content-Type": "application/json"}
        conn_headers = conn_params.get("headers")
        if isinstance(conn_headers, dict):
            headers.update(conn_headers)

        client = self.connection.connect()
        verb = method.value if isinstance(method, HTTPMethod) else method
        try:
            response = client.request(verb, url, headers=headers, params=params, json=json, timeout=self.timeout)
        except Exception as exc:
            logger.error(f"DynamiqMemoryStore: request to {url} failed. Error: {exc}")
            raise MemoryStoreError(f"Failed to call Dynamiq API: {exc}", operation=operation, path=memory_path) from exc

        self._raise_for_status(response, operation=operation, memory_path=memory_path)

        if response.status_code == 204 or not response.content:
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            raise MemoryStoreError(
                f"Received a non-JSON response from the Dynamiq API: {response.text}",
                operation=operation,
                path=memory_path,
            ) from exc

        return payload.get("data") if isinstance(payload, dict) else payload

    @staticmethod
    def _raise_for_status(response: Any, operation: str, memory_path: str) -> None:
        """Map an API status code onto the memory-store exception hierarchy."""
        status = response.status_code
        if status < 400:
            return

        if status == 404:
            raise MemoryNotFoundError(f"Memory '{memory_path}' not found", operation=operation, path=memory_path)
        if status == 403:
            raise MemoryPermissionError(f"Permission denied for '{memory_path}'", operation=operation, path=memory_path)

        raise MemoryStoreError(
            f"Request to Dynamiq API failed: {status} {response.text}",
            operation=operation,
            path=memory_path,
        )

    def list(self, prefix: str = "") -> list[MemoryEntry]:
        """List memories under ``prefix``."""
        data = self._request(
            HTTPMethod.GET,
            # `path` on the wire, `prefix` here: the API keeps the file vocabulary, while the
            # interface names what it actually is - a key prefix, not a directory.
            params={"path": prefix, "user_id": self.user_id},
            operation="list",
            memory_path=prefix,
        )
        return [self._to_entry(entry) for entry in (data or [])]

    def read(self, path: str) -> str:
        """Read one memory's content."""
        data = self._request(
            HTTPMethod.GET,
            "/content",
            params={"path": path, "user_id": self.user_id},
            operation="read",
            memory_path=path,
        )
        if not data:
            raise MemoryNotFoundError(f"Memory '{path}' not found", operation="read", path=path)
        return data.get("content") or ""

    def write(self, path: str, content: str) -> MemoryEntry:
        """Create or replace one memory."""
        data = self._request(
            HTTPMethod.PUT,
            json={"path": path, "content": content, "user_id": self.user_id},
            operation="write",
            memory_path=path,
        )
        return self._to_entry(data or {}, fallback_path=path, fallback_size=len(content))

    def delete(self, path: str) -> bool:
        """Delete one memory. A 404 means it was not there, which is not an error."""
        try:
            data = self._request(
                HTTPMethod.DELETE,
                params={"path": path, "user_id": self.user_id},
                operation="delete",
                memory_path=path,
            )
        except MemoryNotFoundError:
            return False
        if data is None:
            return True
        return bool(data.get("deleted", True))

    @staticmethod
    def _to_entry(data: dict[str, Any], fallback_path: str = "", fallback_size: int = 0) -> MemoryEntry:
        """Build a ``MemoryEntry`` from an API payload, defaulting anything the server omits."""
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = None
        elif not isinstance(updated_at, datetime):
            updated_at = None

        size = data.get("size")
        if size is None:
            content = data.get("content")
            size = len(content) if content is not None else fallback_size

        return MemoryEntry(path=data.get("path") or fallback_path, size=size, updated_at=updated_at)
