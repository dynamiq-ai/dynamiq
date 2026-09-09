"""File storage backed by the Dynamiq platform API."""

import base64
import mimetypes
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import ConfigDict, Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.utils.logger import logger

from .base import FileExistsError, FileInfo, FileNotFoundError, FileStore, PermissionError, StorageError


class DynamiqFileStore(FileStore):
    """File storage delegating every operation to the Dynamiq platform API.

    Files live under ``{connection.url}/v1/memory-stores/{memory_store_id}/files``. Access control is
    enforced server-side from the connection credentials and the ``user`` identity, so this store
    never reads or writes a backend directly.

    Its intended use is as a route of :class:`CompositeFileStore` (persisting the ``memories/``
    prefix while scratch paths stay ephemeral), but it is a complete ``FileStore`` and works
    standalone.

    Note that ``FileStore`` is a plain pydantic model rather than a ``ConnectionNode``, so there is
    no injected client and no async path: requests go through ``connection.connect()`` synchronously,
    the same tradeoff the Dynamiq conversation-memory backend already makes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    memory_store_id: str = Field(min_length=1, description="Identifier of the remote store.")
    user: str | None = Field(
        default=None,
        description="User identity for ACL-enforced access. Sent with every request; never agent-supplied.",
    )
    timeout: float = Field(default=30, description="Timeout in seconds for API requests.")

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude connection details from serialization."""
        return {"connection": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the store, keeping the API key out of the payload.

        The base ``FileStore.to_dict`` discards ``include_secure_params`` instead of acting on it,
        which would dump the connection - API key included - into any serialized agent. Credentials
        are emitted only when a caller explicitly asks for secure params; otherwise the connection
        is reduced to its id and type, and rebuilds from ``DYNAMIQ_API_KEY`` / ``DYNAMIQ_URL`` on
        load, as connections do everywhere else.
        """
        kwargs.pop("for_tracing", False)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        data["connection"] = self.connection.to_dict(for_tracing=not include_secure_params)
        return data

    def supports_extracted_text_cache(self, file_path: str | Path = "") -> bool:
        """Never cache extracted text remotely.

        It would cost an extra write request on every read and leave ``.extracted.txt`` files in a
        durable namespace the agent lists and the user curates.
        """
        return False

    def _base_path(self) -> str:
        return f"/v1/memory-stores/{self.memory_store_id}/files"

    def _with_user(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach the user identity to a request payload."""
        if self.user is not None:
            params["user"] = self.user
        return params

    def _request(
        self,
        method: HTTPMethod,
        path: str = "",
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        operation: str = "",
        file_path: str = "",
    ) -> Any:
        """Execute a request against the store API and unwrap its ``data`` envelope."""
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise StorageError("Dynamiq API base URL is not configured.", operation=operation, path=file_path)

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
            logger.error(f"DynamiqFileStore: request to {url} failed. Error: {exc}")
            raise StorageError(f"Failed to call Dynamiq API: {exc}", operation=operation, path=file_path) from exc

        self._raise_for_status(response, operation=operation, file_path=file_path)

        if response.status_code == 204 or not response.content:
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            raise StorageError(
                f"Received a non-JSON response from the Dynamiq API: {response.text}",
                operation=operation,
                path=file_path,
            ) from exc

        return payload.get("data") if isinstance(payload, dict) else payload

    @staticmethod
    def _raise_for_status(response: Any, operation: str, file_path: str) -> None:
        """Map an API status code onto the storage exception hierarchy."""
        status = response.status_code
        if status < 400:
            return

        if status == 404:
            raise FileNotFoundError(f"File '{file_path}' not found", operation=operation, path=file_path)
        if status == 403:
            raise PermissionError(
                f"Permission denied for '{file_path}'",
                operation=operation,
                path=file_path,
            )
        if status == 409:
            raise FileExistsError(f"File '{file_path}' already exists", operation=operation, path=file_path)

        raise StorageError(
            f"Request to Dynamiq API failed: {status} {response.text}",
            operation=operation,
            path=file_path,
        )

    def store(
        self,
        file_path: str | Path,
        content: str | bytes | BinaryIO,
        content_type: str = None,
        metadata: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> FileInfo:
        """Upload a file to the remote store."""
        file_path = str(file_path)

        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        elif hasattr(content, "read"):
            content_bytes = content.read()
            if hasattr(content, "seek"):
                content.seek(0)
        else:
            raise StorageError(f"Unsupported content type: {type(content)}", operation="store", path=file_path)

        if content_type is None:
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"

        payload = self._with_user(
            {
                "path": file_path,
                "content": base64.b64encode(content_bytes).decode("ascii"),
                "content_type": content_type,
                "metadata": metadata or {},
                "overwrite": overwrite,
            }
        )

        data = self._request(HTTPMethod.PUT, json=payload, operation="store", file_path=file_path)
        return self._to_file_info(data or {}, fallback_path=file_path, content=content_bytes)

    def retrieve(self, file_path: str | Path) -> bytes:
        """Download file content from the remote store."""
        file_path = str(file_path)
        data = self._request(
            HTTPMethod.GET,
            "/content",
            params=self._with_user({"path": file_path}),
            operation="retrieve",
            file_path=file_path,
        )
        if not data:
            raise FileNotFoundError(f"File '{file_path}' not found", operation="retrieve", path=file_path)
        return self._decode_content(data.get("content"))

    def exists(self, file_path: str | Path) -> bool:
        """Check whether a file exists in the remote store."""
        file_path = str(file_path)
        try:
            data = self._request(
                HTTPMethod.GET,
                "/exists",
                params=self._with_user({"path": file_path}),
                operation="exists",
                file_path=file_path,
            )
        except FileNotFoundError:
            return False
        return bool((data or {}).get("exists", False))

    def delete(self, file_path: str | Path) -> bool:
        """Delete a file from the remote store."""
        file_path = str(file_path)
        try:
            data = self._request(
                HTTPMethod.DELETE,
                params=self._with_user({"path": file_path}),
                operation="delete",
                file_path=file_path,
            )
        except FileNotFoundError:
            return False
        if data is None:
            return True
        return bool(data.get("deleted", True))

    def list_files(
        self,
        directory: str | Path = "",
        recursive: bool = False,
        pattern: str = None,
    ) -> list[FileInfo]:
        """List files in the remote store.

        Unlike ``InMemoryFileStore``, this honours ``pattern`` - the argument the base class declares
        and the in-memory store drops.
        """
        params: dict[str, Any] = {"path": str(directory), "recursive": recursive}
        if pattern:
            params["pattern"] = pattern

        data = self._request(
            HTTPMethod.GET,
            params=self._with_user(params),
            operation="list_files",
            file_path=str(directory),
        )
        return [self._to_file_info(entry) for entry in (data or [])]

    def list_files_bytes(self, file_paths: list[str] | None = None) -> list[BytesIO]:
        """Return the named files as ``BytesIO`` objects.

        With no explicit paths this returns an empty list rather than downloading the whole store.
        ``AgentBase._inject_files_into_tool`` calls this with no arguments on *every* tool
        invocation to build its attachment map; fanning out here would mean a full download per tool
        call. Persistent files are memory, not attachments - the agent reads them by path.
        """
        if not file_paths:
            return []

        files: list[BytesIO] = []
        for file_path in file_paths:
            try:
                content = self.retrieve(file_path)
            except (FileNotFoundError, StorageError) as exc:
                logger.warning(f"DynamiqFileStore: skipping '{file_path}' in list_files_bytes. Error: {exc}")
                continue

            file = BytesIO(content)
            file.name = file_path
            file.description = ""
            file.content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            files.append(file)
        return files

    @staticmethod
    def _decode_content(content: Any) -> bytes:
        """Decode the base64 ``content`` field of an API response."""
        if content is None:
            return b""
        if isinstance(content, bytes):
            return content
        try:
            return base64.b64decode(content)
        except Exception as exc:
            raise StorageError(f"Failed to decode file content: {exc}", operation="retrieve") from exc

    def _to_file_info(
        self,
        data: dict[str, Any],
        fallback_path: str | None = None,
        content: bytes | None = None,
    ) -> FileInfo:
        """Build a ``FileInfo`` from an API payload."""
        path = data.get("path") or fallback_path or ""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()

        if content is None and data.get("content") is not None:
            content = self._decode_content(data["content"])

        size = data.get("size")
        if size is None:
            size = len(content) if content is not None else 0

        return FileInfo(
            name=data.get("name") or os.path.basename(path),
            path=path,
            size=size,
            content_type=data.get("content_type") or "application/octet-stream",
            created_at=created_at,
            metadata=data.get("metadata") or {},
            content=content,
        )
