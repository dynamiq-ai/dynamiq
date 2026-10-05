"""Artifact backend served by the Dynamiq platform API."""

import importlib.metadata
import json as jsonlib
from datetime import datetime, timezone
from typing import Any

from pydantic import ConfigDict, Field, model_validator

from dynamiq.artifacts.backends.base import ArtifactBackend
from dynamiq.artifacts.types import (
    TEXT_KINDS,
    Artifact,
    ArtifactConflictError,
    ArtifactError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactPermissionError,
    ArtifactShare,
    ArtifactVersion,
    ArtifactVisibility,
    default_mime_type,
)
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.utils.logger import logger

try:
    _VERSION = importlib.metadata.version("dynamiq")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"

# The platform records the client that wrote each version from the User-Agent.
USER_AGENT = f"dynamiq-python/{_VERSION}"

# The platform keeps an artifact's latest 50 versions, so one page holds every version it has.
_MAX_VERSIONS = 50


class Dynamiq(ArtifactBackend):
    """Artifacts held by the Dynamiq platform at ``{connection.url}/v1/artifacts``.

    Without ``artifact_store_id`` the artifacts belong to the user behind the connection's token, in
    the token's org: a conversation token in a chat. With it they belong to that artifact store,
    which the platform's own connection reaches in app runs, and ``user_id`` keeps one end user's
    artifacts apart from another's; a call's own ``user_id``, the run's, overrides it. The platform
    does not check the store or the end user when an artifact is read by id, so this backend does:
    another store's or end user's artifact reads as not found. Requests go through
    ``connection.connect()`` synchronously, as in ``DynamiqMemoryStore``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    artifact_store_id: str | None = Field(
        default=None, description="Artifact store the artifacts belong to. Unset: the token's user owns them."
    )
    user_id: str | None = Field(
        default=None,
        description="Default end user of the app the artifacts belong to, within the store. Not a Dynamiq user. "
        "A run's user_id overrides it.",
    )
    timeout: float = Field(default=60, description="Timeout in seconds for API requests.")

    @model_validator(mode="after")
    def validate_owner(self):
        if self.user_id is not None and self.artifact_store_id is None:
            raise ValueError("'user_id' requires 'artifact_store_id'")
        return self

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude connection details from the plain dump; ``to_dict`` re-adds them safely."""
        return {"connection": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the backend, emitting credentials only when secure params are requested."""
        kwargs.pop("for_tracing", None)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        data["connection"] = self.connection.to_dict(for_tracing=not include_secure_params)
        return data

    def create(
        self,
        *,
        file_name: str,
        name: str,
        kind: ArtifactKind,
        content: str | bytes,
        mime_type: str | None = None,
        description: str | None = None,
        entry_path: str | None = None,
        user_id: str | None = None,
    ) -> Artifact:
        """Create an artifact: text kinds as JSON, everything else as an upload."""
        kind = ArtifactKind(kind)
        mime_type = mime_type or default_mime_type(kind, file_name)
        fields = {
            **self._owner(user_id),
            "file_name": file_name,
            "name": name,
            "description": description,
            "kind": kind.value,
            "mime_type": mime_type,
        }
        if isinstance(content, str) and kind in TEXT_KINDS:
            data = self._request(HTTPMethod.POST, operation="create", json=_compact({**fields, "content": content}))
        else:
            data = self._request(
                HTTPMethod.POST,
                "/upload",
                operation="create",
                **_multipart({**fields, "entry_path": entry_path}, file_name, content, mime_type),
            )
        return self._to_artifact(data or {})

    def update(
        self,
        artifact_id: str,
        *,
        content: str | bytes,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        entry_path: str | None = None,
        if_match: str | None = None,
        user_id: str | None = None,
    ) -> Artifact:
        """Add a version: text kinds as JSON, everything else as an upload."""
        artifact = self._fetch(artifact_id, operation="update", user_id=user_id)
        fields = {"name": name, "description": description, "mime_type": mime_type}
        headers = {"If-Match": f'"{if_match}"'} if if_match else None
        if isinstance(content, str) and artifact.kind in TEXT_KINDS:
            data = self._request(
                HTTPMethod.POST,
                f"/{artifact_id}/versions",
                operation="update",
                artifact_id=artifact_id,
                headers=headers,
                json=_compact({**fields, "content": content}),
            )
        else:
            data = self._request(
                HTTPMethod.POST,
                f"/{artifact_id}/versions/upload",
                operation="update",
                artifact_id=artifact_id,
                headers=headers,
                **_multipart(
                    {**fields, "entry_path": entry_path},
                    artifact.file_name,
                    content,
                    mime_type or artifact.mime_type,
                ),
            )
        # The platform renames the artifact after each version and keeps the version's MIME type.
        version = self._to_version(data or {}, artifact_id)
        return artifact.model_copy(
            update={"latest_version": version, "name": version.name or artifact.name, "mime_type": version.mime_type}
        )

    def get(
        self, artifact_id: str, version: int | None = None, include_content: bool = True, user_id: str | None = None
    ) -> tuple[Artifact, str | bytes | None]:
        """Fetch the artifact, then the version's bytes when asked; text kinds come back decoded."""
        artifact = self._fetch(artifact_id, operation="get", user_id=user_id)
        if not include_content:
            return artifact, None

        if version is None or version == artifact.version:
            if artifact.latest_version is None:
                raise ArtifactNotFoundError(
                    f"Artifact '{artifact_id}' has no version", operation="get", artifact_id=artifact_id
                )
            version_id = artifact.latest_version.id
        else:
            version_id = self._version_id(artifact_id, version, operation="get")

        response = self._send(
            HTTPMethod.GET, f"/{artifact_id}/versions/{version_id}/download", operation="get", artifact_id=artifact_id
        )
        content: str | bytes = response.content or b""
        if artifact.kind in TEXT_KINDS:
            content = content.decode("utf-8", errors="replace")
        return artifact, content

    def list(self, *, kind: ArtifactKind | None = None, limit: int = 50, user_id: str | None = None) -> list[Artifact]:
        """List the store's artifacts, or the token user's own, most recently updated first."""
        # The platform pages between 10 and 500 items.
        params: dict[str, Any] = {**self._owner(user_id), "page_size": min(max(limit, 10), 500), "sort": "-updated_at"}
        if kind:
            params["kind"] = ArtifactKind(kind).value
        data = self._request(HTTPMethod.GET, operation="list", params=params)
        return [self._to_artifact(item) for item in (data or [])][:limit]

    def share(
        self,
        artifact_id: str,
        *,
        pinned_version: int | None = None,
        expires_at: datetime | None = None,
        user_id: str | None = None,
    ) -> ArtifactShare:
        """Create or update the link share. A naive ``expires_at`` is taken as UTC."""
        self._fetch(artifact_id, operation="share", user_id=user_id)
        body: dict[str, Any] = {}
        if pinned_version is not None:
            body["pinned_version_id"] = self._version_id(artifact_id, pinned_version, operation="share")
        if expires_at is not None:
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            body["expires_at"] = expires_at.isoformat()
        data = self._request(
            HTTPMethod.POST, f"/{artifact_id}/share", operation="share", artifact_id=artifact_id, json=body
        )
        data = data or {}
        return ArtifactShare(
            id=data.get("id") or "",
            artifact_id=data.get("artifact_id") or artifact_id,
            url=data.get("url"),
            pinned_version_id=data.get("pinned_version_id"),
            expires_at=_parse_datetime(data.get("expires_at")),
        )

    def unshare(self, artifact_id: str, *, user_id: str | None = None) -> None:
        """Revoke the link share; the artifact becomes private."""
        self._fetch(artifact_id, operation="unshare", user_id=user_id)
        self._request(HTTPMethod.DELETE, f"/{artifact_id}/share", operation="unshare", artifact_id=artifact_id)

    def _scope(self, user_id: str | None) -> str | None:
        """The end user a call is for: the run's, else the configured default. End users exist only
        in a store; without one the artifacts belong to the token's user."""
        return (user_id or self.user_id) if self.artifact_store_id is not None else None

    def _owner(self, user_id: str | None = None) -> dict[str, str]:
        """The store and end user fields of a create or a list, when set."""
        return _compact({"store_id": self.artifact_store_id, "user_id": self._scope(user_id)})

    def _fetch(self, artifact_id: str, operation: str, user_id: str | None = None) -> Artifact:
        """Fetch an artifact, refusing one outside the backend's store or end user."""
        data = self._request(HTTPMethod.GET, f"/{artifact_id}", operation=operation, artifact_id=artifact_id)
        artifact = self._to_artifact(data or {})
        scope = self._scope(user_id)
        outside_store = self.artifact_store_id is not None and artifact.store_id != self.artifact_store_id
        other_user = scope is not None and artifact.user_id != scope
        if not data or outside_store or other_user:
            raise ArtifactNotFoundError(
                f"Artifact '{artifact_id}' not found", operation=operation, artifact_id=artifact_id
            )
        return artifact

    def _version_id(self, artifact_id: str, version: int, operation: str) -> str:
        """The id of an artifact's version by its number."""
        data = self._request(
            HTTPMethod.GET,
            f"/{artifact_id}/versions",
            operation=operation,
            artifact_id=artifact_id,
            params={"page_size": _MAX_VERSIONS},
        )
        for item in data or []:
            if item.get("version") == version:
                return item["id"]
        raise ArtifactNotFoundError(
            f"Version {version} of artifact '{artifact_id}' not found", operation=operation, artifact_id=artifact_id
        )

    def _send(
        self,
        method: HTTPMethod,
        path: str,
        *,
        operation: str,
        artifact_id: str | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Execute a request and return the raw response after mapping error statuses."""
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise ArtifactError("Dynamiq API base URL is not configured.", operation=operation, artifact_id=artifact_id)

        url = f"{base_url}/v1/artifacts{path}"
        request_headers = {"User-Agent": USER_AGENT}
        if not files:
            request_headers["Content-Type"] = "application/json"
        conn_headers = conn_params.get("headers")
        if isinstance(conn_headers, dict):
            request_headers.update(conn_headers)
        if headers:
            request_headers.update(headers)

        client = self.connection.connect()
        verb = method.value if isinstance(method, HTTPMethod) else method
        try:
            response = client.request(
                verb,
                url,
                headers=request_headers,
                params=params,
                json=json,
                data=data,
                files=files,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.error(f"Dynamiq artifacts: request to {url} failed. Error: {exc}")
            raise ArtifactError(
                f"Failed to call Dynamiq API: {exc}", operation=operation, artifact_id=artifact_id
            ) from exc

        _raise_for_status(response, operation=operation, artifact_id=artifact_id)
        return response

    def _request(self, method: HTTPMethod, path: str = "", **kwargs) -> Any:
        """Execute a request and unwrap its JSON ``data`` envelope."""
        response = self._send(method, path, **kwargs)
        if response.status_code == 204 or not response.content:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise ArtifactError(
                f"Received a non-JSON response from the Dynamiq API: {response.text}",
                operation=kwargs.get("operation"),
                artifact_id=kwargs.get("artifact_id"),
            ) from exc
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    @classmethod
    def _to_artifact(cls, data: dict[str, Any]) -> Artifact:
        """Build an ``Artifact`` from an API payload, defaulting anything the server omits."""
        artifact_id = data.get("id") or ""
        file_name = data.get("file_name") or ""
        try:
            kind = ArtifactKind(data.get("kind"))
        except ValueError:
            kind = ArtifactKind.FILE
        try:
            visibility = ArtifactVisibility(data.get("visibility"))
        except ValueError:
            visibility = ArtifactVisibility.PRIVATE
        latest = data.get("latest_version")
        return Artifact(
            id=artifact_id,
            file_name=file_name,
            name=data.get("name") or file_name,
            kind=kind,
            mime_type=data.get("mime_type") or default_mime_type(kind, file_name),
            description=data.get("description"),
            visibility=visibility,
            store_id=data.get("store_id"),
            user_id=data.get("user_id"),
            latest_version=cls._to_version(latest, artifact_id) if isinstance(latest, dict) else None,
            url=data.get("url"),
        )

    @staticmethod
    def _to_version(data: dict[str, Any], artifact_id: str) -> ArtifactVersion:
        """Build an ``ArtifactVersion`` from an API payload."""
        return ArtifactVersion(
            id=data.get("id") or "",
            artifact_id=data.get("artifact_id") or artifact_id,
            version=int(data.get("version") or 1),
            name=data.get("name") or "",
            description=data.get("description"),
            mime_type=data.get("mime_type") or "application/octet-stream",
            size=int(data.get("size") or 0),
            checksum=data.get("checksum"),
            entry_path=data.get("entry_path"),
            created_at=_parse_datetime(data.get("created_at")),
        )


def _compact(fields: dict[str, Any]) -> dict[str, Any]:
    """The fields that are set."""
    return {k: v for k, v in fields.items() if v is not None}


def _multipart(fields: dict[str, Any], file_name: str, content: str | bytes, mime_type: str) -> dict[str, Any]:
    """Keyword arguments for ``_send`` of an upload: the fields as one JSON ``data`` part and the bytes."""
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return {"data": {"data": jsonlib.dumps(_compact(fields))}, "files": {"file": (file_name, raw, mime_type)}}


def _raise_for_status(response: Any, operation: str, artifact_id: str | None) -> None:
    """Map an API status code onto the artifact exception hierarchy."""
    status = response.status_code
    if status < 400:
        return
    label = f"Artifact '{artifact_id}'" if artifact_id else "Artifact"
    detail = _error_detail(response)
    if status == 404:
        raise ArtifactNotFoundError(f"{label} not found", operation=operation, artifact_id=artifact_id)
    if status == 403:
        raise ArtifactPermissionError(
            f"Permission denied for {label.lower()}", operation=operation, artifact_id=artifact_id
        )
    if status in (409, 412):
        raise ArtifactConflictError(
            f"{label} changed since it was last read: {detail}", operation=operation, artifact_id=artifact_id
        )
    raise ArtifactError(f"Dynamiq API returned {status}: {detail}", operation=operation, artifact_id=artifact_id)


def _error_detail(response: Any) -> str:
    """The message and field details of a platform error body, or the raw text."""
    try:
        error = response.json().get("error") or {}
    except (ValueError, AttributeError):
        return response.text
    if not isinstance(error, dict) or not error.get("message"):
        return response.text
    details = error.get("details")
    return f"{error['message']} {jsonlib.dumps(details)}" if details else error["message"]


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None
