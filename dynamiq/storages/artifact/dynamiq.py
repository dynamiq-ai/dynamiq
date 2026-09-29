"""Artifact store backed by the Dynamiq platform API."""

import json as jsonlib
from datetime import datetime
from typing import Any, Sequence

from pydantic import ConfigDict, Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
from dynamiq.utils.logger import logger

from .base import (
    TEXT_KINDS,
    Artifact,
    ArtifactConflictError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactPermissionError,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactVersion,
    default_media_type,
    infer_kind,
)


class DynamiqArtifactStore(ArtifactStore):
    """Artifacts held by the Dynamiq platform API at ``{connection.url}/v1/artifacts``.

    Owner and org come from the connection's credential (a user, conversation or service token),
    so nothing tenant-shaped is sent from here. Edits are applied server-side to the latest version.
    Requests go through ``connection.connect()`` synchronously, as in ``DynamiqMemoryStore``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    project_id: str | None = Field(
        default=None,
        description="Project that owns created artifacts. Unset means the credential's user owns them.",
    )
    timeout: float = Field(default=60, description="Timeout in seconds for API requests.")

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude connection details from the plain dump; ``to_dict`` re-adds them safely."""
        return {"connection": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the store, emitting credentials only when secure params are requested."""
        kwargs.pop("for_tracing", None)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        data["connection"] = self.connection.to_dict(for_tracing=not include_secure_params)
        return data

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
            raise ArtifactStoreError(
                "Dynamiq API base URL is not configured.", operation=operation, artifact_id=artifact_id
            )

        url = f"{base_url}/v1/artifacts{path}"
        request_headers = {} if files else {"Content-Type": "application/json"}
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
            logger.error(f"DynamiqArtifactStore: request to {url} failed. Error: {exc}")
            raise ArtifactStoreError(
                f"Failed to call Dynamiq API: {exc}", operation=operation, artifact_id=artifact_id
            ) from exc

        self._raise_for_status(response, operation=operation, artifact_id=artifact_id)
        return response

    def _request(self, method: HTTPMethod, path: str = "", **kwargs) -> Any:
        """Execute a request and unwrap its JSON ``data`` envelope."""
        response = self._send(method, path, **kwargs)
        if response.status_code == 204 or not response.content:
            return None
        try:
            payload = response.json()
        except ValueError as exc:
            raise ArtifactStoreError(
                f"Received a non-JSON response from the Dynamiq API: {response.text}",
                operation=kwargs.get("operation"),
                artifact_id=kwargs.get("artifact_id"),
            ) from exc
        return payload.get("data", payload) if isinstance(payload, dict) else payload

    @staticmethod
    def _raise_for_status(response: Any, operation: str, artifact_id: str | None) -> None:
        """Map an API status code onto the artifact-store exception hierarchy."""
        status = response.status_code
        if status < 400:
            return
        label = f"Artifact '{artifact_id}'" if artifact_id else "Artifact"
        if status == 404:
            raise ArtifactNotFoundError(f"{label} not found", operation=operation, artifact_id=artifact_id)
        if status == 403:
            raise ArtifactPermissionError(
                f"Permission denied for {label.lower()}", operation=operation, artifact_id=artifact_id
            )
        if status in (409, 412):
            raise ArtifactConflictError(
                f"{label} changed since it was last read: {response.text}",
                operation=operation,
                artifact_id=artifact_id,
            )
        raise ArtifactStoreError(
            f"Request to Dynamiq API failed: {status} {response.text}", operation=operation, artifact_id=artifact_id
        )

    @staticmethod
    def _body(content: str | bytes | None, fields: dict[str, Any], name: str, media_type: str) -> dict[str, Any]:
        """Keyword arguments for ``_send``: JSON for text, multipart for bytes."""
        fields = {k: v for k, v in fields.items() if v is not None}
        if isinstance(content, bytes):
            data = {k: jsonlib.dumps(v) if isinstance(v, (dict, list)) else v for k, v in fields.items()}
            return {"data": data, "files": {"file": (name, content, media_type)}}
        if content is not None:
            fields["content"] = content
        return {"json": fields}

    def create(
        self,
        *,
        name: str,
        title: str,
        kind: ArtifactKind,
        content: str | bytes,
        media_type: str | None = None,
        summary: str | None = None,
        source: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Create an artifact with its first version."""
        kind = ArtifactKind(kind)
        media_type = media_type or default_media_type(kind, name)
        fields = {
            "name": name,
            "title": title,
            "kind": kind.value,
            "media_type": media_type,
            "summary": summary,
            "project_id": self.project_id,
            "source": source,
            "metadata": metadata,
        }
        data = self._request(HTTPMethod.POST, operation="create", **self._body(content, fields, name, media_type))
        return self._to_artifact(data or {}, fallback={"name": name, "title": title, "kind": kind})

    def update(
        self,
        artifact_id: str,
        *,
        content: str | bytes | None = None,
        edits: Sequence[Any] | None = None,
        title: str | None = None,
        summary: str | None = None,
        if_match: str | None = None,
        source: dict[str, Any] | None = None,
    ) -> Artifact:
        """Add a version; edits are sent as-is and applied server-side."""
        if content is None and not edits and title is None:
            raise ArtifactStoreError(
                "Nothing to update: pass content, edits or title.", operation="update", artifact_id=artifact_id
            )
        fields = {
            "title": title,
            "summary": summary,
            "source": source,
            "edits": [self._edit_to_dict(e) for e in edits] if edits else None,
        }
        body = self._body(content, fields, name="content", media_type="application/octet-stream")
        data = self._request(
            HTTPMethod.PUT,
            f"/{artifact_id}",
            operation="update",
            artifact_id=artifact_id,
            headers={"If-Match": str(if_match)} if if_match else None,
            **body,
        )
        return self._to_artifact(data or {}, fallback={"id": artifact_id})

    def get(
        self, artifact_id: str, version: int | None = None, include_content: bool = True
    ) -> tuple[Artifact, str | bytes | None]:
        """Fetch metadata, then the version's bytes when asked; text kinds come back decoded."""
        data = self._request(
            HTTPMethod.GET,
            f"/{artifact_id}",
            operation="get",
            artifact_id=artifact_id,
            params={"version": version} if version is not None else None,
        )
        if not data:
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found", operation="get", artifact_id=artifact_id)
        artifact = self._to_artifact(data, fallback={"id": artifact_id})
        if not include_content:
            return artifact, None

        resolved = version if version is not None else artifact.version
        response = self._send(
            HTTPMethod.GET, f"/{artifact_id}/versions/{resolved}/content", operation="get", artifact_id=artifact_id
        )
        content: str | bytes = response.content or b""
        if artifact.kind in TEXT_KINDS:
            content = content.decode("utf-8", errors="replace")
        return artifact, content

    def list(self, *, kind: ArtifactKind | None = None, query: str | None = None, limit: int = 50) -> list[Artifact]:
        """List artifacts visible to the credential, newest first."""
        params = {"page_size": limit, "sort": "-updated_at"}
        if kind:
            params["kind"] = ArtifactKind(kind).value
        if query:
            params["query"] = query
        if self.project_id:
            params["project_id"] = self.project_id
        data = self._request(HTTPMethod.GET, operation="list", params=params)
        return [self._to_artifact(item) for item in (data or [])]

    @staticmethod
    def _edit_to_dict(edit: Any) -> dict[str, Any]:
        if hasattr(edit, "model_dump"):
            return edit.model_dump()
        return dict(edit)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @classmethod
    def _to_artifact(cls, data: dict[str, Any], fallback: dict[str, Any] | None = None) -> Artifact:
        """Build an ``Artifact`` from an API payload, defaulting anything the server omits."""
        fallback = fallback or {}
        name = data.get("name") or fallback.get("name") or ""
        media_type = data.get("media_type")
        raw_kind = data.get("kind") or fallback.get("kind")
        try:
            kind = ArtifactKind(raw_kind) if raw_kind else infer_kind(name, media_type)
        except ValueError:
            kind = ArtifactKind.FILE
        media_type = media_type or default_media_type(kind, name)
        artifact_id = data.get("id") or fallback.get("id") or ""
        title = data.get("title") or fallback.get("title") or name

        latest = None
        raw_latest = data.get("latest") or data.get("latest_version")
        if isinstance(raw_latest, dict):
            latest = ArtifactVersion(
                id=raw_latest.get("id") or "",
                artifact_id=raw_latest.get("artifact_id") or artifact_id,
                version=int(raw_latest.get("version") or 1),
                title=raw_latest.get("title") or title,
                kind=raw_latest.get("kind") or kind,
                media_type=raw_latest.get("media_type") or media_type,
                size=int(raw_latest.get("size") or 0),
                checksum=raw_latest.get("checksum"),
                url=raw_latest.get("url"),
                summary=raw_latest.get("summary"),
                created_at=cls._parse_datetime(raw_latest.get("created_at")),
            )

        return Artifact(
            id=artifact_id,
            name=name,
            title=title,
            kind=kind,
            media_type=media_type,
            latest=latest,
            url=data.get("url"),
            metadata=data.get("metadata") or {},
        )
