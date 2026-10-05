import hashlib
from typing import Any

import pytest
from pydantic import Field

from dynamiq.artifacts import (
    Artifact,
    ArtifactBackend,
    ArtifactConflictError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactShare,
    ArtifactVersion,
    ArtifactVisibility,
    default_mime_type,
)


class FakeArtifactBackend(ArtifactBackend):
    """A local stand-in for the platform: immutable versions, If-Match on the latest version id.

    A call with a ``user_id`` acts for that end user: it reaches only that user's artifacts, as the
    Dynamiq backend does within a store.
    """

    calls: list[tuple[str, dict[str, Any]]] = Field(default_factory=list, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._artifacts: dict[str, Artifact] = {}
        self._versions: dict[str, list[tuple[ArtifactVersion, str | bytes]]] = {}

    def create(
        self, *, file_name, name, kind, content, mime_type=None, description=None, entry_path=None, user_id=None
    ):
        self.calls.append(
            ("create", {"file_name": file_name, "kind": kind, "content": content, "entry_path": entry_path})
        )
        artifact_id = f"a{len(self._artifacts) + 1}"
        kind = ArtifactKind(kind)
        artifact = Artifact(
            id=artifact_id,
            file_name=file_name,
            name=name,
            kind=kind,
            mime_type=mime_type or default_mime_type(kind, file_name),
            url=f"https://app.example/artifacts/{artifact_id}",
            user_id=user_id,
        )
        self._artifacts[artifact_id] = artifact
        self._add_version(artifact, content, name, description, entry_path=entry_path)
        return artifact.model_copy(deep=True)

    def update(
        self,
        artifact_id,
        *,
        content,
        name=None,
        description=None,
        mime_type=None,
        entry_path=None,
        if_match=None,
        user_id=None,
    ):
        self.calls.append(
            (
                "update",
                {
                    "artifact_id": artifact_id,
                    "content": content,
                    "if_match": if_match,
                    "mime_type": mime_type,
                    "entry_path": entry_path,
                },
            )
        )
        artifact = self._find(artifact_id, user_id)
        if if_match and if_match != artifact.latest_version.id:
            raise ArtifactConflictError(f"Artifact '{artifact_id}' changed", "update", artifact_id)
        if mime_type:
            artifact.mime_type = mime_type
        self._add_version(artifact, content, name or artifact.name, description, entry_path=entry_path)
        return artifact.model_copy(deep=True)

    def get(self, artifact_id, version=None, include_content=True, user_id=None):
        artifact = self._find(artifact_id, user_id)
        versions = self._versions[artifact_id]
        resolved = version or artifact.version
        if not 1 <= resolved <= len(versions):
            raise ArtifactNotFoundError(f"Version {resolved} not found", "get", artifact_id)
        return artifact.model_copy(deep=True), versions[resolved - 1][1] if include_content else None

    def list(self, *, kind=None, limit=50, user_id=None):
        found = [
            a
            for a in self._artifacts.values()
            if (kind is None or a.kind == kind) and (user_id is None or a.user_id == user_id)
        ]
        return [a.model_copy(deep=True) for a in reversed(found)][:limit]

    def share(self, artifact_id, *, pinned_version=None, expires_at=None, user_id=None):
        self.calls.append(("share", {"pinned_version": pinned_version, "expires_at": expires_at}))
        artifact = self._find(artifact_id, user_id)
        pinned_version_id = None
        if pinned_version is not None:
            versions = self._versions[artifact_id]
            if not 1 <= pinned_version <= len(versions):
                raise ArtifactNotFoundError(f"Version {pinned_version} not found", "share", artifact_id)
            pinned_version_id = versions[pinned_version - 1][0].id
        artifact.visibility = ArtifactVisibility.LINK
        return ArtifactShare(
            id=f"s-{artifact_id}",
            artifact_id=artifact_id,
            url=f"https://app.example/a/s-{artifact_id}",
            pinned_version_id=pinned_version_id,
            expires_at=expires_at,
        )

    def unshare(self, artifact_id, *, user_id=None):
        self._find(artifact_id, user_id).visibility = ArtifactVisibility.PRIVATE

    def bump_behind_the_tools_back(self, artifact_id: str, content: str) -> None:
        """Simulate another writer, e.g. a teammate editing in the UI."""
        artifact = self._artifacts[artifact_id]
        self._add_version(artifact, content, artifact.name)

    def _find(self, artifact_id: str, user_id: str | None = None) -> Artifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None or (user_id is not None and artifact.user_id != user_id):
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found", "get", artifact_id)
        return artifact

    def _add_version(
        self,
        artifact: Artifact,
        content: str | bytes,
        name: str,
        description: str | None = None,
        entry_path: str | None = None,
    ):
        versions = self._versions.setdefault(artifact.id, [])
        raw = content.encode() if isinstance(content, str) else content
        number = len(versions) + 1
        # The platform opens a bundle on index.html unless told otherwise, on every version.
        if artifact.kind == ArtifactKind.BUNDLE and entry_path is None:
            entry_path = "index.html"
        version = ArtifactVersion(
            id=f"{artifact.id}-v{number}",
            artifact_id=artifact.id,
            version=number,
            name=name,
            description=description,
            mime_type=artifact.mime_type,
            size=len(raw),
            checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
            entry_path=entry_path,
        )
        versions.append((version, content))
        artifact.name = name
        artifact.latest_version = version


@pytest.fixture
def fake_backend():
    return FakeArtifactBackend()
