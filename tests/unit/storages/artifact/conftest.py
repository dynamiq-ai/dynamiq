import hashlib
from typing import Any, Sequence

import pytest
from pydantic import Field

from dynamiq.storages.artifact import (
    Artifact,
    ArtifactConflictError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactVersion,
    default_media_type,
)


class FakeArtifactStore(ArtifactStore):
    """A local stand-in for the API-backed store, applying edits the way the server would."""

    calls: list[tuple[str, dict[str, Any]]] = Field(default_factory=list, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._artifacts: dict[str, Artifact] = {}
        self._contents: dict[tuple[str, int], str | bytes] = {}

    def _version(self, artifact: Artifact, version: int, content: str | bytes, title: str, summary=None):
        raw = content.encode() if isinstance(content, str) else content
        self._contents[(artifact.id, version)] = content
        return ArtifactVersion(
            id=f"{artifact.id}-v{version}",
            artifact_id=artifact.id,
            version=version,
            title=title,
            kind=artifact.kind,
            media_type=artifact.media_type,
            size=len(raw),
            checksum="sha256:" + hashlib.sha256(raw).hexdigest(),
            url=f"https://artifacts.example/{artifact.id}/v{version}",
            summary=summary,
        )

    def create(self, *, name, title, kind, content, media_type=None, summary=None, source=None, metadata=None):
        self.calls.append(("create", {"name": name, "kind": kind, "content": content, "source": source}))
        artifact_id = f"a{len(self._artifacts) + 1}"
        artifact = Artifact(
            id=artifact_id,
            name=name,
            title=title,
            kind=ArtifactKind(kind),
            media_type=media_type or default_media_type(ArtifactKind(kind), name),
            url=f"https://artifacts.example/{artifact_id}",
        )
        artifact.latest = self._version(artifact, 1, content, title, summary)
        self._artifacts[artifact_id] = artifact
        return artifact.model_copy(deep=True)

    def update(
        self,
        artifact_id,
        *,
        content=None,
        edits: Sequence[Any] | None = None,
        title=None,
        summary=None,
        if_match=None,
        source=None,
    ):
        self.calls.append(("update", {"artifact_id": artifact_id, "if_match": if_match, "source": source}))
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found", "update", artifact_id)
        if if_match and if_match != artifact.latest.checksum:
            raise ArtifactConflictError(f"Artifact '{artifact_id}' changed", "update", artifact_id)
        if edits:
            content = self._contents[(artifact_id, artifact.latest.version)]
            for edit in edits:
                content = content.replace(edit.find, edit.replace, -1 if edit.replace_all else 1)
        if title:
            artifact.title = title
        artifact.latest = self._version(artifact, artifact.latest.version + 1, content, artifact.title, summary)
        return artifact.model_copy(deep=True)

    def get(self, artifact_id, version=None, include_content=True):
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ArtifactNotFoundError(f"Artifact '{artifact_id}' not found", "get", artifact_id)
        resolved = version or artifact.latest.version
        if (artifact_id, resolved) not in self._contents:
            raise ArtifactNotFoundError(f"Version {resolved} not found", "get", artifact_id)
        return artifact.model_copy(deep=True), self._contents[(artifact_id, resolved)] if include_content else None

    def list(self, *, kind=None, query=None, limit=50):
        found = [a for a in self._artifacts.values() if (kind is None or a.kind == kind)]
        if query:
            found = [a for a in found if query.lower() in a.title.lower()]
        return [a.model_copy(deep=True) for a in reversed(found)][:limit]

    def bump_behind_the_tools_back(self, artifact_id: str, content: str) -> None:
        """Simulate another writer, e.g. a teammate editing in the UI."""
        artifact = self._artifacts[artifact_id]
        artifact.latest = self._version(artifact, artifact.latest.version + 1, content, artifact.title)


@pytest.fixture
def fake_store():
    return FakeArtifactStore()
