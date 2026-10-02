import abc
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, computed_field

from dynamiq.artifacts.types import Artifact, ArtifactKind, ArtifactShare


class ArtifactBackend(abc.ABC, BaseModel):
    """Abstract base class for artifact backends.

    Every write creates a new immutable version; nothing is mutated in place. Ownership and access
    are derived from the backend's credentials and configuration, never supplied by the agent.

    ``_clone_shared`` tells ``Node.clone()`` to share this instance by reference rather than
    deep-copying it, so parallel cloned tools reach the same backend.
    """

    _clone_shared: ClassVar[bool] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def type(self) -> str:
        """Dotted path used to rebuild this backend from a serialized workflow."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Parameters to drop when serializing."""
        return {}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the backend to a dictionary."""
        kwargs.pop("for_tracing", None)
        kwargs.pop("include_secure_params", None)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        return data

    @abc.abstractmethod
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
    ) -> Artifact:
        """Create an artifact with its first version. ``entry_path`` applies to bundles only."""

    @abc.abstractmethod
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
    ) -> Artifact:
        """Add a version with the full content and return the artifact at that version.

        ``if_match`` is the id of the version the caller last saw; when a newer one exists,
        ``ArtifactConflictError`` is raised and nothing is written. Content equal to the latest
        version's adds no version.
        """

    @abc.abstractmethod
    def get(
        self, artifact_id: str, version: int | None = None, include_content: bool = True
    ) -> tuple[Artifact, str | bytes | None]:
        """Return an artifact and, when asked, the content of the requested (default latest) version.

        Text kinds come back as ``str`` and the others as ``bytes``.

        Raises:
            ArtifactNotFoundError: If the artifact or version does not exist.
        """

    @abc.abstractmethod
    def list(self, *, kind: ArtifactKind | None = None, limit: int = 50) -> list[Artifact]:
        """List the artifacts the backend reaches, most recently updated first."""

    @abc.abstractmethod
    def share(
        self, artifact_id: str, *, pinned_version: int | None = None, expires_at: datetime | None = None
    ) -> ArtifactShare:
        """Create the artifact's link share, or update the active one, and return it.

        Without ``pinned_version`` the link follows the latest version; without ``expires_at`` it
        does not expire.
        """

    @abc.abstractmethod
    def unshare(self, artifact_id: str) -> None:
        """Revoke the artifact's link share. Sharing again creates a new link."""
