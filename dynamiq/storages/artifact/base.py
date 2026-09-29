"""Base artifact-store interface and common data structures.

An artifact is a deliverable an agent hands over: a named, typed document with immutable integer
versions and a URL that outlives the conversation. It is deliberately neither a ``FileStore`` (the
agent's workbench) nor a ``MemoryStore`` (what the agent remembers).
"""

import abc
import mimetypes
import posixpath
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ArtifactStoreError(Exception):
    """Base exception for artifact-store operations."""

    def __init__(self, message: str, operation: str | None = None, artifact_id: str | None = None):
        self.message = message
        self.operation = operation
        self.artifact_id = artifact_id
        super().__init__(self.message)


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact or version does not exist."""


class ArtifactPermissionError(ArtifactStoreError):
    """Raised when an artifact operation is not permitted."""


class ArtifactConflictError(ArtifactStoreError):
    """Raised when an update's precondition no longer matches the latest version."""


class ArtifactKind(str, Enum):
    """What an artifact is; selects the renderer. ``media_type`` is kept alongside for fidelity."""

    HTML = "html"
    MARKDOWN = "markdown"
    CODE = "code"
    SVG = "svg"
    MERMAID = "mermaid"
    JSON = "json"
    CSV = "csv"
    CHART = "chart"
    IMAGE = "image"
    PDF = "pdf"
    FILE = "file"
    BUNDLE = "bundle"


TEXT_KINDS = frozenset(
    {
        ArtifactKind.HTML,
        ArtifactKind.MARKDOWN,
        ArtifactKind.CODE,
        ArtifactKind.SVG,
        ArtifactKind.MERMAID,
        ArtifactKind.JSON,
        ArtifactKind.CSV,
        ArtifactKind.CHART,
    }
)

_DEFAULT_MEDIA_TYPES = {
    ArtifactKind.HTML: "text/html",
    ArtifactKind.MARKDOWN: "text/markdown",
    ArtifactKind.CODE: "text/plain",
    ArtifactKind.SVG: "image/svg+xml",
    ArtifactKind.MERMAID: "text/vnd.mermaid",
    ArtifactKind.JSON: "application/json",
    ArtifactKind.CSV: "text/csv",
    ArtifactKind.CHART: "application/json",
    ArtifactKind.IMAGE: "image/png",
    ArtifactKind.PDF: "application/pdf",
    ArtifactKind.FILE: "application/octet-stream",
    ArtifactKind.BUNDLE: "application/zip",
}

_DEFAULT_EXTENSIONS = {
    ArtifactKind.HTML: ".html",
    ArtifactKind.MARKDOWN: ".md",
    ArtifactKind.SVG: ".svg",
    ArtifactKind.MERMAID: ".mmd",
    ArtifactKind.JSON: ".json",
    ArtifactKind.CSV: ".csv",
    ArtifactKind.CHART: ".json",
    ArtifactKind.PDF: ".pdf",
    ArtifactKind.BUNDLE: ".zip",
}

_EXTENSION_KINDS = {
    ".html": ArtifactKind.HTML,
    ".htm": ArtifactKind.HTML,
    ".md": ArtifactKind.MARKDOWN,
    ".markdown": ArtifactKind.MARKDOWN,
    ".svg": ArtifactKind.SVG,
    ".mmd": ArtifactKind.MERMAID,
    ".mermaid": ArtifactKind.MERMAID,
    ".json": ArtifactKind.JSON,
    ".csv": ArtifactKind.CSV,
    ".png": ArtifactKind.IMAGE,
    ".jpg": ArtifactKind.IMAGE,
    ".jpeg": ArtifactKind.IMAGE,
    ".gif": ArtifactKind.IMAGE,
    ".webp": ArtifactKind.IMAGE,
    ".pdf": ArtifactKind.PDF,
}

_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift", ".rb", ".php",
        ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".sql", ".css", ".scss", ".yaml", ".yml", ".toml", ".xml",
    }
)  # fmt: skip


def infer_kind(name: str | None, media_type: str | None = None) -> ArtifactKind:
    """Best-effort kind from a filename extension, falling back to the media type."""
    ext = posixpath.splitext((name or "").lower())[1]
    if ext in _EXTENSION_KINDS:
        return _EXTENSION_KINDS[ext]
    if ext in _CODE_EXTENSIONS:
        return ArtifactKind.CODE
    if media_type:
        for kind, default in _DEFAULT_MEDIA_TYPES.items():
            if kind not in (ArtifactKind.CODE, ArtifactKind.CHART) and media_type == default:
                return kind
        if media_type.startswith("image/"):
            return ArtifactKind.IMAGE
    return ArtifactKind.FILE


def default_media_type(kind: ArtifactKind, name: str | None = None) -> str:
    """Media type for a kind, preferring what the filename says when it is more specific."""
    guessed = mimetypes.guess_type(name)[0] if name else None
    if guessed and kind in (ArtifactKind.CODE, ArtifactKind.IMAGE, ArtifactKind.FILE):
        return guessed
    return _DEFAULT_MEDIA_TYPES[kind]


def default_extension(kind: ArtifactKind) -> str:
    """Filename extension used when the caller gives a title but no name."""
    return _DEFAULT_EXTENSIONS.get(kind, "")


class ArtifactVersion(BaseModel):
    """One immutable version of an artifact, without its content."""

    id: str
    artifact_id: str
    version: int
    title: str
    kind: ArtifactKind
    media_type: str
    size: int = 0
    checksum: str | None = None
    url: str | None = None
    summary: str | None = None
    created_at: datetime | None = None


class Artifact(BaseModel):
    """An artifact and its latest version, without content."""

    id: str
    name: str
    title: str
    kind: ArtifactKind
    media_type: str
    latest: ArtifactVersion | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def version(self) -> int | None:
        return self.latest.version if self.latest else None

    def to_ref(self) -> dict[str, Any]:
        """The small reference that travels in tool output, run output and checkpoints; never bytes."""
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "title": self.title,
            "kind": self.kind.value,
            "media_type": self.media_type,
            "size": self.latest.size if self.latest else 0,
            "url": self.url or (self.latest.url if self.latest else None),
        }


class ArtifactStore(abc.ABC, BaseModel):
    """Abstract base class for artifact backends.

    Every write creates a new immutable version; nothing is mutated in place. Ownership and access
    are derived from the backend's credentials, never supplied by the agent.

    ``_clone_shared`` tells ``Node.clone()`` to share this instance by reference rather than
    deep-copying it, so parallel cloned tools reach the same store.
    """

    _clone_shared: ClassVar[bool] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def type(self) -> str:
        """Dotted path used to rebuild this store from a serialized workflow."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Parameters to drop when serializing."""
        return {}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the store to a dictionary."""
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

    @abc.abstractmethod
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
        """Add a version from full ``content`` or literal find/replace ``edits`` on the latest one.

        ``edits`` items carry ``find``, ``replace`` and ``replace_all``. ``if_match`` is the checksum
        or version number the caller last saw; a mismatch raises ``ArtifactConflictError``.
        """

    @abc.abstractmethod
    def get(
        self, artifact_id: str, version: int | None = None, include_content: bool = True
    ) -> tuple[Artifact, str | bytes | None]:
        """Return an artifact and, when asked, the content of the requested (default latest) version.

        Raises:
            ArtifactNotFoundError: If the artifact or version does not exist.
        """

    @abc.abstractmethod
    def list(self, *, kind: ArtifactKind | None = None, query: str | None = None, limit: int = 50) -> list[Artifact]:
        """List artifacts visible to the caller, newest first."""


class ArtifactStoreConfig(BaseModel):
    """Configuration for an agent's artifact store.

    Attributes:
        enabled: Whether the agent gets an artifact tool at all.
        backend: The store holding artifacts.
        write_enabled: Whether the agent may create and update artifacts.
    """

    enabled: bool = False
    backend: ArtifactStore = Field(..., description="Store holding the agent's artifacts.")
    write_enabled: bool = Field(default=True, description="Whether the agent is permitted to write artifacts.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """The backend serializes itself, so exclude it from the plain dump."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the config to a dictionary, delegating the backend to its own ``to_dict``."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["backend"] = self.backend.to_dict(for_tracing=for_tracing, include_secure_params=include_secure_params)
        return data
