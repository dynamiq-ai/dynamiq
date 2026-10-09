"""Artifact types, errors and kind helpers.

An artifact is a deliverable an agent hands over: a named, typed document with immutable integer
versions and a URL that outlives the conversation.
"""

import mimetypes
import posixpath
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ArtifactError(Exception):
    """Base exception for artifact operations."""

    def __init__(self, message: str, operation: str | None = None, artifact_id: str | None = None):
        self.message = message
        self.operation = operation
        self.artifact_id = artifact_id
        super().__init__(self.message)


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact or version does not exist, or belongs to another end user."""


class ArtifactPermissionError(ArtifactError):
    """Raised when an artifact operation is not permitted."""


class ArtifactConflictError(ArtifactError):
    """Raised when an update's If-Match no longer names the latest version."""


class ArtifactKind(str, Enum):
    """What an artifact is; selects the renderer. Fixed at create: every version shares it."""

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


class ArtifactVisibility(str, Enum):
    """Who can open an artifact besides its owner."""

    PRIVATE = "private"
    ORG = "org"
    LINK = "link"


# The kinds whose content is UTF-8 text: the only kinds the API accepts as JSON content.
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

_DEFAULT_MIME_TYPES = {
    ArtifactKind.HTML: "text/html",
    ArtifactKind.MARKDOWN: "text/markdown",
    ArtifactKind.CODE: "text/plain",
    ArtifactKind.SVG: "image/svg+xml",
    ArtifactKind.MERMAID: "text/plain",
    ArtifactKind.JSON: "application/json",
    ArtifactKind.CSV: "text/csv",
    ArtifactKind.CHART: "application/json",
    ArtifactKind.IMAGE: "image/png",
    ArtifactKind.PDF: "application/pdf",
    ArtifactKind.FILE: "application/octet-stream",
    ArtifactKind.BUNDLE: "application/zip",
}

_MIME_TYPES = mimetypes.MimeTypes()
_EXTENSION_MIME_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".webp": "image/webp",
}

# Kinds whose type follows the file, so each version declares its own.
_FILE_TYPED_KINDS = frozenset({ArtifactKind.FILE, ArtifactKind.IMAGE, ArtifactKind.CODE})

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


def infer_kind(file_name: str | None, mime_type: str | None = None) -> ArtifactKind:
    """Best-effort kind from a file name's extension, falling back to the MIME type.

    A zip stays a plain file: only an explicit ``bundle`` kind publishes it as a site.
    """
    ext = posixpath.splitext((file_name or "").lower())[1]
    if ext in _EXTENSION_KINDS:
        return _EXTENSION_KINDS[ext]
    if ext in _CODE_EXTENSIONS:
        return ArtifactKind.CODE
    if mime_type:
        for kind, default in _DEFAULT_MIME_TYPES.items():
            if kind not in (ArtifactKind.CODE, ArtifactKind.MERMAID, ArtifactKind.CHART, ArtifactKind.BUNDLE) and (
                mime_type == default
            ):
                return kind
        if mime_type.startswith("image/"):
            return ArtifactKind.IMAGE
    return ArtifactKind.FILE


def default_mime_type(kind: ArtifactKind, file_name: str | None = None) -> str:
    """MIME type for a kind; images and plain files take the file name's type when it has one."""
    if kind in (ArtifactKind.IMAGE, ArtifactKind.FILE) and file_name:
        ext = posixpath.splitext(file_name.lower())[1]
        guessed = _EXTENSION_MIME_TYPES.get(ext) or _MIME_TYPES.guess_type(file_name)[0]
        if guessed:
            return guessed
    return _DEFAULT_MIME_TYPES[kind]


def version_mime_type(kind: ArtifactKind, file_name: str) -> str | None:
    """The type a new version declares: its file's for file, image and code; else None, so the kind fixes it."""
    return default_mime_type(kind, file_name) if kind in _FILE_TYPED_KINDS else None


def default_extension(kind: ArtifactKind) -> str:
    """File name extension used when the caller gives a name but no file name."""
    return _DEFAULT_EXTENSIONS.get(kind, "")


class ArtifactVersion(BaseModel):
    """One immutable version of an artifact, without its content."""

    id: str
    artifact_id: str
    version: int
    name: str
    description: str | None = None
    mime_type: str
    size: int = 0
    checksum: str | None = None
    entry_path: str | None = None
    created_at: datetime | None = None


class Artifact(BaseModel):
    """An artifact and its latest version, without content.

    ``store_id`` is set for an artifact in an artifact store, and ``user_id`` for one that belongs to
    an end user of an app within that store.
    """

    id: str
    file_name: str
    name: str
    kind: ArtifactKind
    mime_type: str
    description: str | None = None
    visibility: ArtifactVisibility = ArtifactVisibility.PRIVATE
    store_id: str | None = None
    user_id: str | None = None
    latest_version: ArtifactVersion | None = None
    url: str | None = None

    @property
    def version(self) -> int | None:
        return self.latest_version.version if self.latest_version else None

    def to_ref(self) -> dict[str, Any]:
        """The small reference that travels in tool output, run output and checkpoints; never bytes.

        It is the shape of the platform's artifact message part: the platform attaches a run's
        artifacts to the chat message by ``id`` and ``version_id``.
        """
        return {
            "id": self.id,
            "version_id": self.latest_version.id if self.latest_version else None,
            "version": self.version,
            "name": self.name,
            "kind": self.kind.value,
            "url": self.url,
        }


class ArtifactShare(BaseModel):
    """An artifact's active link share. ``pinned_version_id`` is unset when it follows the latest."""

    id: str
    artifact_id: str
    url: str | None = None
    pinned_version_id: str | None = None
    expires_at: datetime | None = None
