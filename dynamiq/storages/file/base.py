"""Base file storage interface and common data structures."""

import abc
import base64
from datetime import datetime
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer, field_validator


class FileInfo(BaseModel):
    """Information about a stored file."""

    name: str
    path: str
    size: int
    content_type: str = "text/plain"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: bytes | None = Field(default=None)

    @field_serializer("content", when_used="json-unless-none")
    def bytes_as_base64(self, b: bytes) -> str:
        return base64.b64encode(b).decode("ascii")

    def to_bytesio(self) -> BytesIO:
        """Return file content wrapped in a ``BytesIO`` with metadata attributes.

        Returns:
            BytesIO with ``name``, ``path``, ``size``, ``created_at``,
            ``description``, and ``content_type`` set.
        """
        bio = BytesIO(self.content or b"")
        bio.name = self.name
        bio.path = self.path
        bio.size = self.size
        bio.created_at = self.created_at
        bio.description = self.metadata.get("description", "")
        bio.content_type = self.content_type
        return bio


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(self, message: str, operation: str = None, path: str = None):
        self.message = message
        self.operation = operation
        self.path = path
        super().__init__(self.message)


class FileNotFoundError(StorageError):
    """Raised when a file is not found in storage."""

    pass


class FileExistsError(StorageError):
    """Raised when trying to create a file that already exists."""

    pass


class PermissionError(StorageError):
    """Raised when permission is denied for a storage operation."""

    pass


class FileStore(abc.ABC, BaseModel):
    """Abstract base class for file storage implementations.

    This interface provides a unified way to interact with different
    file storage backends (in-memory, file system, cloud storage, etc.).

    The ``_clone_shared`` flag tells the Node.clone() machinery to
    share this instance by reference instead of deep-copying it.
    This preserves stored files across cloned tool invocations.
    """

    _clone_shared: ClassVar[bool] = True

    @computed_field
    @cached_property
    def type(self) -> str:
        """Returns the backend type as a string."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the FileStore instance to a dictionary.

        Returns:
            dict: Dictionary representation of the FileStore instance.
        """
        for param in ("include_secure_params", "for_tracing"):
            kwargs.pop(param, None)
        data = self.model_dump(**kwargs)
        data["type"] = self.type
        return data

    def supports_extracted_text_cache(self, file_path: str | Path = "") -> bool:
        """Whether ``FileReadTool`` may cache extracted text beside ``file_path``.

        Reading a file that needs conversion (PDF, DOCX, ...) writes the extracted text back as
        ``<path>.extracted.txt`` so later reads and searches skip the converter. That trade is only
        worth it when writes are cheap and the namespace is private to the run. A remote or durable
        store should decline: it would pay an extra write request per read and leave derived files
        sitting in a namespace the user curates.
        """
        return True

    @abc.abstractmethod
    def list_files_bytes(self, file_paths: list[str] | None = None) -> list[BytesIO]:
        """Return stored files as BytesIO objects.

        Args:
            file_paths: If provided, return only these files. Otherwise return all files.

        Returns:
            List of BytesIO objects with name, description, and content_type attributes.
        """
        pass

    @abc.abstractmethod
    def store(
        self,
        file_path: str | Path,
        content: str | bytes | BinaryIO,
        content_type: str = None,
        metadata: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> FileInfo:
        """Store a file in the storage backend.

        Args:
            file_path: Path where the file should be stored
            content: File content as string, bytes, or file-like object
            content_type: MIME type of the file content
            metadata: Additional metadata to store with the file
            overwrite: Whether to overwrite existing files

        Returns:
            FileInfo object with details about the stored file

        Raises:
            FileExistsError: If file exists and overwrite=False
            PermissionError: If storage operation is not permitted
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def retrieve(self, file_path: str | Path) -> bytes:
        """Retrieve file content from storage.

        Args:
            file_path: Path of the file to retrieve

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If retrieval is not permitted
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def exists(self, file_path: str | Path) -> bool:
        """Check if a file exists in storage.

        Args:
            file_path: Path of the file to check

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abc.abstractmethod
    def delete(self, file_path: str | Path) -> bool:
        """Delete a file from storage.

        Args:
            file_path: Path of the file to delete

        Returns:
            True if file was deleted, False if it didn't exist

        Raises:
            PermissionError: If deletion is not permitted
            StorageError: For other storage-related errors
        """
        pass

    @abc.abstractmethod
    def list_files(self, directory: str | Path = "", recursive: bool = False, pattern: str = None) -> list[FileInfo]:
        """List files in storage.

        Args:
            directory: Directory to list (empty string for root)
            recursive: Whether to list files recursively
            pattern: Glob pattern to filter files

        Returns:
            List of FileInfo objects
        """
        pass


MEMORY_ROOT = "memories/"


class PersistentStoreConfig(BaseModel):
    """Configuration for a persistent, cross-conversation file namespace.

    This is deliberately a separate config from ``FileStoreConfig`` rather than a field on it: an
    Agent refuses to enable a file store and a sandbox at the same time, so anything nested under
    ``file_store`` would be unreachable for sandbox-backed agents - precisely the agents that need
    only this persistent route and already have a workspace.

    An Agent takes one of these or a list of them. Each entry is one memory, addressed under its own
    ``path_prefix``; ``name`` and ``description`` are what the model is told about it, so it can tell
    one memory from another and file a fact in the right one.

    Attributes:
        enabled: Whether the persistent store is active.
        backend: The store holding persistent files (``DynamiqFileStore`` in practice).
        path_prefix: Path prefix the agent uses to address persistent files.
        write_enabled: Whether the agent may create and edit persistent files.
        name: Short identifier for this memory, shown to the model.
        description: What this memory holds, shown to the model.
    """

    enabled: bool = False
    backend: FileStore = Field(..., description="Store holding persistent files.")
    path_prefix: str = Field(
        default="",
        description=(
            f"Namespace for this memory, addressed under the fixed '{MEMORY_ROOT}' root. Give a "
            "name like 'user' or 'company'; leave empty to hold the root itself. A leading "
            f"'{MEMORY_ROOT}' is accepted and stripped, so both forms mean the same thing."
        ),
    )
    write_enabled: bool = Field(
        default=True,
        description="Whether the agent is permitted to write persistent files.",
    )
    name: str | None = Field(
        default=None,
        description="Short identifier for this memory, shown to the model.",
    )
    description: str | None = Field(
        default=None,
        description="What this memory holds, shown to the model so it can tell memories apart.",
    )

    @field_validator("path_prefix")
    @classmethod
    def clean_namespace(cls, value: str) -> str:
        """Reduce ``path_prefix`` to a namespace under the root, or reject it.

        Every memory lives under ``MEMORY_ROOT``; what a caller supplies only names a place inside
        it. Accepting the root spelled out means the two natural ways of writing it agree instead of
        producing ``memories/memories/...``.
        """
        namespace = value.strip().strip("/")
        if namespace == MEMORY_ROOT.strip("/"):
            namespace = ""
        elif namespace.startswith(MEMORY_ROOT):
            namespace = namespace[len(MEMORY_ROOT) :].strip("/")

        if ".." in namespace.split("/") or ":" in namespace:
            raise ValueError(f"path_prefix must be a plain namespace under '{MEMORY_ROOT}', got {value!r}")
        return namespace

    @property
    def normalized_prefix(self) -> str:
        """The full path this memory is addressed by: the fixed root plus its namespace."""
        return f"{MEMORY_ROOT}{self.path_prefix}/" if self.path_prefix else MEMORY_ROOT

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the PersistentStoreConfig instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        config_data = self.model_dump(exclude=exclude, **kwargs)
        config_data["backend"] = self.backend.to_dict(
            for_tracing=for_tracing, include_secure_params=include_secure_params
        )
        return config_data


def describe_memory_namespaces(configs: "list[PersistentStoreConfig]") -> str:
    """One line per memory, for the prompt and the ``memory-*`` tool descriptions.

    Empty only when there is nothing to tell apart *and* nothing to locate: a lone, unnamed memory
    sitting at the root is fully described by the surrounding wording, which is what agents
    configured before memories could be plural still render.

    A namespaced memory is always listed even when unnamed. The protocol points the agent at the
    root, so without this line it would never learn the one path its writes must go under - and a
    write to the root would fall through to the workspace and quietly not persist.
    """
    if len(configs) <= 1 and not any(
        config.name or config.description or config.normalized_prefix != MEMORY_ROOT for config in configs
    ):
        return ""

    lines = []
    for config in configs:
        detail = " ".join(
            part
            for part in (
                f"{config.name}:" if config.name else "",
                config.description or "",
                "" if config.write_enabled else "Read-only.",
            )
            if part
        )
        lines.append(f"- {config.normalized_prefix} - {detail}" if detail else f"- {config.normalized_prefix}")
    return "\n".join(lines)


def memory_root(configs: "list[PersistentStoreConfig]") -> str:
    """The path the agent lists to see every memory at once.

    Always the fixed root: memories are namespaces beneath it, so one listing reaches all of them.
    Taking ``configs`` keeps the call sites unchanged and the intent readable.
    """
    return MEMORY_ROOT


class FileStoreConfig(BaseModel):
    """Configuration for file storage and related features.

    Attributes:
        enabled: Whether file storage is enabled.
        backend: The file storage backend to use.
        agent_file_write_enabled: Whether the agent can write files.
        todo_enabled: Whether to enable todo management tools (stored in ._agent/todos.json).
        config: Additional configuration options.
    """

    enabled: bool = False
    backend: FileStore = Field(..., description="File storage to use.")
    agent_file_write_enabled: bool = Field(
        default=False, description="Whether the agent is permitted to write files to the file store."
    )
    todo_enabled: bool = Field(
        default=False, description="Whether to enable todo management tools (todos stored in ._agent/todos.json)."
    )
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the FileStoreConfig instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        kwargs.pop("include_secure_params", None)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        config_data = self.model_dump(exclude=exclude, **kwargs)
        config_data["backend"] = self.backend.to_dict()
        return config_data
