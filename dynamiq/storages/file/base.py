"""Base file storage interface and common data structures."""

import abc
from datetime import datetime
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.nodes.node import Node


class FileInfo(BaseModel):
    """Information about a stored file."""

    name: str
    path: str
    size: int
    content_type: str = "text/plain"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: bytes = Field(default=None)


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


class Sandbox(abc.ABC, BaseModel):
    """Abstract base class for sandbox implementations.

    This interface provides a unified way to interact with different
    sandbox backends (in-memory, file system, E2B, Docker, etc.).
    Sandboxes provide file storage and can be extended to support
    code execution and other isolated environment capabilities.
    """

    @computed_field
    @cached_property
    def type(self) -> str:
        """Returns the backend type as a string."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the Sandbox instance to a dictionary.

        Returns:
            dict: Dictionary representation of the Sandbox instance.
        """
        for param in ("include_secure_params", "for_tracing"):
            kwargs.pop(param, None)
        data = self.model_dump(**kwargs)
        data["type"] = self.type
        return data

    @abc.abstractmethod
    def list_files_bytes(self) -> list[BytesIO]:
        """List files in storage and return the content as bytes in BytesIO objects.

        Returns:
            List of BytesIO objects
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

    def get_tools(
        self,
        llm: Any = None,
        file_write_enabled: bool = False,
        todo_enabled: bool = False,
    ) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Base implementation returns file tools. Subclasses can override
        to add additional tools (e.g., code execution) by calling
        super().get_tools() and extending the list.

        Args:
            llm: LLM instance for tools that need it (e.g., FileReadTool).
            file_write_enabled: Whether to include file write tool.
            todo_enabled: Whether to include todo management tool.

        Returns:
            List of tool instances (Node objects).
        """
        # Lazy imports to avoid circular dependencies
        from dynamiq.nodes.tools.file_tools import FileListTool, FileReadTool, FileSearchTool, FileWriteTool

        tools: list[Any] = [
            FileReadTool(file_store=self, llm=llm),
            FileSearchTool(file_store=self),
            FileListTool(file_store=self),
        ]

        if file_write_enabled:
            tools.append(FileWriteTool(file_store=self))

        if todo_enabled:
            from dynamiq.nodes.tools.todo_tools import TodoWriteTool

            tools.append(TodoWriteTool(file_store=self))

        return tools


class SandboxConfig(BaseModel):
    """Configuration for sandbox and related features.

    Attributes:
        enabled: Whether sandbox is enabled.
        backend: The sandbox backend to use.
        agent_file_write_enabled: Whether the agent can write files.
        todo_enabled: Whether to enable todo management tools (stored in ._agent/todos.json).
        config: Additional configuration options.
    """

    enabled: bool = False
    backend: Sandbox = Field(..., description="Sandbox backend to use.")
    agent_file_write_enabled: bool = Field(
        default=False, description="Whether the agent is permitted to write files to the sandbox."
    )
    todo_enabled: bool = Field(
        default=False, description="Whether to enable todo management tools (todos stored in ._agent/todos.json)."
    )
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the SandboxConfig instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        kwargs.pop("include_secure_params", None)
        config_data = self.model_dump(exclude={"backend"}, **kwargs)
        config_data["backend"] = self.backend.to_dict()
        return config_data


# Backwards compatibility aliases
FileStoreConfig = SandboxConfig
FileStore = Sandbox
