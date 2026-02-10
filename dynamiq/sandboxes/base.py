"""Base sandbox interface and common data structures."""

import abc
from enum import Enum
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.connections.connections import BaseConnection
from dynamiq.nodes.node import Node


class SandboxTool(str, Enum):
    """Enum for sandbox tool types."""

    SHELL = "shell"
    FILES = "files"


class ShellCommandResult(BaseModel):
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int | None


class FileEntry(BaseModel):
    """Entry representing a file or directory in the sandbox."""

    name: str
    path: str
    is_dir: bool = False
    size: int | None = None


class Sandbox(abc.ABC, BaseModel):
    """Abstract base class for sandbox implementations.

    This interface provides a unified way to interact with different
    sandbox backends (in-memory, file system, E2B, Docker, etc.).
    Sandboxes provide file storage and can be extended to support
    code execution and other isolated environment capabilities.
    """

    connection: BaseConnection | None = Field(default=None, description="Connection to the sandbox backend.")
    tools: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for sandbox tools. Keys are tool names (e.g., 'shell'), values are tool configs.",
    )

    @computed_field
    @cached_property
    def type(self) -> str:
        """Returns the backend type as a string."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the Sandbox instance to a dictionary.

        Args:
            for_tracing: If True, exclude sensitive fields like connection credentials.

        Returns:
            dict: Dictionary representation of the Sandbox instance.
        """
        for_tracing = kwargs.pop("for_tracing", False)
        kwargs.pop("include_secure_params", None)

        has_connection = getattr(self, "connection", None) is not None
        exclude_fields = {"connection"} if has_connection else set()
        data = self.model_dump(exclude=exclude_fields, **kwargs)
        data["type"] = self.type

        if has_connection:
            data["connection"] = self.connection.to_dict(for_tracing=for_tracing, **kwargs)

        return data

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the sandbox.

        This is an optional capability. Subclasses that support command execution
        should override this method. The base implementation raises NotImplementedError.

        Args:
            command: Shell command or script to execute.
            timeout: Timeout in seconds (default 60).
            run_in_background_enabled: If True, run command in background (no output).

        Returns:
            ShellCommandResult with stdout, stderr, and exit_code.

        Raises:
            NotImplementedError: If the sandbox does not support command execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support command execution. "
            "Use a sandbox backend that supports shell commands (e.g., E2BSandbox)."
        )

    @abc.abstractmethod
    def get_tools(self) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Subclasses must implement this method to return tools specific
        to their sandbox type. Tools are configured via the `tools` field.

        Returns:
            List of tool instances (Node objects).
        """
        ...

    def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox.

        Args:
            path: Path of the file in the sandbox.

        Returns:
            File content as bytes.

        Raises:
            NotImplementedError: If the sandbox does not support file reads.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file reads. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def write_file(self, path: str, content: bytes) -> str:
        """Write a file to the sandbox.

        Args:
            path: Destination path in the sandbox.
            content: File content as bytes.

        Returns:
            The path where the file was written.

        Raises:
            NotImplementedError: If the sandbox does not support file writes.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file writes. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def list_files(self, path: str) -> list[FileEntry]:
        """List files and directories at the given path in the sandbox.

        Args:
            path: Directory path in the sandbox.

        Returns:
            List of FileEntry objects.

        Raises:
            NotImplementedError: If the sandbox does not support file listing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file listing. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def upload_file(self, file_name: str, content: bytes, destination_path: str | None = None) -> str:
        """Upload a file to the sandbox.

        Args:
            file_name: Name of the file.
            content: File content as bytes.
            destination_path: Optional destination path in sandbox. If None, uses base_path/file_name.

        Returns:
            The path where the file was uploaded in the sandbox.

        Raises:
            NotImplementedError: If the sandbox does not support file uploads.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file uploads. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def close(self) -> None:
        """Close the sandbox."""
        raise NotImplementedError(f"Implementation of close() is not implemented for {self.__class__.__name__}")


class SandboxConfig(BaseModel):
    """Configuration for sandbox and related features.

    Attributes:
        enabled: Whether sandbox is enabled.
        backend: The sandbox backend to use.
        config: Additional configuration options.
    """

    enabled: bool = False
    backend: Sandbox = Field(..., description="Sandbox backend to use.")
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the SandboxConfig instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        kwargs.pop("include_secure_params", None)
        if for_tracing and not self.enabled:
            return {"enabled": False}
        config_data = self.model_dump(exclude={"backend"}, **kwargs)
        config_data["backend"] = self.backend.to_dict(for_tracing=for_tracing, **kwargs)
        return config_data
