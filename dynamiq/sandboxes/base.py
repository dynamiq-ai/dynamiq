"""Base sandbox interface and common data structures."""

import abc
import io
import logging
import mimetypes
from enum import Enum
from functools import cached_property
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.connections.connections import BaseConnection
from dynamiq.nodes.node import Node


class SandboxTool(str, Enum):
    """Enum for sandbox tool types."""

    SHELL = "shell"


class ShellCommandResult(BaseModel):
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int | None


class SandboxInfo(BaseModel):
    """Schema for sandbox metadata returned by get_sandbox_info()."""

    base_path: str
    output_dir: str
    sandbox_id: str | None = None
    public_host: str | None = None
    public_url: str | None = None
    public_url_error: str | None = None

    model_config = ConfigDict(extra="allow")


class Sandbox(abc.ABC, BaseModel):
    """Abstract base class for sandbox implementations.

    This interface provides a unified way to interact with different
    sandbox backends (in-memory, file system, E2B, Docker, etc.).
    Sandboxes provide file storage and can be extended to support
    code execution and other isolated environment capabilities.
    """

    connection: BaseConnection | None = Field(default=None, description="Connection to the sandbox backend.")
    base_path: str = Field(default="/home/user", description="Base path in the sandbox filesystem.")
    max_output_files: int = Field(
        default=50, description="Maximum number of files to collect from the output directory."
    )
    _clone_shared: ClassVar[bool] = True

    @property
    def output_dir(self) -> str:
        """Absolute path to the output directory inside the sandbox."""
        return f"{self.base_path}/output"

    @computed_field
    @cached_property
    def type(self) -> str:
        """Returns the backend type as a string."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"connection": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the Sandbox instance to a dictionary.

        Args:
            for_tracing: If True, exclude sensitive fields like connection credentials.

        Returns:
            dict: Dictionary representation of the Sandbox instance.
        """
        for_tracing = kwargs.pop("for_tracing", False)
        kwargs.pop("include_secure_params", None)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)

        has_connection = getattr(self, "connection", None) is not None
        data = self.model_dump(exclude=exclude, **kwargs)
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
    def get_tools(self, llm: Any = None) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Subclasses must implement this method to return tools specific
        to their sandbox type. Tools are configured via the `tools` field.

        Args:
            llm: Optional LLM instance passed to tools that require one (e.g. FileReadTool).

        Returns:
            List of tool instances (Node objects).
        """
        ...

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

    def list_output_files(self) -> list[str]:
        """List files in the sandbox output directory.

        Args:
            target_dir: Directory to list. Defaults to the output directory.

        Returns:
            List of absolute file paths found in the output directory.
        """
        return self.list_files(target_dir=self.output_dir)

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the sandbox directory.

        Args:
            target_dir: Directory to list. Defaults to the output directory.

        Implementations should respect ``max_output_files`` when scanning
        for files.

        Returns:
            List of absolute file paths found in the directory.

        Raises:
            NotImplementedError: If the sandbox does not support file listing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file listing. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def is_output_empty(self) -> bool:
        """Check whether the sandbox output directory contains any files.

        Returns:
            True if the output directory is empty or does not exist, False otherwise.
        """
        try:
            return len(self.list_output_files()) == 0
        except NotImplementedError:
            return True

    def collect_files(self, target_dir: str | None = None, file_paths: list[str] | None = None) -> list[io.BytesIO]:
        """Collect files from the sandbox directory as BytesIO objects.

        Args:
            target_dir: Directory to collect files from. Defaults to the base path.
            file_paths: List of file paths to collect. If None, all files in the target directory are collected.

        Returns:
            List of BytesIO objects with name, description, and content_type attributes.

        Raises:
            FileNotFoundError: If explicit ``file_paths`` were requested and any
                of them could not be retrieved.
        """
        file_paths_requested = bool(file_paths)

        if file_paths_requested:
            resolved: list[str] = []
            for file_path in file_paths:
                if not file_path.startswith("/"):
                    file_path = f"{self.base_path.rstrip('/')}/{file_path.lstrip('/')}"
                resolved.append(file_path)
            file_paths = resolved

        if not file_paths:
            file_paths = self.list_files(target_dir=target_dir)

        if not file_paths:
            return []

        result_files: list[io.BytesIO] = []
        for file_path in file_paths:
            file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
            try:
                content = self.retrieve(file_path)
                content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

                file_bytesio = io.BytesIO(content)
                file_bytesio.name = file_name
                file_bytesio.description = f"Generated file from sandbox: {file_path}"
                file_bytesio.content_type = content_type
                file_bytesio.seek(0)

                result_files.append(file_bytesio)
            except Exception as e:
                if file_paths_requested:
                    raise FileNotFoundError(f"Failed to download requested file '{file_path}': {e}") from e
                logging.getLogger(__name__).warning(f"Failed to download file '{file_path}': {e}")
                continue

        return result_files

    def collect_output_files(self) -> list[io.BytesIO]:
        """Collect output files from the sandbox output directory as BytesIO objects.

        Only files placed in the dedicated output directory are collected.

        Returns:
            List of BytesIO objects with name, description, and content_type attributes.
        """
        return self.collect_files(target_dir=self.output_dir)

    def exists(self, file_path: str) -> bool:
        """Check whether a file exists in the sandbox filesystem.

        Required for FileReadTool compatibility.

        Args:
            file_path: Path to the file (relative or absolute).

        Returns:
            True if the file exists, False otherwise.

        Raises:
            NotImplementedError: If the sandbox does not support file existence checks.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file existence checks. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def retrieve(self, file_path: str) -> bytes:
        """Read file content from the sandbox filesystem.

        Required for FileReadTool compatibility.

        Args:
            file_path: Path to the file (relative or absolute).

        Returns:
            The file content as bytes.

        Raises:
            NotImplementedError: If the sandbox does not support file retrieval.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support file retrieval. "
            "Use a sandbox backend that supports file operations (e.g., E2BSandbox)."
        )

    def get_sandbox_info(self, port: int | None = None) -> SandboxInfo:
        """Return sandbox metadata for the agent (e.g. base_path, optional public URL for a port).

        Subclasses that support a public URL (e.g. E2B) may override and include
        sandbox_id, public_host, and public_url when port is provided.

        Args:
            port: Optional port number; if provided and the backend supports it,
                the returned schema may include public_host and public_url.

        Returns:
            SandboxInfo with at least base_path and output_dir; backends may add
            sandbox_id, public_host, public_url (when port is given), etc.
        """
        return SandboxInfo(
            base_path=self.base_path,
            output_dir=self.output_dir,
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

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"backend": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the SandboxConfig instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        kwargs.pop("include_secure_params", None)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        config_data = self.model_dump(exclude=exclude, **kwargs)
        config_data["backend"] = self.backend.to_dict(for_tracing=for_tracing, **kwargs)
        return config_data
