"""Base file storage interface and common data structures."""

import abc
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.connections.connections import BaseConnection
from dynamiq.nodes.node import Node


class ShellCommandResult(BaseModel):
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    exit_code: int | None


class Sandbox(abc.ABC, BaseModel):
    """Abstract base class for sandbox implementations.

    This interface provides a unified way to interact with different
    sandbox backends (in-memory, file system, E2B, Docker, etc.).
    Sandboxes provide file storage and can be extended to support
    code execution and other isolated environment capabilities.
    """

    connection: BaseConnection | None = Field(default=None, description="Connection to the sandbox backend.")

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

    def get_tools(self) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Base implementation returns shell tool. Subclasses can override
        to add or replace tools specific to their sandbox type.

        Returns:
            List of tool instances (Node objects).
        """
        # Lazy import to avoid circular dependency
        from dynamiq.sandboxes.tools.shell import SandboxShellTool

        shell_tool = SandboxShellTool(
            name="shell",
            sandbox=self,
        )
        return [shell_tool]

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
