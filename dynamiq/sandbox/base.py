"""Base file storage interface and common data structures."""

import abc
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

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

    def run_command(
        self,
        command: str,
        timeout: int = 60,
        background: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the sandbox.

        This is an optional capability. Subclasses that support command execution
        should override this method. The base implementation raises NotImplementedError.

        Args:
            command: Shell command or script to execute.
            timeout: Timeout in seconds (default 60).
            background: If True, run command in background (no output).

        Returns:
            ShellCommandResult with stdout, stderr, and exit_code.

        Raises:
            NotImplementedError: If the sandbox does not support command execution.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support command execution. "
            "Use a sandbox backend that supports shell commands (e.g., E2BSandbox)."
        )

    def get_tools(
        self,
        llm: Any = None,
        file_write_enabled: bool = False,
    ) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Base implementation returns an empty list. Subclasses can override
        to add tools specific to their sandbox type (e.g., shell execution).

        Args:
            llm: LLM instance for tools that need it.
            file_write_enabled: Whether to include file write tool.

        Returns:
            List of tool instances (Node objects).
        """
        # Lazy import to avoid circular dependency
        from dynamiq.sandbox.tools.shell import SandboxShellTool

        shell_tool = SandboxShellTool(
            name="shell",
            sandbox=self,
        )
        return [shell_tool]


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
