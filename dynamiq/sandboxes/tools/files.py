"""Sandbox file tools for reading, writing, and listing files."""

import logging
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox

logger = logging.getLogger(__name__)


class SandboxFileAction(str, Enum):
    """Supported file actions."""

    READ = "read"
    WRITE = "write"
    LIST = "list"


class SandboxFilesInputSchema(BaseModel):
    """Input schema for SandboxFilesTool."""

    action: SandboxFileAction = Field(
        ...,
        description=(
            "File action to perform: "
            "'read' to read file content, "
            "'write' to write content to a file, "
            "'list' to list files in a directory."
        ),
    )
    path: str = Field(
        ...,
        description="File or directory path in the sandbox.",
    )
    content: str | None = Field(
        default=None,
        description="Content to write (required for 'write' action).",
    )


class SandboxFilesTool(Node):
    """A tool for reading, writing, and listing files in a sandbox environment.

    This tool delegates file operations to the sandbox backend,
    allowing the sandbox to determine how files are managed
    (e.g., locally, in a container, or in a remote E2B environment).

    Attributes:
        sandbox: The sandbox backend to perform file operations in.
        blocked_paths: Optional list of blocked path prefixes for security.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SandboxFilesTool"
    description: str = (
        "Use by default root path '/' for all actions."
        "Read, write, and list files in a sandbox environment.\n\n"
        "Use this tool to interact with the sandbox filesystem.\n"
        "The operations run in an isolated sandbox environment.\n\n"
        "Examples:\n"
        '- Read a file: {"action": "read", "path": "data.txt"}\n'
        '- Write a file: {"action": "write", "path": "output.txt", "content": "Hello World"}\n'
        '- List directory: {"action": "list", "path": "/"} (default is the root directory)\n\n'
        "Parameters:\n"
        "- action: 'read', 'write', or 'list'\n"
        "- path: File or directory path in the sandbox\n"
        "- content: Content to write (required for 'write' action)"
    )

    sandbox: Sandbox = Field(..., description="Sandbox backend to perform file operations in.")
    blocked_paths: list[str] | None = Field(
        default=None,
        description="Optional list of blocked path prefixes. Operations on these paths are never permitted.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SandboxFilesInputSchema]] = SandboxFilesInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"sandbox": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        data = super().to_dict(for_tracing=for_tracing, **kwargs)
        data["sandbox"] = self.sandbox.to_dict(for_tracing=for_tracing, **kwargs) if self.sandbox else None
        return data

    def _validate_path(self, path: str) -> None:
        """Validate path against blocked list."""
        if self.blocked_paths:
            for blocked in self.blocked_paths:
                if path.startswith(blocked):
                    raise ToolExecutionException(
                        f"Path '{path}' is blocked for security reasons.",
                        recoverable=True,
                    )

    def execute(
        self,
        input_data: SandboxFilesInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute a file operation in the sandbox.

        Args:
            input_data: Input containing the action, path, and optional content.
            config: Runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with operation results.
        """
        logger.info(f"Tool {self.name} - {self.id}: executing {input_data.action.value} on path: {input_data.path}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            self._validate_path(input_data.path)

            if input_data.action == SandboxFileAction.READ:
                return self._execute_read(input_data.path)
            elif input_data.action == SandboxFileAction.WRITE:
                return self._execute_write(input_data.path, input_data.content)
            elif input_data.action == SandboxFileAction.LIST:
                return self._execute_list(input_data.path)
            else:
                raise ToolExecutionException(
                    f"Unknown action: {input_data.action}",
                    recoverable=True,
                )

        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: file operation failed: {e}")
            raise ToolExecutionException(
                f"Failed to execute file operation: {e}",
                recoverable=True,
            )

    def _execute_read(self, path: str) -> dict[str, Any]:
        """Read file content from sandbox."""
        content_bytes = self.sandbox.read_file(path)
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = f"(binary file, {len(content_bytes)} bytes)"

        logger.info(f"Tool {self.name} - {self.id}: read {len(content_bytes)} bytes from {path}")
        return {
            "content": content,
            "path": path,
            "size": len(content_bytes),
            "action": "read",
        }

    def _execute_write(self, path: str, content: str | None) -> dict[str, Any]:
        """Write content to a file in sandbox."""
        if content is None:
            raise ToolExecutionException(
                "Content is required for 'write' action.",
                recoverable=True,
            )

        content_bytes = content.encode("utf-8")
        written_path = self.sandbox.write_file(path, content_bytes)

        logger.info(f"Tool {self.name} - {self.id}: wrote {len(content_bytes)} bytes to {written_path}")
        return {
            "content": f"File written successfully to {written_path} ({len(content_bytes)} bytes)",
            "path": written_path,
            "size": len(content_bytes),
            "action": "write",
        }

    def _execute_list(self, path: str) -> dict[str, Any]:
        """List files in a sandbox directory."""
        entries = self.sandbox.list_files(path)

        entries_output = []
        for entry in entries:
            entry_type = "dir" if entry.is_dir else "file"
            size_str = f", {entry.size} bytes" if entry.size is not None else ""
            entries_output.append(f"[{entry_type}] {entry.name} ({entry.path}{size_str})")

        listing = "\n".join(entries_output) if entries_output else "(empty directory)"

        logger.info(f"Tool {self.name} - {self.id}: listed {len(entries)} entries at {path}")
        return {
            "content": f"Files at {path}:\n{listing}",
            "path": path,
            "entries": [entry.model_dump() for entry in entries],
            "count": len(entries),
            "action": "list",
        }
