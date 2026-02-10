"""Sandbox file tools for reading, writing, and listing files."""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox

logger = logging.getLogger(__name__)

# Path where the helper script is uploaded in the sandbox
_HELPER_SCRIPT_PATH = "/home/user/.dynamiq/sys_tools/_file_helper.py"

# Load the helper script content from the adjacent file
_HELPER_SCRIPT = (Path(__file__).parent / "_file_helper.py").read_text()


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

    This tool uploads a Python helper script to the sandbox on first use,
    then delegates file operations to it via shell commands. This avoids
    direct use of sandbox file APIs and handles content safely via stdin.

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
    _helper_uploaded: bool = False

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"sandbox": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        for_tracing = kwargs.pop("for_tracing", False)
        data = super().to_dict(for_tracing=for_tracing, **kwargs)
        data["sandbox"] = self.sandbox.to_dict(for_tracing=for_tracing, **kwargs) if self.sandbox else None
        return data

    def _ensure_helper(self) -> None:
        """Upload the Python helper script to the sandbox if not already done."""
        if self._helper_uploaded:
            return

        # Create directory and upload helper via heredoc (avoids file API)
        self.sandbox.run_command_shell(f"mkdir -p $(dirname {_HELPER_SCRIPT_PATH})")
        command = f"cat << 'DYNAMIQ_HELPER_EOF' > {_HELPER_SCRIPT_PATH}\n{_HELPER_SCRIPT}DYNAMIQ_HELPER_EOF"
        result = self.sandbox.run_command_shell(command)
        if result.exit_code and result.exit_code != 0:
            raise ToolExecutionException(
                f"Failed to upload file helper script: {result.stderr}",
                recoverable=False,
            )
        # Make the helper script read-only to prevent agent from modifying it
        self.sandbox.run_command_shell(f"chmod 444 {_HELPER_SCRIPT_PATH}")
        self._helper_uploaded = True
        logger.debug(f"Tool {self.name} - {self.id}: helper script uploaded to {_HELPER_SCRIPT_PATH}")

    def _run_helper(self, command: str) -> dict[str, Any]:
        """Run the helper script and parse JSON output."""
        self._ensure_helper()
        result = self.sandbox.run_command_shell(command)

        if result.exit_code and result.exit_code != 0:
            raise ToolExecutionException(
                f"Helper script failed: {result.stderr}",
                recoverable=True,
            )

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            raise ToolExecutionException(
                f"Invalid helper output: {result.stdout[:200]}",
                recoverable=True,
            )

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
        """Read file content from sandbox via helper script."""
        data = self._run_helper(f"python3 {_HELPER_SCRIPT_PATH} read {path}")

        if not data.get("ok"):
            raise ToolExecutionException(
                f"Failed to read file '{path}': {data.get('error', 'unknown error')}",
                recoverable=True,
            )

        logger.info(f"Tool {self.name} - {self.id}: read {data['size']} chars from {path}")
        return {
            "content": data["content"],
            "path": path,
            "size": data["size"],
            "action": "read",
        }

    def _execute_write(self, path: str, content: str | None) -> dict[str, Any]:
        """Write content to a file in sandbox via helper script.

        Content is passed via stdin to avoid shell escaping issues.
        """
        if content is None:
            raise ToolExecutionException(
                "Content is required for 'write' action.",
                recoverable=True,
            )

        # Pipe content via heredoc to stdin of the helper script
        command = f"cat << 'DYNAMIQ_EOF' | python3 {_HELPER_SCRIPT_PATH} write {path}\n" f"{content}\n" f"DYNAMIQ_EOF"
        data = self._run_helper(command)

        if not data.get("ok"):
            raise ToolExecutionException(
                f"Failed to write file '{path}': {data.get('error', 'unknown error')}",
                recoverable=True,
            )

        logger.info(f"Tool {self.name} - {self.id}: wrote {data['size']} chars to {path}")
        return {
            "content": f"File written successfully to {path} ({data['size']} bytes)",
            "path": path,
            "size": data["size"],
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
