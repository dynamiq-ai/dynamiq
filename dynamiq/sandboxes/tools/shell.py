"""Sandbox shell tools for command execution and file operations."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import Sandbox

logger = logging.getLogger(__name__)


class SandboxShellInputSchema(BaseModel):
    """Input schema for SandboxShellTool."""

    command: str = Field(..., description="Shell command to execute in the sandbox.")
    timeout: int = Field(
        default=60,
        description="Timeout in seconds for command execution.",
    )
    run_in_background_enabled: bool = Field(
        default=False,
        description="If True, run the command in background without waiting for output.",
    )


class SandboxShellTool(Node):
    """A tool for executing shell commands in a sandbox environment.

    This tool delegates command execution to the sandbox backend,
    allowing the sandbox to determine how commands are executed
    (e.g., locally, in a container, or in a remote E2B environment).

    Attributes:
        sandbox: The sandbox backend to execute commands in.
        allowed_commands: Optional list of allowed command prefixes for security.
        blocked_commands: Optional list of blocked command prefixes for security.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SandboxShellTool"
    description: str = (
        "Execute shell commands in a sandbox environment.\n\n"
        "Use this tool to run shell commands, scripts, or system utilities.\n"
        "The command runs in an isolated sandbox environment.\n\n"
        "Examples:\n"
        '- {"command": "ls -la"}\n'
        '- {"command": "python script.py"}\n'
        '- {"command": "echo Hello World"}\n'
        '- {"command": "pip install pandas", "timeout": 120}\n\n'
        "Parameters:\n"
        "- command: The shell command to execute\n"
        "- timeout: Maximum time to wait for command completion (default: 60s)\n"
        "- run_in_background_enabled: Run command in background without waiting (default: false)"
    )

    sandbox: Sandbox = Field(..., description="Sandbox backend to execute commands in.")
    allowed_commands: list[str] | None = Field(
        default=None,
        description="Optional list of allowed command prefixes. If set, only these commands are permitted.",
    )
    blocked_commands: list[str] | None = Field(
        default=None,
        description="Optional list of blocked command prefixes. These commands are never permitted.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SandboxShellInputSchema]] = SandboxShellInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"sandbox": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        # Pop for_tracing to avoid passing it twice (explicitly and in **kwargs).
        for_tracing = kwargs.pop("for_tracing", False)
        data = super().to_dict(for_tracing=for_tracing, **kwargs)
        data["sandbox"] = self.sandbox.to_dict(for_tracing=for_tracing, **kwargs) if self.sandbox else None
        return data

    def _validate_command(self, command: str) -> None:
        """Validate command against allowed/blocked lists."""
        cmd_lower = command.strip().lower()

        # Check blocked commands
        if self.blocked_commands:
            for blocked in self.blocked_commands:
                if cmd_lower.startswith(blocked.lower()):
                    raise ToolExecutionException(
                        f"Command '{command}' is blocked for security reasons.",
                        recoverable=True,
                    )

        # Check allowed commands
        if self.allowed_commands:
            is_allowed = any(cmd_lower.startswith(allowed.lower()) for allowed in self.allowed_commands)
            if not is_allowed:
                raise ToolExecutionException(
                    f"Command '{command}' is not in the allowed commands list.",
                    recoverable=True,
                )

    def execute(
        self,
        input_data: SandboxShellInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute a shell command in the sandbox.

        Args:
            input_data: Input containing the command and options.
            config: Runtime configuration.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with stdout, stderr, and exit_code from command execution.
        """
        logger.info(f"Tool {self.name} - {self.id}: executing command: {input_data.command[:100]}...")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            # Validate command against security rules
            self._validate_command(input_data.command)

            # Execute command via sandbox
            result = self.sandbox.run_command_shell(
                command=input_data.command,
                timeout=input_data.timeout,
                run_in_background_enabled=input_data.run_in_background_enabled,
            )

            # Handle None exit_code: treat as success unless stderr indicates error
            is_success = result.exit_code is None or result.exit_code == 0
            output = {
                "content": result.stdout if result.stdout else "(no output)",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "success": is_success,
            }

            if not is_success:
                output["content"] = (
                    f"Command failed with exit code {result.exit_code}.\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            logger.info(f"Tool {self.name} - {self.id}: command completed with exit code {result.exit_code}")
            return output

        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: command execution failed: {e}")
            raise ToolExecutionException(
                f"Failed to execute command: {e}",
                recoverable=True,
            )
