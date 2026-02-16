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
        blocked_commands: Optional list of blocked substrings; command is blocked if it contains any (case-insensitive).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SandboxShellTool"
    description: str = (
        "Execute shell commands in an isolated sandbox environment.\n\n"
        "Parameters:\n"
        "- command (str, required): The shell command to execute.\n"
        "- timeout (int, default 60): Max seconds to wait for completion.\n"
        "- run_in_background_enabled (bool, default false): Run without waiting for output.\n\n"
        "Examples:\n"
        '- {"command": "ls -la"}\n'
        '- {"command": "echo Hello World"}\n'
        '- {"command": "pip install pandas", "timeout": 120}\n'
        '- {"command": "cp result.csv /home/user/workspace/output/"}\n'
        '- {"command": "cat <<\'EOF\' > script.py && python3 script.py\\nimport csv\\n'
        "with open('data.csv') as f:\\n"
        "    reader = csv.reader(f)\\n"
        "    print(list(reader))\\n"
        "print('Done')\\n"
        'EOF"}'
    )

    sandbox: Sandbox = Field(..., description="Sandbox backend to execute commands in.")
    blocked_commands: list[str] | None = Field(
        default=None,
        description="Optional list of blocked substrings. A command is blocked if"
        " it contains any of these strings (case-insensitive).",
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
        """Validate command against blocked list. Blocked entries are matched as substrings
        (case-insensitive): if the command contains any blocked string, it is rejected."""
        cmd_lower = command.strip().lower()

        # Check blocked commands
        if self.blocked_commands:
            for blocked in self.blocked_commands:
                if blocked.lower() in cmd_lower:
                    raise ToolExecutionException(
                        f"Command '{command}' is blocked for security reasons.",
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
            is_success = result.exit_code == 0 or (result.exit_code is None and not result.stderr)
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
