"""E2B sandbox implementation."""

from typing import Any, ClassVar

from e2b_desktop import Sandbox as E2BDesktopSandbox
from pydantic import ConfigDict, PrivateAttr

from dynamiq.connections import E2B
from dynamiq.nodes import Node
from dynamiq.sandboxes.base import Sandbox, SandboxTool, ShellCommandResult
from dynamiq.utils.logger import logger


class E2BSandbox(Sandbox):
    """E2B sandbox implementation.

    This implementation stores files in E2B remote sandbox filesystem.
    Files persist for the lifetime of the sandbox session.
    """

    # Registry mapping tool types to (tool_class_path, config_keys)
    # Each sandbox implementation can define its own supported tools
    TOOL_REGISTRY: ClassVar[dict[SandboxTool, tuple[str, list[str]]]] = {
        SandboxTool.SHELL: (
            "dynamiq.sandboxes.tools.shell.SandboxShellTool",
            ["allowed_commands", "blocked_commands"],
        ),
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: E2B
    timeout: int = 3600
    base_path: str = "/home/user"
    _sandbox: E2BDesktopSandbox | None = PrivateAttr(default=None)
    _sandbox_id: str | None = PrivateAttr(default=None)
    # Local cache for metadata (E2B doesn't store custom metadata)
    _file_metadata: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        """Initialize the E2B sandbox storage."""
        super().__init__(**kwargs)
        self._file_metadata = {}

    def _ensure_sandbox(self) -> "E2BDesktopSandbox":
        """Lazily initialize the E2B sandbox."""
        if self._sandbox is None:
            self._sandbox = E2BDesktopSandbox.create(
                api_key=self.connection.api_key,
                timeout=self.timeout,
                domain=getattr(self.connection, "domain", None),
            )
            self._sandbox_id = self._sandbox.sandbox_id
            logger.debug(f"E2B sandbox created: {self._sandbox_id}")
        return self._sandbox

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the E2B sandbox.

        Args:
            command: Shell command or script to execute.
            timeout: Timeout in seconds (default 60).
            background: If True, run command in background (no output).

        Returns:
            ShellCommandResult with stdout, stderr, and exit_code.
        """
        sandbox = self._ensure_sandbox()
        logger.debug(f"E2BSandbox running command: {command[:100]}...")

        try:
            if run_in_background_enabled:
                sandbox.commands.run(command, background=True)
                return ShellCommandResult(
                    stdout=f"Command started in background: {command}",
                    stderr="",
                    exit_code=0,
                )

            result = sandbox.commands.run(command, timeout=timeout)
            return ShellCommandResult(
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                exit_code=getattr(result, "exit_code", None),
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return ShellCommandResult(
                stdout="",
                stderr=str(e),
                exit_code=1,
            )

    def get_tools(self) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Creates tools based on tools config and TOOL_REGISTRY.
        Each tool is enabled by default unless explicitly disabled.

        Returns:
            List of tool instances (Node objects).
        """
        from dynamiq.nodes.managers import NodeManager

        result = []
        for tool_type, (tool_class_path, config_keys) in self.TOOL_REGISTRY.items():
            tool_config = self.tools.get(tool_type, {})
            # Tools are enabled by default unless explicitly disabled
            if not tool_config.get("enabled", True):
                continue

            # Build tool kwargs from config
            tool_kwargs = {"sandbox": self}
            for key in config_keys:
                if key in tool_config:
                    tool_kwargs[key] = tool_config[key]

            # Get tool class from NodeManager and instantiate
            tool_class = NodeManager.get_node_by_type(tool_class_path)
            result.append(tool_class(**tool_kwargs))

        return result

    def close(self) -> None:
        """Close and kill the E2B sandbox."""
        if self._sandbox:
            try:
                self._sandbox.kill()
                logger.debug(f"E2B sandbox killed: {self._sandbox_id}")
            except Exception as e:
                logger.warning(f"E2BSandbox close() failed: {e}")
            finally:
                self._sandbox = None
                self._sandbox_id = None
                self._file_metadata.clear()

    def __enter__(self):
        """Context manager entry."""
        self._ensure_sandbox()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - attempt to close sandbox on garbage collection."""
        try:
            self.close()
        except Exception:
            # Cannot reliably log in __del__, just suppress
            ...  # noqa: E701
