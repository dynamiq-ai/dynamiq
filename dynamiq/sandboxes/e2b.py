"""E2B sandbox implementation."""

from typing import ClassVar

from e2b.exceptions import RateLimitException as E2BRateLimitException
from e2b_desktop import Sandbox as E2BDesktopSandbox
from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import E2B
from dynamiq.nodes import Node
from dynamiq.nodes.tools.e2b_sandbox import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import FileEntry, Sandbox, SandboxTool, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger

# Default timeout (in seconds) for E2B file operations (read, write, list, upload)
E2B_FILE_OPERATION_TIMEOUT = 60


class E2BSandbox(Sandbox):
    """E2B sandbox implementation.

    This implementation stores files in E2B remote sandbox filesystem.
    Files persist for the lifetime of the sandbox session.

    Supports reconnecting to existing sandboxes by providing sandbox_id.
    """

    # Registry mapping tool types to (tool_class_path, config_keys)
    # Each sandbox implementation can define its own supported tools
    TOOL_REGISTRY: ClassVar[dict[SandboxTool, tuple[str, list[str]]]] = {
        SandboxTool.SHELL: (
            "dynamiq.sandboxes.tools.shell.SandboxShellTool",
            ["blocked_commands"],
        ),
        SandboxTool.FILES: (
            "dynamiq.sandboxes.tools.files.SandboxFilesTool",
            ["blocked_paths"],
        ),
    }

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: E2B
    timeout: int = 3600
    base_path: str = "/home/user"
    sandbox_id: str | None = Field(
        default=None,
        description="Existing sandbox ID to reconnect to. If None, a new sandbox is created.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for sandbox creation and reconnection (rate-limit and transient errors).",
    )
    _sandbox: E2BDesktopSandbox | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """Initialize the E2B sandbox storage."""
        super().__init__(**kwargs)

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current sandbox ID (for saving/reconnecting later)."""
        return self.sandbox_id

    def _ensure_sandbox(self) -> E2BDesktopSandbox:
        """Lazily initialize or reconnect to E2B sandbox, with retries on rate-limit and transient errors."""
        if self._sandbox is not None:
            return self._sandbox

        if self.sandbox_id:
            try:
                self._sandbox = self._reconnect_with_retry()
                logger.debug(f"E2B sandbox reconnected: {self.sandbox_id}")
                return self._sandbox
            except Exception as e:
                raise SandboxConnectionError(self.sandbox_id, cause=e) from e

        # Create new sandbox (no sandbox_id)
        self._sandbox = self._create_with_retry()
        self.sandbox_id = self._sandbox.sandbox_id
        logger.debug(f"E2B sandbox created: {self.sandbox_id}")
        return self._sandbox

    def _reconnect_with_retry(self) -> E2BDesktopSandbox:
        """Reconnect to existing sandbox with exponential backoff on rate-limit."""
        cfg = self.creation_error_handling

        @retry(
            retry=retry_if_exception_type(E2BRateLimitException),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def connect() -> E2BDesktopSandbox:
            logger.debug(f"Reconnecting to E2B sandbox: {self.sandbox_id}")
            return E2BDesktopSandbox.connect(
                sandbox_id=self.sandbox_id,
                api_key=self.connection.api_key,
                domain=getattr(self.connection, "domain", None),
            )

        return connect()

    def _create_with_retry(self) -> E2BDesktopSandbox:
        """Create a new sandbox with exponential backoff on rate-limit."""
        cfg = self.creation_error_handling

        @retry(
            retry=retry_if_exception_type(E2BRateLimitException),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def create() -> E2BDesktopSandbox:
            try:
                return E2BDesktopSandbox.create(
                    api_key=self.connection.api_key,
                    timeout=self.timeout,
                    domain=getattr(self.connection, "domain", None),
                )
            except E2BRateLimitException:
                logger.warning("E2B sandbox creation rate-limited. Retrying with exponential backoff.")
                raise

        return create()

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

    def read_file(self, path: str) -> bytes:
        """Read a file from the E2B sandbox.

        Args:
            path: Path of the file in the sandbox.

        Returns:
            File content as bytes.
        """
        sandbox = self._ensure_sandbox()
        try:
            content = sandbox.files.read(path, request_timeout=E2B_FILE_OPERATION_TIMEOUT)
            if isinstance(content, str):
                content = content.encode("utf-8")
            logger.debug(f"E2BSandbox read file: {path} ({len(content)} bytes)")
            return content
        except Exception as e:
            logger.error(f"Failed to read file {path}: {e}")
            raise

    def write_file(self, path: str, content: bytes) -> str:
        """Write a file to the E2B sandbox.

        Args:
            path: Destination path in the sandbox.
            content: File content as bytes.

        Returns:
            The path where the file was written.
        """
        sandbox = self._ensure_sandbox()
        try:
            sandbox.files.write(path, content, request_timeout=E2B_FILE_OPERATION_TIMEOUT)
            logger.debug(f"E2BSandbox wrote file: {path} ({len(content)} bytes)")
            return path
        except Exception as e:
            logger.error(f"Failed to write file {path}: {e}")
            raise

    def list_files(self, path: str) -> list[FileEntry]:
        """List files and directories at the given path in the E2B sandbox.

        Args:
            path: Directory path in the sandbox.

        Returns:
            List of FileEntry objects.
        """
        sandbox = self._ensure_sandbox()
        try:
            entries = sandbox.files.list(path, request_timeout=E2B_FILE_OPERATION_TIMEOUT)
            result = []
            for entry in entries:
                result.append(
                    FileEntry(
                        name=getattr(entry, "name", str(entry)),
                        path=f"{path.rstrip('/')}/{getattr(entry, 'name', str(entry))}",
                        is_dir=getattr(entry, "is_dir", False),
                        size=getattr(entry, "size", None),
                    )
                )
            logger.debug(f"E2BSandbox listed {len(result)} entries at: {path}")
            return result
        except Exception as e:
            logger.error(f"Failed to list files at {path}: {e}")
            raise

    def upload_file(self, file_name: str, content: bytes, destination_path: str | None = None) -> str:
        """Upload a file to the E2B sandbox.

        Args:
            file_name: Name of the file.
            content: File content as bytes.
            destination_path: Optional destination path in sandbox. If None, uses base_path/file_name.

        Returns:
            The path where the file was uploaded in the sandbox.
        """
        sandbox = self._ensure_sandbox()

        if destination_path is None:
            destination_path = f"{self.base_path}/{file_name}"

        try:
            sandbox.files.write(destination_path, content, request_timeout=E2B_FILE_OPERATION_TIMEOUT)
            logger.debug(f"E2BSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

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

    def close(self, kill: bool = True) -> None:
        """Close the E2B sandbox connection.

        Args:
            kill: If True (default), kills the sandbox. If False, just disconnects
                  but keeps the sandbox alive for reconnection using sandbox_id.
        """
        if self._sandbox:
            try:
                if kill:
                    self._sandbox.kill()
                    logger.debug(f"E2B sandbox killed: {self.sandbox_id}")
                    self.sandbox_id = None
                else:
                    logger.debug(f"E2B sandbox disconnected (kept alive): {self.sandbox_id}")
            except Exception as e:
                logger.warning(f"E2BSandbox close() failed: {e}")
            finally:
                self._sandbox = None

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
