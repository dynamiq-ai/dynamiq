"""E2B sandbox implementation."""

import shlex
from typing import Any

from e2b.exceptions import RateLimitException as E2BRateLimitException
from e2b_desktop import Sandbox as E2BDesktopSandbox
from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import E2B
from dynamiq.nodes import Node
from dynamiq.nodes.tools.e2b_sandbox import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger


class E2BSandbox(Sandbox):
    """E2B sandbox implementation.

    This implementation stores files in E2B remote sandbox filesystem.
    Files persist for the lifetime of the sandbox session.

    Supports reconnecting to existing sandboxes by providing sandbox_id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: E2B
    timeout: int = 3600

    template: str | None = Field(
        default=None, description="Template to use for sandbox creation. " "If None, the default template is used."
    )
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
                self._ensure_output_dir()
                return self._sandbox
            except Exception as e:
                raise SandboxConnectionError(self.sandbox_id, cause=e) from e

        # Create new sandbox (no sandbox_id)
        self._sandbox = self._create_with_retry()
        self.sandbox_id = self._sandbox.sandbox_id
        logger.debug(f"E2B sandbox created: {self.sandbox_id}")
        self._ensure_output_dir()
        return self._sandbox

    def _ensure_output_dir(self) -> None:
        """Create the output directory inside the sandbox if it does not exist."""
        if self._sandbox is None:
            return
        try:
            self._sandbox.commands.run(f"mkdir -p {shlex.quote(self.output_dir)}")
            logger.debug(f"E2BSandbox ensured output dir exists: {self.output_dir}")
        except Exception as e:
            logger.warning(f"E2BSandbox failed to create output dir {self.output_dir}: {e}")

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
                    template=self.template,
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
            sandbox.files.write(destination_path, content)
            logger.debug(f"E2BSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_output_files(self) -> list[str]:
        """List files in the E2B sandbox output directory.

        Searches for files in the output directory (``{base_path}/output``),
        returning at most ``max_output_files`` file paths.

        Returns:
            List of absolute file paths found in the output directory.
        """
        sandbox = self._ensure_sandbox()
        target_dir = self.output_dir

        try:
            # Check if the directory exists
            check_cmd = f"test -d {shlex.quote(target_dir)} && echo exists"
            check_result = sandbox.commands.run(check_cmd)
            if check_result.exit_code != 0 or "exists" not in (check_result.stdout or ""):
                return []

            # List files recursively (configurable limit)
            cmd = f"find {shlex.quote(target_dir)} -type f 2>/dev/null | head -{self.max_output_files}"
            result = sandbox.commands.run(cmd)
            if result.exit_code != 0 or not (result.stdout or "").strip():
                return []

            return [f.strip() for f in result.stdout.splitlines() if f.strip()]

        except Exception as e:
            logger.warning(f"E2BSandbox list_files failed for {target_dir}: {e}")
            return []

    def _resolve_path(self, file_path: str) -> str:
        """Resolve relative file paths against sandbox base path."""
        if file_path.startswith("/"):
            return file_path
        return f"{self.base_path.rstrip('/')}/{file_path.lstrip('/')}"

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in sandbox filesystem."""
        sandbox = self._ensure_sandbox()
        resolved_path = self._resolve_path(file_path)
        check_cmd = f"test -f {shlex.quote(resolved_path)}"
        result = sandbox.commands.run(check_cmd)
        return getattr(result, "exit_code", 1) == 0

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from sandbox filesystem."""
        sandbox = self._ensure_sandbox()
        resolved_path = self._resolve_path(file_path)
        return sandbox.files.read(resolved_path, "bytes")

    def get_tools(self, llm: Any = None) -> list[Node]:
        """Return tools this sandbox provides for agent use.

        Creates tools based on tools config and TOOL_REGISTRY.
        Each tool is enabled by default unless explicitly disabled.

        Args:
            llm: Optional LLM instance passed to tools that require one (e.g. FileReadTool).

        Returns:
            List of tool instances (Node objects).
        """
        from dynamiq.nodes.tools.file_tools import FileReadTool
        from dynamiq.sandboxes.tools.shell import SandboxShellTool

        if llm is not None:
            return [SandboxShellTool(sandbox=self), FileReadTool(file_store=self, llm=llm)]
        else:
            return [SandboxShellTool(sandbox=self)]

    def close(self, kill: bool = False) -> None:
        """Close the E2B sandbox connection.

        Args:
            kill: If False (default), just disconnects
                  but keeps the sandbox alive for reconnection using sandbox_id. If True, kills the sandbox.
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
