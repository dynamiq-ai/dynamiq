"""E2B sandbox implementation."""

import shlex
import threading
from typing import Any, ClassVar

from e2b.exceptions import RateLimitException as E2BRateLimitException
from e2b.sandbox.commands.command_handle import CommandExitException
from e2b_code_interpreter import Sandbox as E2BCodeInterpreterSandbox
from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import E2B
from dynamiq.nodes import Node
from dynamiq.nodes.tools.e2b_sandbox import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger


class E2BSandbox(Sandbox):
    """E2B sandbox implementation.

    This implementation stores files in E2B remote sandbox filesystem.
    Files persist for the lifetime of the sandbox session.

    Supports reconnecting to existing sandboxes by providing sandbox_id.
    """

    DEFAULT_E2B_DOMAIN: ClassVar[str] = "e2b.app"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: E2B
    timeout: int = 3600

    template: str | None = Field(
        default=None, description="Template to use for sandbox creation. " "If None, the default template is used."
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Custom environment variables passed to the sandbox on creation.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Custom metadata attached to the sandbox on creation.",
    )
    sandbox_id: str | None = Field(
        default=None,
        description="Existing sandbox ID to reconnect to. If None, a new sandbox is created.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for sandbox creation and reconnection (rate-limit and transient errors).",
    )
    _sandbox: E2BCodeInterpreterSandbox | None = PrivateAttr(default=None)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude sensitive fields from model_dump; re-added manually in to_dict when not tracing."""
        return super().to_dict_exclude_params | {"envs": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the E2BSandbox instance to a dictionary.

        Extends the base serialization to conditionally include the ``envs``
        field.  When ``for_tracing`` is True the field is omitted to prevent
        secrets from leaking into tracing/callback output.

        Args:
            **kwargs: Keyword arguments forwarded to ``Sandbox.to_dict``.
                Accepts ``for_tracing`` (bool, default False) to control
                whether sensitive fields are included.

        Returns:
            dict[str, Any]: Dictionary representation of the sandbox.
        """
        for_tracing = kwargs.get("for_tracing", False)
        data = super().to_dict(**kwargs)
        if not for_tracing and self.envs is not None:
            data["envs"] = self.envs
        return data

    @property
    def _sdk_class(self) -> type:
        """Return the E2B SDK Sandbox class used for .create() and .connect().

        Subclasses override this to swap the underlying SDK (e.g. e2b_desktop).
        """
        return E2BCodeInterpreterSandbox

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current sandbox ID (for saving/reconnecting later)."""
        return self.sandbox_id

    def get_public_host(self, port: int) -> str:
        """Return the public host for a given port so the sandbox can be reached at https://{host}.

        E2B exposes each sandbox at a URL like https://{port}-{sandbox_id}.e2b.app.
        Use this when the agent starts a server in the sandbox (e.g. dev server on port 5173)
        so it can report the URL to the user.

        Args:
            port: Port number the service listens on inside the sandbox (e.g. 3000, 5173).

        Returns:
            Host string (e.g. 3000-abc123.e2b.app). Full URL is https://{host}.
        """
        self._ensure_sandbox()
        raw = self._sandbox
        get_host = getattr(raw, "get_host", None)
        if callable(get_host):
            try:
                return get_host(port)
            except Exception as e:
                logger.debug("E2B get_host(port) failed, using URL pattern: %s", e)
        domain = getattr(self.connection, "domain", None) or self.DEFAULT_E2B_DOMAIN
        return f"{port}-{self.sandbox_id}.{domain}"

    def apply_public_preview_branding(
        self, public_host: str | None, public_url: str | None
    ) -> tuple[str | None, str | None]:
        """Rewrite E2B public preview URLs to a custom branded domain.

        When public preview domain is configured, hosts that end with
        the active E2B domain are rewritten to the configured suffix
        while preserving the host prefix.

        Args:
            public_host: Original public host returned by E2B.
            public_url: Original public URL associated with public_host.

        Returns:
            Tuple of (public_host, public_url). Returns rewritten values only
            when public preview is configured and the host matches the E2B domain,
            otherwise returns the input values unchanged.
        """
        suffix = (self.connection.public_preview_domain or "").strip().lstrip(".")
        domain = getattr(self.connection, "domain", None) or self.DEFAULT_E2B_DOMAIN
        tail = f".{domain}"
        if not suffix or not public_host or not public_host.endswith(tail):
            return public_host, public_url
        host = f"{public_host.removesuffix(tail)}.{suffix}"
        return host, f"https://{host}"

    def get_sandbox_info(self, port: int | None = None) -> SandboxInfo:
        """Return sandbox metadata including optional public URL for a port."""
        public_host: str | None = None
        public_url: str | None = None
        public_url_error: str | None = None
        if port is not None:
            try:
                public_host = self.get_public_host(port)  # may trigger _ensure_sandbox() and set self.sandbox_id
                public_url = f"https://{public_host}"
            except Exception as e:
                logger.debug("get_public_host failed: %s", e)
                public_url_error = str(e)
        public_host, public_url = self.apply_public_preview_branding(public_host, public_url)
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.sandbox_id,
            public_host=public_host,
            public_url=public_url,
            public_url_error=public_url_error,
        )

    def _ensure_sandbox(self) -> E2BCodeInterpreterSandbox:
        """Lazily initialize or reconnect to E2B sandbox, with retries on rate-limit and transient errors.

        Uses double-checked locking so concurrent threads (e.g. parallel tool
        calls via ThreadPoolExecutor) never create duplicate sandboxes.
        """
        if self._sandbox is not None:
            return self._sandbox

        with self._sandbox_lock:
            # Double-check after acquiring the lock
            if self._sandbox is not None:
                return self._sandbox

            if self.sandbox_id:
                try:
                    self._sandbox = self._reconnect_with_retry()
                    logger.debug(f"E2B sandbox reconnected: {self.sandbox_id}")
                    # Call set_timeout explicitly to ensure the sandbox timeout
                    # is set to the proper value after reconnect.
                    try:
                        self._sandbox.set_timeout(self.timeout)
                    except Exception as e:
                        logger.debug("set_timeout after reconnect failed: %s", e)
                    self._ensure_directories()
                    return self._sandbox
                except Exception as e:
                    raise SandboxConnectionError(self.sandbox_id, cause=e) from e

            # Create new sandbox (no sandbox_id)
            self._sandbox = self._create_with_retry()
            self.sandbox_id = self._sandbox.sandbox_id
            logger.debug(f"E2B sandbox created: {self.sandbox_id}")
            self._ensure_directories()
            return self._sandbox

    def _ensure_directories(self) -> None:
        """Create the base directory inside the sandbox if it does not exist."""
        if self._sandbox is None:
            return
        try:
            self._sandbox.commands.run(f"mkdir -p {shlex.quote(self.base_path)}")
            logger.debug(f"E2BSandbox ensured directory exists: {self.base_path}")
        except Exception as e:
            logger.warning(f"E2BSandbox failed to create directory: {e}")

    def _reconnect_with_retry(self) -> E2BCodeInterpreterSandbox:
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
        def connect():
            logger.debug(f"Reconnecting to E2B sandbox: {self.sandbox_id}")
            # Always pass timeout explicitly to avoid resetting
            # the sandbox timeout to the 5-minute default
            # if connect() is called without a timeout value.
            return self._sdk_class.connect(
                sandbox_id=self.sandbox_id,
                api_key=self.connection.api_key,
                domain=getattr(self.connection, "domain", None),
                timeout=self.timeout,
            )

        return connect()

    def _create_with_retry(self) -> E2BCodeInterpreterSandbox:
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
        def create():
            try:
                from datetime import datetime, timezone

                metadata = self.metadata.copy() if self.metadata else {}
                metadata.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                return self._sdk_class.create(
                    template=self.template,
                    api_key=self.connection.api_key,
                    timeout=self.timeout,
                    domain=getattr(self.connection, "domain", None),
                    envs=self.envs or {},
                    metadata=metadata,
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
                return ShellCommandResult(background=True)

            result = sandbox.commands.run(command, timeout=timeout)
            return ShellCommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
            )
        except CommandExitException as e:
            logger.debug(f"Command exited with non-zero code: {e.exit_code}, stderr: {e.stderr}")
            return ShellCommandResult(
                stdout=e.stdout,
                stderr=e.stderr,
                exit_code=e.exit_code,
            )
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return ShellCommandResult(error=str(e))

    def upload_file(
        self,
        file_name: str,
        content: bytes,
        destination_path: str | None = None,
        ensure_parent_dirs: bool = True,
    ) -> str:
        """Upload a file to the E2B sandbox.

        When ensure_parent_dirs is True, parent directories are created with ``mkdir -p``
        before writing so that existing directories (e.g. from a previous skill ingestion)
        do not cause 500 errors from backends that use non-idempotent mkdir.

        Args:
            file_name: Name of the file.
            content: File content as bytes.
            destination_path: Optional destination path in sandbox. If None, uses base_path/file_name.
            ensure_parent_dirs: When True, run mkdir -p for the destination's parent before write.

        Returns:
            The path where the file was uploaded in the sandbox.
        """
        sandbox = self._ensure_sandbox()

        if destination_path is None:
            destination_path = f"{self.base_path}/{file_name}"

        if ensure_parent_dirs and destination_path:
            parent = destination_path.rsplit("/", 1)[0]
            if parent and parent != destination_path:
                try:
                    sandbox.commands.run(f"mkdir -p {shlex.quote(parent)}")
                    logger.debug(f"E2BSandbox ensured parent dir: {parent}")
                except Exception as e:
                    logger.warning(f"E2BSandbox mkdir -p for {parent!r} failed (continuing): {e}")

        try:
            sandbox.files.write(destination_path, content)
            logger.debug(f"E2BSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the E2B sandbox directory.

        Searches for files in the given directory (defaults to output directory),
        returning at most ``max_output_files`` file paths.

        Returns:
            List of absolute file paths found in the directory.
        """
        sandbox = self._ensure_sandbox()
        if target_dir is None:
            target_dir = self.base_path

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

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in sandbox filesystem."""
        try:
            sandbox = self._ensure_sandbox()
            resolved_path = self._resolve_path(file_path)
            check_cmd = f"test -f {shlex.quote(resolved_path)}"
            result = sandbox.commands.run(check_cmd)
            return getattr(result, "exit_code", 1) == 0
        except Exception as e:
            logger.debug(f"E2BSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

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
        from dynamiq.nodes.tools.file_tools import FileReadTool, FileWriteTool
        from dynamiq.nodes.tools.todo_tools import TodoWriteTool
        from dynamiq.sandboxes.tools.sandbox_info import SandboxInfoTool
        from dynamiq.sandboxes.tools.shell import SandboxShellTool

        tools = [
            SandboxShellTool(sandbox=self),
            FileWriteTool(file_store=self, absolute_file_paths_allowed=True),
            TodoWriteTool(file_store=self),
            SandboxInfoTool(sandbox=self),
        ]
        if llm is not None:
            tools.append(FileReadTool(file_store=self, llm=llm, absolute_file_paths_allowed=True))

        return tools

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
