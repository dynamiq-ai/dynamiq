import shlex
import threading
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import Cloudflare
from dynamiq.connections.cloudflare_sandbox import (
    SANDBOX_MARKER_PATH,
    CloudflareRateLimitError,
    CloudflareSandboxInstance,
    resolve_workspace_path,
)
from dynamiq.nodes import Node
from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger


class CloudflareSandbox(Sandbox):
    """Cloudflare Sandboxes implementation.

    Runs commands and stores files in a Cloudflare sandbox reached through a
    deployed sandbox bridge Worker (plain HTTP, no SDK). The container image and
    instance type are fixed by the deployed Worker's wrangler configuration, so
    unlike E2B/Daytona there are no template/image fields here. All file paths
    are constrained to /workspace by the bridge.

    Supports reconnecting to existing sandboxes by providing sandbox_id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: Cloudflare
    base_path: str = Field(default="/workspace", description="Base path in the sandbox filesystem.")
    timeout: int = Field(
        default=3600,
        description="Timeout budget in seconds for sandbox boot during creation and reconnection.",
    )

    envs: dict[str, str] | None = Field(
        default=None,
        description="Environment variables applied to every command executed in the sandbox.",
    )
    sandbox_id: str | None = Field(
        default=None,
        description="Existing sandbox ID to reconnect to. If None, a new sandbox is created.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for sandbox creation and reconnection.",
    )
    _sandbox: CloudflareSandboxInstance | None = PrivateAttr(default=None)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("base_path")
    @classmethod
    def validate_base_path(cls, value: str) -> str:
        """The bridge constrains all file operations to /workspace; fail fast otherwise."""
        return resolve_workspace_path(value)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude sensitive fields from serialization."""
        return super().to_dict_exclude_params | {"envs": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the CloudflareSandbox instance to a dictionary."""
        for_tracing = kwargs.get("for_tracing", False)
        data = super().to_dict(**kwargs)
        if not for_tracing and self.envs is not None:
            data["envs"] = self.envs
        return data

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current sandbox ID (for saving/reconnecting later)."""
        return self.sandbox_id

    def _ensure_sandbox(self) -> CloudflareSandboxInstance:
        """Lazily create or reconnect to a Cloudflare sandbox, with retries on capacity errors.

        Uses double-checked locking so concurrent threads never create duplicate sandboxes.
        """
        if self._sandbox is not None:
            return self._sandbox

        with self._sandbox_lock:
            if self._sandbox is not None:
                return self._sandbox

            client = self.connection.get_client()

            if self.sandbox_id:
                try:
                    self._reconnect_with_retry()
                    self._sandbox = CloudflareSandboxInstance(client, self.sandbox_id)
                    logger.debug(f"Cloudflare sandbox reconnected: {self.sandbox_id}")
                    return self._sandbox
                except Exception as e:
                    raise SandboxConnectionError(self.sandbox_id, cause=e, provider="Cloudflare") from e

            self.sandbox_id = self._create_with_retry()
            self._sandbox = CloudflareSandboxInstance(client, self.sandbox_id)
            logger.debug(f"Cloudflare sandbox created: {self.sandbox_id}")
            return self._sandbox

    def _reconnect_with_retry(self) -> None:
        """Verify an existing sandbox is alive and kept its state, with backoff on capacity errors.

        Sandbox ids are Durable-Object addressed, so any well-formed id "exists" and
        would silently get a fresh empty container; the creation marker distinguishes
        a live sandbox from an expired one.
        """
        cfg = self.creation_error_handling
        client = self.connection.get_client()

        @retry(
            retry=retry_if_exception_type(CloudflareRateLimitError),
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
            logger.debug(f"Reconnecting to Cloudflare sandbox: {self.sandbox_id}")
            result = client.exec(self.sandbox_id, f"test -f {shlex.quote(SANDBOX_MARKER_PATH)}", timeout=self.timeout)
            if result.exit_code != 0:
                raise ValueError(f"Cloudflare sandbox {self.sandbox_id} has expired or was destroyed (state lost)")

        connect()

    def _create_with_retry(self) -> str:
        """Create a new sandbox and boot its container, with backoff on capacity errors.

        POST /sandbox only mints an id - the container is allocated lazily on the
        first operation, so capacity errors surface there. Running the warm-up
        command inside the retry loop makes CloudflareRateLimitError retryable and
        leaves the creation marker plus the base directory in place.
        """
        cfg = self.creation_error_handling
        client = self.connection.get_client()

        @retry(
            retry=retry_if_exception_type(CloudflareRateLimitError),
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
                sandbox_id = client.create_sandbox()
                marker_dir = SANDBOX_MARKER_PATH.rsplit("/", 1)[0]
                result = client.exec(
                    sandbox_id,
                    f"mkdir -p {shlex.quote(self.base_path)} {shlex.quote(marker_dir)} "
                    f"&& touch {shlex.quote(SANDBOX_MARKER_PATH)}",
                    timeout=self.timeout,
                )
                if result.exit_code != 0:
                    raise ValueError(f"Failed to initialize Cloudflare sandbox: {result.stderr or result.stdout}")
                return sandbox_id
            except CloudflareRateLimitError:
                logger.warning("Cloudflare sandbox creation rate-limited. Retrying with exponential backoff.")
                raise

        return create()

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the Cloudflare sandbox."""
        sandbox = self._ensure_sandbox()
        logger.debug(f"CloudflareSandbox running command: {command[:100]}...")

        try:
            if run_in_background_enabled:
                sandbox.exec(f"nohup {command} > /dev/null 2>&1 &", cwd=self.base_path, env=self.envs)
                return ShellCommandResult(background=True)

            result = sandbox.exec(command, cwd=self.base_path, env=self.envs, timeout=timeout)
            if result.exit_code is None:
                return ShellCommandResult(
                    stdout=result.stdout,
                    stderr=result.stderr or None,
                    error="Command stream ended without an exit status (sandbox may have restarted)",
                )
            return ShellCommandResult(
                stdout=result.stdout,
                stderr=result.stderr or None,
                exit_code=result.exit_code,
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
        """Upload a file to the Cloudflare sandbox.

        Parent directories are created automatically by the sandbox file API,
        so ``ensure_parent_dirs`` needs no extra round-trip here.
        """
        sandbox = self._ensure_sandbox()

        if destination_path is None:
            destination_path = f"{self.base_path}/{file_name}"

        try:
            sandbox.write_file(destination_path, content)
            logger.debug(f"CloudflareSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the Cloudflare sandbox directory."""
        sandbox = self._ensure_sandbox()
        if target_dir is None:
            target_dir = self.base_path

        try:
            result = sandbox.exec(
                f"find {shlex.quote(target_dir)} -maxdepth 4 -type f 2>/dev/null "
                f"| head -{int(self.max_output_files)}"
            )
            if result.exit_code != 0:
                logger.warning(f"CloudflareSandbox list_files failed for {target_dir}: {result.stderr}")
                return []
            return [line for line in (result.stdout or "").splitlines() if line.strip()]
        except Exception as e:
            logger.warning(f"CloudflareSandbox list_files failed for {target_dir}: {e}")
            return []

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in sandbox filesystem."""
        try:
            sandbox = self._ensure_sandbox()
            resolved_path = self._resolve_path(file_path)
            result = sandbox.exec(f"test -f {shlex.quote(resolved_path)}")
            return result.exit_code == 0
        except Exception as e:
            logger.debug(f"CloudflareSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from sandbox filesystem."""
        sandbox = self._ensure_sandbox()
        resolved_path = self._resolve_path(file_path)
        return sandbox.read_file(resolved_path)

    def get_tools(self, llm: Any = None) -> list[Node]:
        """Return tools this sandbox provides for agent use."""
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

    def get_sandbox_info(self, port: int | None = None) -> SandboxInfo:
        """Return sandbox metadata including optional public tunnel URL for a port."""
        public_host: str | None = None
        public_url: str | None = None
        public_url_error: str | None = None
        if port is not None:
            try:
                self._ensure_sandbox()
                client = self.connection.get_client()
                tunnel = client.create_tunnel(self.sandbox_id, port)
                public_url = tunnel.get("url")
                public_host = tunnel.get("hostname")
                if public_host is None and public_url:
                    public_host = public_url.split("://", 1)[-1]
            except Exception as e:
                logger.debug("create_tunnel failed: %s", e)
                public_url_error = str(e)
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.sandbox_id,
            public_host=public_host,
            public_url=public_url,
            public_url_error=public_url_error,
        )

    def close(self, kill: bool = False) -> None:
        """Close the Cloudflare sandbox connection.

        Args:
            kill: If False (default), just disconnects but keeps the sandbox alive
                  for reconnection (the container sleeps on idle). If True,
                  destroys the sandbox.
        """
        if self._sandbox:
            try:
                if kill:
                    self._sandbox.destroy()
                    logger.debug(f"Cloudflare sandbox destroyed: {self.sandbox_id}")
                    self.sandbox_id = None
                else:
                    logger.debug(f"Cloudflare sandbox disconnected (kept alive): {self.sandbox_id}")
            except Exception as e:
                logger.warning(f"CloudflareSandbox close() failed: {e}")
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
            ...
