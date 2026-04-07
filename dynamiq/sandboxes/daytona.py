import shlex
import threading
from typing import Any

from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, DaytonaRateLimitError
from daytona import Sandbox as DaytonaSandboxInstance
from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import Daytona
from dynamiq.nodes import Node
from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger


class DaytonaSandbox(Sandbox):
    """Daytona sandbox implementation.

    This implementation stores files in a Daytona remote sandbox filesystem.
    Files persist for the lifetime of the sandbox session.

    Supports reconnecting to existing sandboxes by providing sandbox_id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: Daytona
    base_path: str = Field(default="/home/daytona", description="Base path in the sandbox filesystem.")
    timeout: int = 3600

    image: str | None = Field(
        default=None,
        description="Container image to use for sandbox creation. If None, the default image is used.",
    )
    snapshot: str | None = Field(
        default=None,
        description="Snapshot ID to create sandbox from. Takes precedence over image if both are set.",
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Custom environment variables passed to the sandbox on creation.",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Custom labels attached to the sandbox on creation.",
    )
    sandbox_id: str | None = Field(
        default=None,
        description="Existing sandbox ID to reconnect to. If None, a new sandbox is created.",
    )
    auto_stop_interval: int | None = Field(
        default=None,
        description="Auto-stop interval in minutes. 0 means no auto-stop.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for sandbox creation and reconnection.",
    )
    _sandbox: DaytonaSandboxInstance | None = PrivateAttr(default=None)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude sensitive fields from serialization."""
        return super().to_dict_exclude_params | {"envs": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the DaytonaSandbox instance to a dictionary."""
        for_tracing = kwargs.get("for_tracing", False)
        data = super().to_dict(**kwargs)
        if not for_tracing and self.envs is not None:
            data["envs"] = self.envs
        return data

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current sandbox ID (for saving/reconnecting later)."""
        return self.sandbox_id

    def _ensure_sandbox(self):
        """Lazily initialize or reconnect to Daytona sandbox, with retries on rate-limit.

        Uses double-checked locking so concurrent threads never create duplicate sandboxes.
        """
        if self._sandbox is not None:
            return self._sandbox

        with self._sandbox_lock:
            if self._sandbox is not None:
                return self._sandbox

            if self.sandbox_id:
                try:
                    self._sandbox = self._reconnect_with_retry()
                    logger.debug(f"Daytona sandbox reconnected: {self.sandbox_id}")

                    self._ensure_directories()
                    return self._sandbox
                except Exception as e:
                    raise SandboxConnectionError(self.sandbox_id, cause=e, provider="Daytona") from e

            self._sandbox = self._create_with_retry()
            self.sandbox_id = self._sandbox.id
            logger.debug(f"Daytona sandbox created: {self.sandbox_id}")
            self._ensure_directories()
            return self._sandbox

    def _ensure_directories(self) -> None:
        """Create the base directory inside the sandbox if it does not exist."""
        if self._sandbox is None:
            return
        try:
            self._sandbox.process.exec(f"mkdir -p {shlex.quote(self.base_path)}")
            logger.debug(f"DaytonaSandbox ensured directory exists: {self.base_path}")
        except Exception as e:
            logger.warning(f"DaytonaSandbox failed to create directory: {e}")

    def _reconnect_with_retry(self):
        """Reconnect to existing sandbox with exponential backoff on rate-limit."""
        cfg = self.creation_error_handling
        client = self.connection.get_client()

        @retry(
            retry=retry_if_exception_type(DaytonaRateLimitError),
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
            logger.debug(f"Reconnecting to Daytona sandbox: {self.sandbox_id}")
            return client.get(self.sandbox_id)

        return connect()

    def _create_with_retry(self):
        """Create a new sandbox with exponential backoff on rate-limit."""
        cfg = self.creation_error_handling
        client = self.connection.get_client()

        @retry(
            retry=retry_if_exception_type(DaytonaRateLimitError),
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
                if self.snapshot:
                    params = CreateSandboxFromSnapshotParams(
                        snapshot=self.snapshot,
                        env_vars=self.envs,
                        labels=self.labels,
                        auto_stop_interval=self.auto_stop_interval,
                    )
                elif self.image:
                    params = CreateSandboxFromImageParams(
                        image=self.image,
                        env_vars=self.envs,
                        labels=self.labels,
                        auto_stop_interval=self.auto_stop_interval,
                    )
                else:
                    params = CreateSandboxFromSnapshotParams(
                        env_vars=self.envs,
                        labels=self.labels,
                        auto_stop_interval=self.auto_stop_interval,
                    )
                return client.create(params, timeout=self.timeout)
            except DaytonaRateLimitError:
                logger.warning("Daytona sandbox creation rate-limited. Retrying with exponential backoff.")
                raise

        return create()

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the Daytona sandbox."""
        sandbox = self._ensure_sandbox()
        logger.debug(f"DaytonaSandbox running command: {command[:100]}...")

        try:
            if run_in_background_enabled:
                sandbox.process.exec(f"nohup {command} > /dev/null 2>&1 &")
                return ShellCommandResult(background=True)

            result = sandbox.process.exec(command, timeout=timeout)
            return ShellCommandResult(
                stdout=result.result,
                stderr=None,
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
        """Upload a file to the Daytona sandbox."""
        sandbox = self._ensure_sandbox()

        if destination_path is None:
            destination_path = f"{self.base_path}/{file_name}"

        if ensure_parent_dirs and destination_path:
            parent = destination_path.rsplit("/", 1)[0]
            if parent and parent != destination_path:
                try:
                    sandbox.process.exec(f"mkdir -p {shlex.quote(parent)}")
                    logger.debug(f"DaytonaSandbox ensured parent dir: {parent}")
                except Exception as e:
                    logger.warning(f"DaytonaSandbox mkdir -p for {parent!r} failed (continuing): {e}")

        try:
            sandbox.fs.upload_file(content, destination_path)
            logger.debug(f"DaytonaSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the Daytona sandbox directory."""
        sandbox = self._ensure_sandbox()
        if target_dir is None:
            target_dir = self.base_path

        try:
            file_infos = sandbox.fs.list_files(target_dir)
            result = []
            for fi in file_infos:
                full_path = f"{target_dir.rstrip('/')}/{fi.name}"
                if fi.is_dir:
                    result.extend(self._list_files_recursive(sandbox, full_path))
                else:
                    result.append(full_path)
                if len(result) >= self.max_output_files:
                    break
            return result[: self.max_output_files]
        except Exception as e:
            logger.warning(f"DaytonaSandbox list_files failed for {target_dir}: {e}")
            return []

    def _list_files_recursive(self, sandbox, dir_path: str, depth: int = 0) -> list[str]:
        """Recursively list files in a directory."""
        if depth > 3:
            return []
        result = []
        try:
            file_infos = sandbox.fs.list_files(dir_path)
            for fi in file_infos:
                full_path = f"{dir_path.rstrip('/')}/{fi.name}"
                if fi.is_dir:
                    result.extend(self._list_files_recursive(sandbox, full_path, depth + 1))
                else:
                    result.append(full_path)
        except Exception as e:
            logger.warning(f"DaytonaSandbox _list_files_recursive failed for {dir_path}: {e}")
        return result

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in sandbox filesystem."""
        try:
            sandbox = self._ensure_sandbox()
            resolved_path = self._resolve_path(file_path)
            sandbox.fs.get_file_info(resolved_path)
            return True
        except Exception as e:
            logger.debug(f"DaytonaSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from sandbox filesystem."""
        sandbox = self._ensure_sandbox()
        resolved_path = self._resolve_path(file_path)
        return sandbox.fs.download_file(resolved_path)

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
        """Return sandbox metadata including optional public URL for a port."""
        public_host: str | None = None
        public_url: str | None = None
        public_url_error: str | None = None
        if port is not None:
            try:
                sandbox = self._ensure_sandbox()
                preview = sandbox.get_preview_link(port)
                public_url = preview.url
                # Extract host from URL
                if public_url and public_url.startswith("https://"):
                    public_host = public_url[len("https://") :]
                elif public_url and public_url.startswith("http://"):
                    public_host = public_url[len("http://") :]
            except Exception as e:
                logger.debug("get_preview_link failed: %s", e)
                public_url_error = str(e)
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.sandbox_id,
            public_host=public_host,
            public_url=public_url,
            public_url_error=public_url_error,
        )

    def close(self, kill: bool = False) -> None:
        """Close the Daytona sandbox connection.

        Args:
            kill: If False (default), just disconnects but keeps the sandbox alive
                  for reconnection. If True, deletes the sandbox.
        """
        if self._sandbox:
            try:
                if kill:
                    client = self.connection.get_client()
                    client.delete(self._sandbox)
                    logger.debug(f"Daytona sandbox deleted: {self.sandbox_id}")
                    self.sandbox_id = None
                else:
                    logger.debug(f"Daytona sandbox disconnected (kept alive): {self.sandbox_id}")
            except Exception as e:
                logger.warning(f"DaytonaSandbox close() failed: {e}")
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
