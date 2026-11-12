import shlex
import threading
import uuid
from typing import Any

from boxd import Boxd as BoxdClient
from boxd import Machine, NotFoundError, RateLimitError
from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import Boxd
from dynamiq.nodes import Node
from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger

BOXD_HOME = "/home/boxd"
BOXD_WORKSPACE = f"{BOXD_HOME}/workspace"
DEFAULT_MACHINE_NAME_PREFIX = "dynamiq-"
# The machine image ships Python under PEP 668, so pip installs go to a virtualenv. It lives outside the home
# directory so that listing the base path does not return its files.
PYTHON_ENV_PATH = "/opt/dynamiq/venv"
_PREPARE_PYTHON_ENV = (
    f"test -x {PYTHON_ENV_PATH}/bin/python || "
    f'(sudo install -d -o "$(id -un)" -g "$(id -gn)" /opt/dynamiq && uv venv -q --seed {PYTHON_ENV_PATH})'
)
_ACTIVATE_PYTHON_ENV = f'export VIRTUAL_ENV={PYTHON_ENV_PATH} PATH="{PYTHON_ENV_PATH}/bin:$PATH"\n'
# Machine statuses that need a call before the machine accepts commands again.
MACHINE_STATUS_STOPPED = "stopped"
MACHINE_STATUS_HIBERNATED = "hibernated"
MACHINE_STATUS_SUSPENDED = "suspended"


def boxd_shell_argv(command: str) -> list[str]:
    """Wrap a shell command so that it runs in bash with the Python virtualenv on PATH.

    Passing an argv runs the whole command under one bash process, so environment variables given to the exec
    call reach every part of a compound command.
    """
    return ["bash", "-c", _ACTIVATE_PYTHON_ENV + command]


def create_boxd_machine(
    connection: Boxd,
    *,
    name_prefix: str = DEFAULT_MACHINE_NAME_PREFIX,
    snapshot: str | None = None,
    labels: dict[str, str] | None = None,
    vcpu: int | None = None,
    memory: str | None = None,
    auto_suspend_timeout: int | None = 0,
    delete_after: int | None = None,
    ready_timeout: float = 120,
) -> Machine:
    """Create an isolated machine, wait until it accepts commands and prepare its Python environment.

    ``delete_after`` is the number of seconds after which boxd deletes the machine and its disk. None or 0 sets no
    timer.
    """
    client: BoxdClient = connection.get_client()
    machine = client.machines.create(
        f"{name_prefix}{uuid.uuid4().hex[:8]}",
        from_snapshot=snapshot,
        org=connection.org,
        # Agents run untrusted code: an isolated machine has no private network peers and no platform credentials.
        isolated=True,
        labels=labels,
        vcpu=vcpu,
        memory=memory,
        auto_suspend_timeout=auto_suspend_timeout,
        delete_after=delete_after or None,
    )
    try:
        machine = client.machines.wait_until_ready(machine.id, timeout=ready_timeout)
        _prepare_python_env(client, machine.id)
    except Exception:
        _delete_quietly(client, machine.id)
        raise
    return machine


def connect_boxd_machine(connection: Boxd, machine_id: str, ready_timeout: float = 120) -> Machine:
    """Return a running machine by id, starting or waking it first when it is not running."""
    client: BoxdClient = connection.get_client()
    machine = client.machines.get(machine_id)
    if machine.status == MACHINE_STATUS_STOPPED:
        client.machines.start(machine_id)
    elif machine.status == MACHINE_STATUS_HIBERNATED:
        client.machines.wake(machine_id)
    elif machine.status == MACHINE_STATUS_SUSPENDED:
        client.machines.resume(machine_id)
    machine = client.machines.wait_until_ready(machine_id, timeout=ready_timeout)
    _prepare_python_env(client, machine_id)
    return machine


def _prepare_python_env(client: BoxdClient, machine_id: str) -> None:
    result = client.machines.exec(machine_id, ["bash", "-c", _PREPARE_PYTHON_ENV], timeout=120)
    if result.exit_code != 0:
        raise RuntimeError(f"Failed to prepare the Python environment: {result.stderr or result.stdout}")


def _delete_quietly(client: BoxdClient, machine_id: str) -> None:
    try:
        client.machines.delete(machine_id)
    except NotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Failed to delete boxd machine {machine_id}: {e}")


class BoxdSandbox(Sandbox):
    """boxd sandbox implementation.

    Runs commands and stores files in an isolated boxd machine: a full Ubuntu VM with a persistent disk.
    Files persist until the machine is deleted with ``close(kill=True)``, or by boxd once ``delete_after`` seconds
    have passed. ``auto_suspend_timeout`` sets when an idle machine goes to standby; a reconnect starts or wakes the
    machine before the first command.

    Supports reconnecting to existing machines by providing sandbox_id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: Boxd
    base_path: str = Field(default=BOXD_WORKSPACE, description="Base path in the sandbox filesystem.")
    timeout: int = Field(default=120, description="Seconds to wait for a new or woken machine to accept commands.")

    snapshot: str | None = Field(
        default=None,
        description="Name of a boxd snapshot to create the machine from. If None, the default image is used.",
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Custom environment variables passed to every command run in the sandbox.",
    )
    labels: dict[str, str] | None = Field(
        default=None,
        description="Custom labels attached to the machine on creation.",
    )
    sandbox_id: str | None = Field(
        default=None,
        description="Existing machine ID to reconnect to. If None, a new machine is created.",
    )
    name_prefix: str = Field(
        default=DEFAULT_MACHINE_NAME_PREFIX,
        description="Prefix of the machine name. A random suffix makes each name unique.",
    )
    vcpu: int | None = Field(default=None, description="vCPUs of a new machine. If None, the org default is used.")
    memory: str | None = Field(
        default=None, description="Memory of a new machine, for example '8G'. If None, the org default is used."
    )
    auto_suspend_timeout: int | None = Field(
        default=0,
        description=(
            "Seconds without network traffic before the machine goes to standby. 0 (the default) disables it. "
            "If None, the org default is used."
        ),
    )
    delete_after: int | None = Field(
        default=None,
        description=(
            "Seconds after creation when boxd deletes the machine and its disk, even while it is in use. "
            "None or 0 keeps the machine until close(kill=True)."
        ),
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for machine creation and reconnection.",
    )
    _machine: Machine | None = PrivateAttr(default=None)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude sensitive fields from serialization."""
        return super().to_dict_exclude_params | {"envs": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Convert the BoxdSandbox instance to a dictionary."""
        for_tracing = kwargs.get("for_tracing", False)
        data = super().to_dict(**kwargs)
        if not for_tracing and self.envs is not None:
            data["envs"] = self.envs
        return data

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current machine ID (for saving/reconnecting later)."""
        return self.sandbox_id

    @property
    def supports_views(self) -> bool:
        """boxd sandboxes can share one machine between views."""
        return True

    def ensure_started(self) -> str | None:
        """Ensure the machine exists (create/reconnect) and return its id."""
        self._ensure_sandbox()
        return self.current_sandbox_id

    def create_view(self, base_path: str, sandbox_id: str | None = None) -> "BoxdSandbox":
        """Create a sibling BoxdSandbox that reconnects to the same machine."""
        return BoxdSandbox(
            connection=self.connection,
            sandbox_id=sandbox_id or self.current_sandbox_id,
            base_path=base_path,
            timeout=self.timeout,
            snapshot=self.snapshot,
            envs=self.envs,
            labels=self.labels,
            name_prefix=self.name_prefix,
            vcpu=self.vcpu,
            memory=self.memory,
            auto_suspend_timeout=self.auto_suspend_timeout,
            delete_after=self.delete_after,
            creation_error_handling=self.creation_error_handling,
        )

    @property
    def _client(self) -> BoxdClient:
        return self.connection.get_client()

    def _ensure_sandbox(self) -> Machine:
        """Lazily create or reconnect to the machine, with retries on rate-limit.

        Uses double-checked locking so concurrent threads never create duplicate machines.
        """
        if self._machine is not None:
            return self._machine

        with self._sandbox_lock:
            if self._machine is not None:
                return self._machine

            if self.sandbox_id:
                try:
                    machine = self._with_retry(
                        lambda: connect_boxd_machine(self.connection, self.sandbox_id, self.timeout)
                    )
                    logger.debug(f"boxd machine reconnected: {self.sandbox_id}")
                    self._ensure_directories(machine)
                except Exception as e:
                    raise SandboxConnectionError(self.sandbox_id, cause=e, provider="boxd") from e
                # Published only now: other threads skip the lock once it is set and run commands in base_path.
                self._machine = machine
                return machine

            machine = self._with_retry(
                lambda: create_boxd_machine(
                    self.connection,
                    name_prefix=self.name_prefix,
                    snapshot=self.snapshot,
                    labels=self.labels,
                    vcpu=self.vcpu,
                    memory=self.memory,
                    auto_suspend_timeout=self.auto_suspend_timeout,
                    delete_after=self.delete_after,
                    ready_timeout=self.timeout,
                )
            )
            logger.debug(f"boxd machine created: {machine.id}")
            try:
                self._ensure_directories(machine)
            except Exception:
                _delete_quietly(self._client, machine.id)
                raise
            self.sandbox_id = machine.id
            self._machine = machine
            return machine

    def _with_retry(self, fn):
        """Call fn with exponential backoff on rate-limit."""
        cfg = self.creation_error_handling

        @retry(
            retry=retry_if_exception_type(RateLimitError),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def call():
            return fn()

        return call()

    def _ensure_directories(self, machine: Machine) -> None:
        """Create the base directory inside the machine: every command runs in it."""
        result = self._client.machines.exec(machine.id, ["mkdir", "-p", self.base_path], timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create {self.base_path}: {result.stderr or result.stdout}")
        logger.debug(f"BoxdSandbox ensured directory exists: {self.base_path}")

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the boxd machine, in the base path."""
        machine = self._ensure_sandbox()
        logger.debug(f"BoxdSandbox running command: {command[:100]}...")

        try:
            if run_in_background_enabled:
                # setsid detaches the command from the exec session, so it keeps running after the call returns.
                background = f"setsid nohup bash -c {shlex.quote(command)} > /dev/null 2>&1 < /dev/null &"
                self._client.machines.exec(
                    machine.id, boxd_shell_argv(background), env=self.envs, cwd=self.base_path, timeout=60
                )
                return ShellCommandResult(background=True)

            result = self._client.machines.exec(
                machine.id, boxd_shell_argv(command), env=self.envs, cwd=self.base_path, timeout=timeout
            )
            return ShellCommandResult(
                stdout=result.stdout,
                stderr=result.stderr,
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
        """Upload a file to the boxd machine."""
        machine = self._ensure_sandbox()

        if destination_path is None:
            destination_path = f"{self.base_path}/{file_name}"

        if ensure_parent_dirs and destination_path:
            parent = destination_path.rsplit("/", 1)[0]
            if parent and parent != destination_path:
                try:
                    self._client.machines.exec(machine.id, ["mkdir", "-p", parent], timeout=60)
                    logger.debug(f"BoxdSandbox ensured parent dir: {parent}")
                except Exception as e:
                    logger.warning(f"BoxdSandbox mkdir -p for {parent!r} failed (continuing): {e}")

        try:
            self._client.machines.files.upload(machine.id, destination_path, content)
            logger.debug(f"BoxdSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the boxd machine directory, at most ``max_output_files``."""
        machine = self._ensure_sandbox()
        if target_dir is None:
            target_dir = self.base_path

        try:
            cmd = (
                f"test -d {shlex.quote(target_dir)} || exit 0; "
                f"find {shlex.quote(target_dir)} -type f 2>/dev/null | head -{self.max_output_files}"
            )
            result = self._client.machines.exec(machine.id, ["bash", "-c", cmd], timeout=60)
            if result.exit_code != 0:
                return []
            return [f.strip() for f in result.stdout.splitlines() if f.strip()]
        except Exception as e:
            logger.warning(f"BoxdSandbox list_files failed for {target_dir}: {e}")
            return []

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in the machine filesystem."""
        try:
            machine = self._ensure_sandbox()
            resolved_path = self._resolve_path(file_path)
            result = self._client.machines.exec(machine.id, ["test", "-f", resolved_path], timeout=60)
            return result.exit_code == 0
        except Exception as e:
            logger.debug(f"BoxdSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from the machine filesystem."""
        machine = self._ensure_sandbox()
        resolved_path = self._resolve_path(file_path)
        return self._client.machines.files.download(machine.id, resolved_path)

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

    def get_public_host(self, port: int) -> str:
        """Return the public HTTPS host that forwards to a port inside the machine.

        Creates a named proxy ``p<port>`` on the machine the first time a port is asked for. The URL is public:
        anyone who has it can reach the service. A new URL can take up to a minute before it answers, so retry the
        first request.

        Args:
            port: Port number the service listens on inside the machine (e.g. 3000, 5173).

        Returns:
            Host string (e.g. p3000.<machine>.boxd.sh). Full URL is https://{host}.
        """
        machine = self._ensure_sandbox()
        proxy_name = f"p{port}"
        proxies = self._client.machines.proxies.list(machine.id)
        if not any(p.name == proxy_name and p.port == port for p in proxies):
            self._client.machines.proxies.create(machine.id, proxy_name, port)
        machine_host = machine.access.url.split("://", 1)[-1].rstrip("/")
        return f"{proxy_name}.{machine_host}"

    def get_sandbox_info(self, port: int | None = None) -> SandboxInfo:
        """Return sandbox metadata including optional public URL for a port."""
        public_host: str | None = None
        public_url: str | None = None
        public_url_error: str | None = None
        if port is not None:
            try:
                public_host = self.get_public_host(port)
                public_url = f"https://{public_host}"
            except Exception as e:
                logger.debug("get_public_host failed: %s", e)
                public_url_error = str(e)
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.sandbox_id,
            public_host=public_host,
            public_url=public_url,
            public_url_error=public_url_error,
        )

    def close(self, kill: bool = False) -> None:
        """Close the boxd sandbox connection.

        Args:
            kill: If False (default), just disconnects but keeps the machine for reconnection. If True, deletes
                the machine and its disk.
        """
        if self._machine:
            try:
                if kill:
                    self._client.machines.delete(self._machine.id)
                    logger.debug(f"boxd machine deleted: {self.sandbox_id}")
                    self.sandbox_id = None
                else:
                    logger.debug(f"boxd machine disconnected (kept): {self.sandbox_id}")
            except Exception as e:
                logger.warning(f"BoxdSandbox close() failed: {e}")
            finally:
                self._machine = None

    def __enter__(self):
        """Context manager entry."""
        self._ensure_sandbox()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit. Disconnects and keeps the machine, like ``close()``.

        Call ``close(kill=True)`` inside the block, or set ``delete_after``, to delete the machine.
        """
        self.close()

    def __del__(self):
        """Destructor - attempt to close sandbox on garbage collection."""
        try:
            self.close()
        except Exception:
            ...
