import shlex
import threading
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import AWS
from dynamiq.connections.agentcore import (
    DEFAULT_CODE_INTERPRETER_IDENTIFIER,
    AgentCoreCodeInterpreterClient,
    AgentCoreSession,
    AgentCoreThrottlingError,
    normalize_sandbox_path,
)
from dynamiq.nodes import Node
from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.utils.logger import logger


class BedrockAgentCoreSandbox(Sandbox):
    """AWS Bedrock AgentCore Code Interpreter sandbox implementation.

    Stores files and executes shell commands in an AWS-managed microVM session.
    Files persist for the lifetime of the session (up to 8 hours). Supports
    reconnecting to existing sessions by providing session_id. AgentCore does
    not expose public ports, so no public URLs are available.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: AWS
    base_path: str = Field(default=".", description="Base path in the sandbox filesystem (workspace-relative).")
    timeout: int = Field(
        default=3600,
        description="Session timeout in seconds, fixed at session start (AWS caps it at 28800).",
    )
    code_interpreter_identifier: str = Field(
        default=DEFAULT_CODE_INTERPRETER_IDENTIFIER,
        description="AgentCore code interpreter identifier (built-in or custom interpreter ID).",
    )
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to reconnect to. If None, a new session is started.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for session creation and reconnection.",
    )
    _client: AgentCoreCodeInterpreterClient | None = PrivateAttr(default=None)
    _session: AgentCoreSession | None = PrivateAttr(default=None)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current session ID (for saving/reconnecting later)."""
        return self.session_id

    def _get_client(self) -> AgentCoreCodeInterpreterClient:
        if self._client is None:
            self._client = AgentCoreCodeInterpreterClient(self.connection)
        return self._client

    def _resolve_path(self, file_path: str) -> str:
        """Resolve relative file paths against the workspace-relative base path."""
        return normalize_sandbox_path(super()._resolve_path(file_path))

    def _ensure_session(self) -> AgentCoreSession:
        """Lazily start or reconnect to an AgentCore session, with retries on throttling.

        Uses double-checked locking so concurrent threads never create duplicate sessions.
        """
        if self._session is not None:
            return self._session

        with self._sandbox_lock:
            if self._session is not None:
                return self._session

            if self.session_id:
                try:
                    self._session = self._reconnect_with_retry()
                    logger.debug(f"AgentCore session reconnected: {self.session_id}")
                    self._ensure_directories()
                    return self._session
                except Exception as e:
                    raise SandboxConnectionError(self.session_id, cause=e, provider="BedrockAgentCore") from e

            self._session = self._create_with_retry()
            self.session_id = self._session.session_id
            logger.debug(f"AgentCore session started: {self.session_id}")
            self._ensure_directories()
            return self._session

    def _ensure_directories(self) -> None:
        """Create the base directory inside the sandbox if it does not exist."""
        if self._session is None or self.base_path in (".", ""):
            return
        try:
            self._get_client().invoke(
                self._session, "executeCommand", {"command": f"mkdir -p {shlex.quote(self.base_path)}"}
            )
            logger.debug(f"BedrockAgentCoreSandbox ensured directory exists: {self.base_path}")
        except Exception as e:
            logger.warning(f"BedrockAgentCoreSandbox failed to create directory: {e}")

    def _reconnect_with_retry(self) -> AgentCoreSession:
        """Reconnect to an existing session with exponential backoff on throttling."""
        cfg = self.creation_error_handling
        client = self._get_client()

        @retry(
            retry=retry_if_exception_type(AgentCoreThrottlingError),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def connect() -> AgentCoreSession:
            logger.debug(f"Reconnecting to AgentCore session: {self.session_id}")
            status = client.get_session_status(self.code_interpreter_identifier, self.session_id)
            if status != "READY":
                raise ValueError(
                    f"AgentCore session {self.session_id} is {status or 'unavailable'}; "
                    "terminated sessions cannot be restarted"
                )
            return AgentCoreSession(identifier=self.code_interpreter_identifier, session_id=self.session_id)

        return connect()

    def _create_with_retry(self) -> AgentCoreSession:
        """Start a new session with exponential backoff on throttling."""
        cfg = self.creation_error_handling
        client = self._get_client()

        @retry(
            retry=retry_if_exception_type(AgentCoreThrottlingError),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def create() -> AgentCoreSession:
            try:
                return client.start_session(
                    identifier=self.code_interpreter_identifier,
                    session_timeout_seconds=self.timeout,
                )
            except AgentCoreThrottlingError:
                logger.warning("AgentCore session creation throttled. Retrying with exponential backoff.")
                raise

        return create()

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the AgentCore session."""
        session = self._ensure_session()
        logger.debug(f"BedrockAgentCoreSandbox running command: {command[:100]}...")

        try:
            if run_in_background_enabled:
                self._get_client().invoke(session, "startCommandExecution", {"command": command})
                return ShellCommandResult(background=True)

            result = self._get_client().invoke(session, "executeCommand", {"command": command})
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
        """Upload a file to the AgentCore session."""
        session = self._ensure_session()

        if destination_path is None:
            destination_path = f"{self.base_path.rstrip('/')}/{file_name}"
        destination_path = normalize_sandbox_path(destination_path)

        if ensure_parent_dirs and "/" in destination_path:
            parent = destination_path.rsplit("/", 1)[0]
            if parent:
                try:
                    self._get_client().invoke(session, "executeCommand", {"command": f"mkdir -p {shlex.quote(parent)}"})
                    logger.debug(f"BedrockAgentCoreSandbox ensured parent dir: {parent}")
                except Exception as e:
                    logger.warning(f"BedrockAgentCoreSandbox mkdir -p for {parent!r} failed (continuing): {e}")

        try:
            result = self._get_client().invoke(
                session, "writeFiles", {"content": [{"path": destination_path, "blob": content}]}
            )
            if result.is_error:
                raise RuntimeError(result.error_text)
            logger.debug(f"BedrockAgentCoreSandbox uploaded file: {destination_path}")
            return destination_path
        except Exception as e:
            logger.error(f"Failed to upload file {file_name}: {e}")
            raise

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the AgentCore session directory."""
        session = self._ensure_session()
        if target_dir is None:
            target_dir = self.base_path

        try:
            command = f"find {shlex.quote(target_dir)} -maxdepth 3 -type f 2>/dev/null | head -{self.max_output_files}"
            result = self._get_client().invoke(session, "executeCommand", {"command": command})
            files = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            return files[: self.max_output_files]
        except Exception as e:
            logger.warning(f"BedrockAgentCoreSandbox list_files failed for {target_dir}: {e}")
            return []

    def exists(self, file_path: str) -> bool:
        """Return True when file exists in the sandbox filesystem."""
        try:
            session = self._ensure_session()
            resolved_path = self._resolve_path(file_path)
            result = self._get_client().invoke(
                session, "executeCommand", {"command": f"test -e {shlex.quote(resolved_path)}"}
            )
            return result.exit_code == 0
        except Exception as e:
            logger.debug(f"BedrockAgentCoreSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from the sandbox filesystem."""
        session = self._ensure_session()
        resolved_path = self._resolve_path(file_path)
        client = self._get_client()
        result = client.invoke(session, "readFiles", {"paths": [resolved_path]})
        if result.is_error:
            raise FileNotFoundError(f"Failed to read file '{resolved_path}': {result.error_text}")
        return client.extract_file_bytes(result)

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
        """Return sandbox metadata. AgentCore does not expose public ports or URLs."""
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.session_id,
            public_url_error=("AgentCore sandboxes do not expose public ports or URLs" if port is not None else None),
        )

    def close(self, kill: bool = False) -> None:
        """Close the AgentCore session connection.

        Args:
            kill: If False (default), just disconnects but keeps the session alive
                  for reconnection until its timeout. If True, stops the session.
        """
        if self._session:
            try:
                if kill:
                    self._get_client().stop_session(self._session)
                    logger.debug(f"AgentCore session stopped: {self.session_id}")
                    self.session_id = None
                else:
                    logger.debug(f"AgentCore session disconnected (kept alive): {self.session_id}")
            except Exception as e:
                logger.warning(f"BedrockAgentCoreSandbox close() failed: {e}")
            finally:
                self._session = None

    def __enter__(self):
        """Context manager entry."""
        self._ensure_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - attempt to close the session connection on garbage collection."""
        try:
            self.close()
        except Exception:
            ...
