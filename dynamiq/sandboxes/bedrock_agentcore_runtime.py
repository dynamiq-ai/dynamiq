"""AWS Bedrock AgentCore **Runtime** sandbox backend.

A second, independent AgentCore backend alongside
:class:`~dynamiq.sandboxes.bedrock_agentcore.BedrockAgentCoreSandbox`, which is built on the
Code Interpreter API. Both are kept: the Code Interpreter backend is simpler and needs no
provisioned image, while this one is the only AgentCore backend that can carry a custom
image, environment variables, cross-session persistence and shared views.

The two differ in more than transport:

- Commands are ``exec``'d, not shell-interpreted — the client wraps every one in ``bash -c``.
- There is a real per-invocation ``timeout`` (1..3600 s) and a real ``exitCode``, so the
  ``timeout`` coreutils wrapper the Code Interpreter backend needs is gone.
- There is **no file API**. Uploads and downloads go through the shell: inline base64 for
  small files, an S3 relay for anything larger.
- Sessions are addressed by a caller-chosen id and survive being stopped, so views and
  reconnection work.
"""

import base64
import shlex
import threading
import uuid
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.connections import AWS
from dynamiq.nodes import Node
from dynamiq.nodes.tools.code_interpreter import SandboxCreationErrorHandling
from dynamiq.sandboxes.base import Sandbox, SandboxInfo, ShellCommandResult
from dynamiq.sandboxes.bedrock_agentcore_runtime_client import (
    DEFAULT_BASE_PATH,
    DEFAULT_INLINE_TRANSFER_MAX_BYTES,
    DEFAULT_SESSION_ENV_PATH,
    AgentCoreRuntimeClient,
    AgentCoreRuntimeCommandResult,
    AgentCoreRuntimeConflictError,
    AgentCoreRuntimeThrottlingError,
    build_env_bootstrap_command,
    build_env_prefix,
    derive_session_id,
    normalize_sandbox_path,
)
from dynamiq.utils.logger import logger

PRESIGNED_URL_EXPIRY_SECONDS = 3600


class BedrockAgentCoreRuntimeSandbox(Sandbox):
    """AWS Bedrock AgentCore Runtime sandbox.

    Executes shell commands and stores files inside an AWS-managed microVM session
    addressed by ``agent_runtime_arn`` + ``session_id``. The runtime itself is a
    deployment-time resource (an image plus its configuration, closer to an E2B *template*
    than to a sandbox): it is provisioned out of band and is never created here.

    AgentCore exposes no inbound ports, so no public preview URLs are available.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    connection: AWS
    agent_runtime_arn: str = Field(description="ARN of the provisioned AgentCore agent runtime.")
    qualifier: str | None = Field(
        default=None,
        description=(
            "Endpoint alias pinning a runtime version (e.g. 'v7'). Omitted means the default "
            "endpoint. Persist it per conversation so a deploy cannot move a live session to a "
            "new version and wipe its storage."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to reconnect to. Derived from session_seed, or random, if unset.",
    )
    session_seed: str | None = Field(
        default=None,
        description=(
            "Seed for deterministic session-ID derivation (typically a conversation ID). Lets a "
            "caller reconnect without storing the ID."
        ),
    )
    base_path: str = Field(default=DEFAULT_BASE_PATH, description="Base path in the sandbox filesystem.")
    session_env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Environment variables written to a session-scoped env file and sourced by every "
            "shell command. The values travel inside a command string, so prefer "
            "session_env_param for secrets."
        ),
    )
    session_env_param: str | None = Field(
        default=None,
        description=(
            "SSM Parameter Store parameter holding the env-file content, fetched from inside "
            "the container with the runtime's execution role. Only the name is ever sent, so "
            "the value reaches no command string. Takes precedence over session_env."
        ),
    )
    session_env_path: str = Field(
        default=DEFAULT_SESSION_ENV_PATH,
        description=(
            "Where the session env file lives. Pinned to the session-storage mount rather "
            "than base_path so that every view of a session shares one file."
        ),
    )
    timeout: int = Field(default=300, description="Per-command timeout in seconds (AgentCore allows 1..3600).")
    s3_relay_bucket: str | None = Field(
        default=None,
        description="Bucket used to relay files larger than inline_transfer_max_bytes. Required for large transfers.",
    )
    s3_relay_prefix: str = Field(default="dynamiq-agentcore-relay", description="Key prefix for relayed objects.")
    inline_transfer_max_bytes: int = Field(
        default=DEFAULT_INLINE_TRANSFER_MAX_BYTES,
        description="Files at or below this size transfer inline as base64; larger ones use the S3 relay.",
    )
    creation_error_handling: SandboxCreationErrorHandling = Field(
        default_factory=SandboxCreationErrorHandling,
        description="Retry and backoff config for transient throttling and session-provisioning conflicts.",
    )
    _client: AgentCoreRuntimeClient | None = PrivateAttr(default=None)
    _s3_client: Any = PrivateAttr(default=None)
    _started: bool = PrivateAttr(default=False)
    _sandbox_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude sensitive fields from model_dump; re-added in to_dict when not tracing."""
        return super().to_dict_exclude_params | {"session_env": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the sandbox, omitting ``session_env`` from tracing output.

        Mirrors :class:`~dynamiq.sandboxes.e2b.E2BSandbox`. Note what this does *not* buy:
        the field is still present in a normal (``for_tracing=False``) dump, because the
        executing process has to receive the values somehow. Anywhere that dump crosses a
        trust boundary — a serialized workflow on a queue, a persisted node config — the
        values cross with it. ``session_env_param`` is the answer there; it carries a name
        instead of a secret and is safe to serialize.
        """
        for_tracing = kwargs.get("for_tracing", False)
        data = super().to_dict(**kwargs)
        if not for_tracing and self.session_env is not None:
            data["session_env"] = self.session_env
        return data

    @property
    def current_sandbox_id(self) -> str | None:
        """Get the current session ID (for saving/reconnecting later)."""
        return self.session_id

    @property
    def supports_views(self) -> bool:
        """Concurrent invocations against one session are supported by the platform."""
        return True

    def _get_client(self) -> AgentCoreRuntimeClient:
        if self._client is None:
            self._client = AgentCoreRuntimeClient(self.connection)
        return self._client

    def _get_s3_client(self):
        if self._s3_client is None:
            self._s3_client = self.connection.get_boto3_session().client("s3")
        return self._s3_client

    def _resolve_path(self, file_path: str) -> str:
        """Resolve a path against ``base_path``, applying the prefix at most once.

        Idempotent: a path that is already absolute is returned unchanged, so callers that
        pre-resolve (``collect_files``/``delete_file`` in the base class) are not
        double-prefixed.
        """
        normalized = normalize_sandbox_path(file_path)
        if normalized.startswith("/"):
            return normalized
        return normalize_sandbox_path(f"{self.base_path.rstrip('/')}/{normalized}")

    def ensure_started(self) -> str | None:
        """Materialize the session id and make sure ``base_path`` exists.

        AgentCore creates the session implicitly on first invocation, so there is no session
        to "start" — this only fixes the id and prepares the workspace.
        """
        if self._started:
            return self.session_id

        with self._sandbox_lock:
            if self._started:
                return self.session_id

            if not self.session_id:
                seed = self.session_seed or uuid.uuid4().hex
                self.session_id = derive_session_id(seed)
                logger.debug(f"AgentCore Runtime session id derived: {self.session_id}")

            self._started = True
            self._ensure_directories()
            self._bootstrap_env()
            return self.session_id

    @property
    def _has_session_env(self) -> bool:
        return bool(self.session_env or self.session_env_param)

    def _bootstrap_env(self) -> None:
        """Write the session env file so later commands can source it.

        Overwrites unconditionally rather than skipping an existing file. Session storage
        outlives the credentials in it — a caller refreshing a short-lived token constructs
        a new sandbox against the same session, and a "write once" bootstrap would silently
        keep serving the expired value.

        Goes through :meth:`_invoke_with_retry` rather than :meth:`run_command_shell` for two
        reasons: ``_sandbox_lock`` is not reentrant and is held here, and the prefix
        ``run_command_shell`` applies would source the very file this is writing.
        """
        if not self._has_session_env:
            return

        command = build_env_bootstrap_command(
            self.session_env_path,
            env=self.session_env,
            param_name=self.session_env_param,
            region=self.connection.region or None,
        )
        # Failures are reported without their output: for the inline form the command
        # carries the values, and a shell is free to echo the failing command back.
        try:
            result = self._invoke_with_retry(command, timeout=self.timeout)
            if not result.is_success:
                logger.warning(
                    "BedrockAgentCoreRuntimeSandbox failed to write the session env file at "
                    f"{self.session_env_path} (exit code {result.exit_code}); commands will run without it"
                )
        except Exception:
            logger.warning(
                "BedrockAgentCoreRuntimeSandbox failed to write the session env file at "
                f"{self.session_env_path}; commands will run without it"
            )

    def _ensure_directories(self) -> None:
        """Create the base directory inside the sandbox if it does not exist."""
        if not self.base_path or self.base_path == "/":
            return
        try:
            result = self._invoke_with_retry(f"mkdir -p {shlex.quote(self.base_path)}", timeout=self.timeout)
            if not result.is_success:
                logger.warning(f"BedrockAgentCoreRuntimeSandbox failed to create directory: {result.error_text}")
        except Exception as e:
            logger.warning(f"BedrockAgentCoreRuntimeSandbox failed to create directory: {e}")

    def _invoke_with_retry(self, command: str, timeout: int | None = None) -> AgentCoreRuntimeCommandResult:
        """Invoke a command, backing off on throttling and session-provisioning conflicts.

        ``RetryableConflictException`` fires while a session provisions or tears down, which
        makes it the most likely cause of a flaky first call against a new session.
        """
        cfg = self.creation_error_handling

        @retry(
            retry=retry_if_exception_type((AgentCoreRuntimeThrottlingError, AgentCoreRuntimeConflictError)),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(
                initial=cfg.initial_wait_seconds,
                max=cfg.max_wait_seconds,
                exp_base=cfg.exponential_base,
                jitter=cfg.jitter,
            ),
            reraise=True,
        )
        def invoke() -> AgentCoreRuntimeCommandResult:
            return self._get_client().invoke_command(
                agent_runtime_arn=self.agent_runtime_arn,
                session_id=self.session_id,
                command=command,
                timeout=timeout if timeout is not None else self.timeout,
                qualifier=self.qualifier,
            )

        return invoke()

    def create_view(self, base_path: str, sandbox_id: str | None = None) -> "BedrockAgentCoreRuntimeSandbox":
        """Return a sibling bound to the same session but a different base path."""
        return BedrockAgentCoreRuntimeSandbox(
            connection=self.connection,
            agent_runtime_arn=self.agent_runtime_arn,
            qualifier=self.qualifier,
            session_id=sandbox_id or self.ensure_started(),
            base_path=base_path,
            session_env=self.session_env,
            session_env_param=self.session_env_param,
            session_env_path=self.session_env_path,
            timeout=self.timeout,
            s3_relay_bucket=self.s3_relay_bucket,
            s3_relay_prefix=self.s3_relay_prefix,
            inline_transfer_max_bytes=self.inline_transfer_max_bytes,
            creation_error_handling=self.creation_error_handling,
            max_output_files=self.max_output_files,
        )

    def run_command_shell(
        self,
        command: str,
        timeout: int = 60,
        run_in_background_enabled: bool = False,
    ) -> ShellCommandResult:
        """Execute a shell command in the AgentCore Runtime session.

        Commands run from the container's own working directory; ``base_path`` is applied to
        file operations via :meth:`_resolve_path`, not via ``cd``, so callers passing
        already-resolved paths are not double-prefixed.

        Background execution has no native equivalent on this API — there is no
        ``startCommandExecution`` — so it is emulated with ``nohup … &``, which returns as
        soon as the command is detached and therefore captures no output.

        Only this path applies the session-env prefix. The internal callers that also reach
        the wire through :meth:`_invoke_with_retry` — directory setup, transfers, listing —
        need no user env and are already tight against the 64 KB command budget.
        """
        self.ensure_started()
        logger.debug(f"BedrockAgentCoreRuntimeSandbox running command: {command[:100]}...")

        prefix = build_env_prefix(self.session_env_path) if self._has_session_env else ""

        try:
            if run_in_background_enabled:
                # The prefix sits outside ``nohup`` so that ``&`` still detaches only the
                # user's command; the env is sourced first, in the foreground.
                detached = f"{prefix}nohup {command} > /dev/null 2>&1 &"
                result = self._invoke_with_retry(detached, timeout=timeout)
                if not result.is_success:
                    return ShellCommandResult(error=result.error_text)
                return ShellCommandResult(background=True)

            result = self._invoke_with_retry(f"{prefix}{command}", timeout=timeout)
            if result.timed_out:
                return ShellCommandResult(
                    stdout=result.stdout,
                    stderr=result.stderr or None,
                    exit_code=result.exit_code,
                    error=f"Command timed out after {timeout}s",
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
        """Upload a file, inline for small payloads and via the S3 relay for large ones."""
        self.ensure_started()

        if destination_path is None:
            destination_path = f"{self.base_path.rstrip('/')}/{file_name}"
        destination_path = self._resolve_path(destination_path)

        if ensure_parent_dirs and "/" in destination_path:
            parent = destination_path.rsplit("/", 1)[0]
            if parent:
                try:
                    self._invoke_with_retry(f"mkdir -p {shlex.quote(parent)}")
                except Exception as e:
                    logger.warning(f"BedrockAgentCoreRuntimeSandbox mkdir -p for {parent!r} failed (continuing): {e}")

        if len(content) <= self.inline_transfer_max_bytes:
            self._upload_inline(destination_path, content)
        else:
            self._upload_via_s3(destination_path, content)

        logger.debug(f"BedrockAgentCoreRuntimeSandbox uploaded file: {destination_path}")
        return destination_path

    def _upload_inline(self, destination_path: str, content: bytes) -> None:
        payload = base64.b64encode(content).decode("ascii")
        command = f"printf %s {shlex.quote(payload)} | base64 -d > {shlex.quote(destination_path)}"
        result = self._invoke_with_retry(command)
        if not result.is_success:
            raise RuntimeError(f"Failed to upload '{destination_path}': {result.error_text}")

    def _upload_via_s3(self, destination_path: str, content: bytes) -> None:
        """Relay a large file through S3: caller PUTs, sandbox downloads with curl."""
        if not self.s3_relay_bucket:
            raise ValueError(
                f"File is {len(content)} bytes, above the inline transfer limit of "
                f"{self.inline_transfer_max_bytes}. Set s3_relay_bucket on the sandbox to "
                f"enable large-file transfer."
            )
        key = f"{self.s3_relay_prefix.strip('/')}/{uuid.uuid4().hex}"
        s3 = self._get_s3_client()
        s3.put_object(Bucket=self.s3_relay_bucket, Key=key, Body=content)
        try:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3_relay_bucket, "Key": key},
                ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
            )
            result = self._invoke_with_retry(
                f"curl -fsS {shlex.quote(url)} -o {shlex.quote(destination_path)}",
                timeout=max(self.timeout, 300),
            )
            if not result.is_success:
                raise RuntimeError(f"Failed to relay '{destination_path}' through S3: {result.error_text}")
        finally:
            try:
                s3.delete_object(Bucket=self.s3_relay_bucket, Key=key)
            except Exception as e:
                logger.warning(f"BedrockAgentCoreRuntimeSandbox failed to clean up relay object {key}: {e}")

    def retrieve(self, file_path: str) -> bytes:
        """Read file bytes from the sandbox filesystem."""
        self.ensure_started()
        resolved_path = self._resolve_path(file_path)

        size_result = self._invoke_with_retry(f"stat -c %s {shlex.quote(resolved_path)} 2>/dev/null")
        if not size_result.is_success:
            raise FileNotFoundError(f"Failed to read file '{resolved_path}': {size_result.error_text}")
        try:
            size = int(size_result.stdout.strip())
        except (TypeError, ValueError) as e:
            raise FileNotFoundError(f"Failed to stat file '{resolved_path}': {size_result.stdout!r}") from e

        if size <= self.inline_transfer_max_bytes:
            return self._retrieve_inline(resolved_path)
        return self._retrieve_via_s3(resolved_path)

    def _retrieve_inline(self, resolved_path: str) -> bytes:
        result = self._invoke_with_retry(f"base64 -w0 {shlex.quote(resolved_path)}")
        if not result.is_success:
            raise FileNotFoundError(f"Failed to read file '{resolved_path}': {result.error_text}")
        try:
            return base64.b64decode(result.stdout.strip(), validate=True)
        except Exception as e:
            raise FileNotFoundError(f"Malformed base64 while reading '{resolved_path}': {e}") from e

    def _retrieve_via_s3(self, resolved_path: str) -> bytes:
        """Relay a large file back through S3: sandbox PUTs, caller downloads."""
        if not self.s3_relay_bucket:
            raise ValueError(
                f"File '{resolved_path}' is above the inline transfer limit of "
                f"{self.inline_transfer_max_bytes} bytes. Set s3_relay_bucket on the sandbox "
                f"to enable large-file transfer."
            )
        key = f"{self.s3_relay_prefix.strip('/')}/{uuid.uuid4().hex}"
        s3 = self._get_s3_client()
        url = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.s3_relay_bucket, "Key": key},
            ExpiresIn=PRESIGNED_URL_EXPIRY_SECONDS,
        )
        try:
            result = self._invoke_with_retry(
                f"curl -fsS -X PUT --upload-file {shlex.quote(resolved_path)} {shlex.quote(url)}",
                timeout=max(self.timeout, 300),
            )
            if not result.is_success:
                raise FileNotFoundError(f"Failed to relay '{resolved_path}' through S3: {result.error_text}")
            return s3.get_object(Bucket=self.s3_relay_bucket, Key=key)["Body"].read()
        finally:
            try:
                s3.delete_object(Bucket=self.s3_relay_bucket, Key=key)
            except Exception as e:
                logger.warning(f"BedrockAgentCoreRuntimeSandbox failed to clean up relay object {key}: {e}")

    def list_files(self, target_dir: str | None = None) -> list[str]:
        """List files in the sandbox directory."""
        self.ensure_started()
        if target_dir is None:
            target_dir = self.base_path
        target_dir = self._resolve_path(target_dir)

        try:
            command = f"find {shlex.quote(target_dir)} -maxdepth 3 -type f 2>/dev/null | head -{self.max_output_files}"
            result = self._invoke_with_retry(command)
            files = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            return files[: self.max_output_files]
        except Exception as e:
            logger.warning(f"BedrockAgentCoreRuntimeSandbox list_files failed for {target_dir}: {e}")
            return []

    def exists(self, file_path: str) -> bool:
        """Return True when the file exists in the sandbox filesystem."""
        try:
            self.ensure_started()
            resolved_path = self._resolve_path(file_path)
            result = self._invoke_with_retry(f"test -f {shlex.quote(resolved_path)}")
            return result.exit_code == 0
        except Exception as e:
            logger.debug(f"BedrockAgentCoreRuntimeSandbox exists({file_path}) failed (treating as missing): {e}")
            return False

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
        """Return sandbox metadata. AgentCore Runtime exposes no inbound ports."""
        return SandboxInfo(
            base_path=self.base_path,
            sandbox_id=self.session_id,
            public_url_error=(
                "AgentCore Runtime sessions are not addressable from outside; no public ports or URLs"
                if port is not None
                else None
            ),
        )

    def close(self, kill: bool = False) -> None:
        """Disconnect from the session.

        Args:
            kill: If False (default), keeps the session alive for reconnection — its compute
                stops on its own idle timeout and idle compute is not billed. If True, calls
                StopRuntimeSession, which flushes session storage durably but blocks for
                several seconds.
        """
        if not self._started:
            return
        try:
            if kill and self.session_id:
                self._get_client().stop_session(self.agent_runtime_arn, self.session_id, qualifier=self.qualifier)
                logger.debug(f"AgentCore Runtime session stopped: {self.session_id}")
                self.session_id = None
            else:
                logger.debug(f"AgentCore Runtime session disconnected (kept alive): {self.session_id}")
        except Exception as e:
            logger.warning(f"BedrockAgentCoreRuntimeSandbox close() failed: {e}")
        finally:
            self._started = False

    def __enter__(self):
        """Context manager entry."""
        self.ensure_started()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - attempt to disconnect on garbage collection."""
        try:
            self.close()
        except Exception:
            ...
