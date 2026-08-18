"""Interpreter tool backed by the AWS Bedrock AgentCore **Runtime** API.

AgentCore's other API, Code Interpreter, is a different service behind the same boto3 client
name. Two differences from it shape this implementation:

- **There is no ``executeCode`` primitive.** Python is run by writing the source into the
  session and invoking ``python3`` on it, which also means large programs transfer through
  the same inline/S3 path as any other file.
- **There is a real per-invocation timeout**, so ``input_data.timeout`` is honoured per
  execution instead of being fixed for the session's lifetime.

File transfer and command execution are delegated to
:class:`~dynamiq.sandboxes.bedrock_agentcore_runtime.BedrockAgentCoreRuntimeSandbox` so the
inline-base64/S3-relay switch lives in exactly one place.
"""

from __future__ import annotations

import io
import shlex
import uuid
from typing import TYPE_CHECKING, ClassVar

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field, PrivateAttr

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import DESCRIPTION_SANDBOX_INTERPRETER, BaseCodeInterpreterTool
from dynamiq.sandboxes.bedrock_agentcore_runtime_client import (
    DEFAULT_BASE_PATH,
    DEFAULT_INLINE_TRANSFER_MAX_BYTES,
    DEFAULT_SESSION_ENV_PATH,
    AgentCoreRuntimeConflictError,
    AgentCoreRuntimeThrottlingError,
)
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.sandboxes.bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox

AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER = "-AgentCore Runtime Notes:-"


class BedrockAgentCoreRuntimeInterpreterTool(BaseCodeInterpreterTool):
    """AWS Bedrock AgentCore Runtime implementation of the sandbox interpreter tool.

    Executes code and shell commands inside an AWS-managed microVM session addressed by
    ``agent_runtime_arn`` + a caller-chosen session ID. The runtime is provisioned
    out-of-band, mirroring how E2B templates are built.

    Session storage persists across the session's idle timeout and across an explicit stop,
    so a workspace survives between executions. AgentCore exposes no inbound ports, so no
    public URLs are available.
    """

    name: str = "bedrock-agentcore-runtime-interpreter-tool"
    description: str = DESCRIPTION_SANDBOX_INTERPRETER
    base_path: str = DEFAULT_BASE_PATH
    timeout: int = Field(default=300, description="Per-command timeout in seconds (AgentCore allows 1..3600).")
    connection: AWSConnection | None = None
    agent_runtime_arn: str = Field(description="ARN of the provisioned AgentCore agent runtime.")
    qualifier: str | None = Field(
        default=None,
        description="Endpoint alias pinning a runtime version. Omitted means the default endpoint.",
    )
    session_seed: str | None = Field(
        default=None,
        description="Seed for deterministic session-ID derivation (typically a conversation ID).",
    )
    session_env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Environment variables sourced by every shell command in the session. The values "
            "travel inside a command string and are present in a serialized node config, so "
            "prefer session_env_param for secrets."
        ),
    )
    session_env_param: str | None = Field(
        default=None,
        description=(
            "SSM Parameter Store parameter holding the env-file content, fetched from inside "
            "the container. Only the name is ever sent. Takes precedence over session_env."
        ),
    )
    session_env_path: str = Field(
        default=DEFAULT_SESSION_ENV_PATH,
        description="Where the session env file lives; shared by every view of the session.",
    )
    s3_relay_bucket: str | None = Field(
        default=None, description="Bucket used to relay files above the inline transfer limit."
    )
    inline_transfer_max_bytes: int = Field(
        default=DEFAULT_INLINE_TRANSFER_MAX_BYTES,
        description="Files at or below this size transfer inline as base64; larger ones use the S3 relay.",
    )
    _rate_limit_exception: ClassVar[type[Exception]] = AgentCoreRuntimeThrottlingError
    _initialized_persistent: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        if kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        # The base class gates eager initialization on `connection.api_key`, which AWS
        # connections do not have (boto3 resolves credentials lazily from env, profile or
        # IMDS). Suppress the base's eager path and run our own guarded version below.
        persistent = kwargs.get("persistent_sandbox", False)
        kwargs["persistent_sandbox"] = False
        super().__init__(**kwargs)
        self.persistent_sandbox = persistent
        self._append_runtime_description_note()
        if persistent:
            self._initialize_persistent_sandbox_safely()

    @property
    def to_dict_exclude_secure_params(self) -> dict:
        """Treat ``session_env`` as a credential, alongside ``connection``.

        It is therefore absent from tracing and logging dumps, and present only where
        secure params are explicitly requested — which includes workflow serialization
        (``Flow.to_dict`` defaults ``include_secure_params=True``), since the executing
        process has to receive the values. Use ``session_env_param`` where that payload
        crosses a trust boundary.
        """
        return super().to_dict_exclude_secure_params | {"session_env": True}

    def _append_runtime_description_note(self) -> None:
        """Add backend-specific guidance to the agent-facing description.

        Guarded by a marker so a tool rebuilt from a serialized description (which already
        carries the note) does not accumulate copies.
        """
        if AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER in self.description:
            return
        self.description = self.description.rstrip() + (
            f"\n\n{AGENTCORE_RUNTIME_DESCRIPTION_NOTE_HEADER}\n"
            f"- The workspace at {self.base_path} persists between executions in this session.\n"
            f"- Only files written to {self.output_dir} are collected and returned to the user.\n"
            f"- The sandbox exposes no public URLs or ports, so return results as printed output or files."
        )

    def _initialize_persistent_sandbox_safely(self) -> None:
        """Eagerly start the persistent session, degrading to per-execution sessions on failure.

        AWS credentials resolve lazily, so a missing region or expired token would otherwise
        surface as a raw botocore error inside the constructor. On failure the partially
        initialized session is closed and persistent mode is disabled, so each ``execute``
        transparently creates its own session rather than leaking unstored ones.
        """
        try:
            self._initialize_persistent_sandbox()
            self._initialized_persistent = True
        except (
            BotoCoreError,
            ClientError,
            AgentCoreRuntimeThrottlingError,
            AgentCoreRuntimeConflictError,
            ToolExecutionException,
        ) as e:
            logger.warning(
                f"Tool {self.name} - {self.id}: Could not eagerly initialize persistent AgentCore Runtime "
                f"session ({e}); falling back to a fresh session per execution (non-persistent)."
            )
            if self._sandbox is not None:
                try:
                    self._sandbox.close(kill=True)
                except Exception as stop_error:
                    logger.warning(
                        f"Tool {self.name} - {self.id}: Failed to stop partially initialized session: {stop_error}"
                    )
            self._sandbox = None
            self.persistent_sandbox = False

    def _create_sandbox(self) -> BedrockAgentCoreRuntimeSandbox:
        # Imported lazily: dynamiq.sandboxes imports this package through
        # code_interpreter, so a module-scope import would be circular.
        from dynamiq.sandboxes.bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox

        sandbox = BedrockAgentCoreRuntimeSandbox(
            connection=self.connection,
            agent_runtime_arn=self.agent_runtime_arn,
            qualifier=self.qualifier,
            session_seed=self.session_seed,
            base_path=self.base_path,
            session_env=self.session_env,
            session_env_param=self.session_env_param,
            session_env_path=self.session_env_path,
            timeout=self.timeout,
            s3_relay_bucket=self.s3_relay_bucket,
            inline_transfer_max_bytes=self.inline_transfer_max_bytes,
            creation_error_handling=self.creation_error_handling,
        )
        sandbox.ensure_started()
        return sandbox

    def _get_sandbox_id(self, sandbox: BedrockAgentCoreRuntimeSandbox) -> str:
        return sandbox.session_id

    def _get_sandbox_host(self, sandbox: BedrockAgentCoreRuntimeSandbox) -> str | None:
        return None

    def _run(self, command: str, sandbox: BedrockAgentCoreRuntimeSandbox, timeout: int | None = None):
        return sandbox.run_command_shell(command, timeout=timeout if timeout is not None else self.timeout)

    def _execute_python_code(
        self,
        code: str,
        sandbox: BedrockAgentCoreRuntimeSandbox,
        params: dict | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run Python by materializing it as a file — Runtime has no ``executeCode``.

        Uploading rather than inlining the source into the command keeps large programs
        working: the upload path switches to the S3 relay above the inline limit, whereas an
        inlined program would hit the 64 KB command ceiling.
        """
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")

        if params:
            vars_code = "\n# Tool params variables\n"
            for key, value in params.items():
                resolved = self._resolve_param_value(value, sandbox)
                vars_code += f"{key} = {repr(resolved)}\n"
            code = vars_code + "\n" + code

        script_path = f"{self.base_path.rstrip('/')}/.dynamiq_exec_{uuid.uuid4().hex}.py"
        try:
            logger.info(f"Executing Python code: {code}")
            sandbox.upload_file(script_path.rsplit("/", 1)[-1], code.encode("utf-8"), destination_path=script_path)
            result = self._run(f"python3 {shlex.quote(script_path)}", sandbox, timeout)
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)
        finally:
            try:
                self._run(f"rm -f {shlex.quote(script_path)}", sandbox)
            except Exception as cleanup_error:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to remove {script_path}: {cleanup_error}")

        if not result.is_success:
            raise ToolExecutionException(
                f"Error during Python code execution: {result.error or result.stderr or result.stdout}",
                recoverable=True,
            )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr] {result.stderr}")
        return "\n".join(output_parts) if output_parts else ""

    def _compose_shell_command(self, command: str, env: dict | None = None, cwd: str | None = None) -> str:
        """Compose cd/export into the command; the API takes no env or cwd arguments.

        The composed string is interpreted by a shell only because the client wraps every
        command in ``bash -c`` — see ``wrap_shell_command``. Without that wrapper this
        ``&&``-joined form would be passed to ``argv`` verbatim and silently misbehave.
        """
        parts = []
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        for key, value in (env or {}).items():
            parts.append(f"export {key}={shlex.quote(str(value))}")
        parts.append(command)
        return " && ".join(parts)

    def _execute_shell_command(
        self,
        command: str,
        sandbox: BedrockAgentCoreRuntimeSandbox,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            composed = self._compose_shell_command(command, env=env, cwd=cwd)
            result = self._run(composed, sandbox, timeout)
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        if not result.is_success:
            raise ToolExecutionException(
                f"Error during shell command execution: {result.error or result.stderr or result.stdout}",
                recoverable=True,
            )
        return result.stdout or ""

    def _install_packages(self, sandbox: BedrockAgentCoreRuntimeSandbox, packages: str) -> None:
        if not packages:
            return
        logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
        # Quote each requirement: the command is shell-interpreted, so an unquoted version
        # pin like "pandas>=2.0" would be parsed as a redirection and silently install the
        # latest version instead. Split on commas and whitespace so both "a,b" and "a b"
        # work; shlex.quote leaves plain names untouched.
        requirements = [shlex.quote(pkg) for pkg in packages.replace(",", " ").split()]
        if not requirements:
            return
        # `pip3` is not on PATH in a slim base image, but `python3 -m pip` always is.
        try:
            result = self._run(f"python3 -m pip install -qq {' '.join(requirements)}", sandbox, timeout=self.timeout)
        except Exception as e:
            raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

        if not result.is_success:
            raise ToolExecutionException(
                f"Error during package installation: {result.error or result.stderr or result.stdout}",
                recoverable=True,
            )

    def _upload_file_to_sandbox(
        self, file: io.BytesIO, target_path: str, sandbox: BedrockAgentCoreRuntimeSandbox
    ) -> str:
        try:
            return sandbox.upload_file(target_path.rsplit("/", 1)[-1], file.read(), destination_path=target_path)
        except Exception as e:
            raise ToolExecutionException(f"Error during file upload: {e}", recoverable=True)

    def _download_file_bytes(self, file_path: str, sandbox: BedrockAgentCoreRuntimeSandbox) -> bytes:
        try:
            return sandbox.retrieve(file_path)
        except Exception as e:
            raise ToolExecutionException(f"Error during file download: {e}", recoverable=True)

    def _run_shell_command(self, command: str, sandbox: BedrockAgentCoreRuntimeSandbox) -> tuple[int, str]:
        result = self._run(command, sandbox)
        exit_code = result.exit_code if result.exit_code is not None else 1
        return exit_code, result.stdout or ""

    def _reconnect_sandbox(self, sandbox_id: str) -> BedrockAgentCoreRuntimeSandbox:
        """Reconnect by session ID.

        Unlike the Code Interpreter backend there is no session status to check: a session
        is created implicitly on first use, so an expired ID simply yields a fresh session
        with empty storage rather than an error.
        """
        # Imported lazily: dynamiq.sandboxes imports this package through
        # code_interpreter, so a module-scope import would be circular.
        from dynamiq.sandboxes.bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox

        sandbox = BedrockAgentCoreRuntimeSandbox(
            connection=self.connection,
            agent_runtime_arn=self.agent_runtime_arn,
            qualifier=self.qualifier,
            session_id=sandbox_id,
            base_path=self.base_path,
            session_env=self.session_env,
            session_env_param=self.session_env_param,
            session_env_path=self.session_env_path,
            timeout=self.timeout,
            s3_relay_bucket=self.s3_relay_bucket,
            inline_transfer_max_bytes=self.inline_transfer_max_bytes,
            creation_error_handling=self.creation_error_handling,
        )
        sandbox.ensure_started()
        return sandbox

    def _destroy_sandbox(self, sandbox: BedrockAgentCoreRuntimeSandbox) -> None:
        sandbox.close(kill=True)

    def close(self) -> None:
        """Close the persistent sandbox session if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            try:
                self._sandbox.close(kill=True)
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to stop session: {e}")
            self._sandbox = None
