import io
import posixpath
import shlex
from typing import ClassVar

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import Field, PrivateAttr

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections.agentcore import (
    DEFAULT_CODE_INTERPRETER_IDENTIFIER,
    AgentCoreCodeInterpreterClient,
    AgentCoreSession,
    AgentCoreThrottlingError,
    normalize_sandbox_path,
)
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import DESCRIPTION_SANDBOX_INTERPRETER, BaseCodeInterpreterTool
from dynamiq.utils.logger import logger

AGENTCORE_DESCRIPTION_NOTE_HEADER = "-AgentCore Notes:-"


class BedrockAgentCoreInterpreterTool(BaseCodeInterpreterTool):
    """AWS Bedrock AgentCore Code Interpreter implementation of the sandbox interpreter tool.

    Executes code in AWS-managed microVM sandbox sessions via the ``bedrock-agentcore``
    data plane. Sessions are addressed by a code interpreter identifier (the built-in
    ``aws.codeinterpreter.v1`` by default, or a custom interpreter ID) and expire after
    ``timeout`` seconds of their fixed lifetime (capped at 8 hours by AWS). AgentCore has
    no per-invocation timeout and does not expose public ports; ``env``/``cwd`` for shell
    commands are emulated by composing ``cd``/``export`` into the command.

    Note: ``timeout`` sets the session time-to-live at creation; it cannot be changed for
    a live session and there is no per-``execute`` timeout, so the per-call
    ``input_data.timeout`` override is not applied to individual invocations.
    """

    name: str = "bedrock-agentcore-code-interpreter-tool"
    description: str = DESCRIPTION_SANDBOX_INTERPRETER
    base_path: str = "."
    # Session lifetime (time-to-live). Defaults higher than the base tool so a persistent
    # session is unlikely to hit its hard TTL between executions.
    timeout: int = Field(default=3600, description="Session lifetime in seconds (AgentCore caps it at 28800).")
    connection: AWSConnection | None = None
    code_interpreter_identifier: str = Field(
        default=DEFAULT_CODE_INTERPRETER_IDENTIFIER,
        description="AgentCore code interpreter identifier (built-in or custom interpreter ID).",
    )
    _client: AgentCoreCodeInterpreterClient | None = PrivateAttr(default=None)
    _rate_limit_exception: ClassVar[type[Exception]] = AgentCoreThrottlingError

    def __init__(self, **kwargs):
        if kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)
        self._append_agentcore_description_note()

    def _append_agentcore_description_note(self) -> None:
        """Add AgentCore-specific guidance to the agent-facing description.

        The shared description is written for VM-backed sandboxes rooted at an absolute
        home directory, so it tells the model to use absolute paths. AgentCore's file APIs
        resolve against the session workspace instead, and outputs are only collected from
        the output directory, so the model is told explicitly to keep paths relative.
        Guarded by a marker so a tool rebuilt from a serialized description (which already
        carries the note) does not accumulate copies.
        """
        if AGENTCORE_DESCRIPTION_NOTE_HEADER in self.description:
            return
        self.description = self.description.rstrip() + (
            f"\n\n{AGENTCORE_DESCRIPTION_NOTE_HEADER}\n"
            f"- Paths resolve against the session workspace, not the filesystem root. Always use "
            f"workspace-relative paths such as {self.input_dir}/data.csv and {self.output_dir}/result.csv, "
            f"even where the guidance above mentions absolute paths or the root directory.\n"
            f"- Only files written to {self.output_dir} are collected and returned to the user.\n"
            f"- The sandbox exposes no public URLs or ports, so return results as printed output or files."
        )

    def _has_connection_credentials(self) -> bool:
        # The boto3 default credential chain (env vars, profile, IMDS) may resolve
        # credentials even when the connection fields are unset, so allow eager
        # persistent initialization to attempt it (guarded by the override below).
        return True

    def _initialize_persistent_sandbox(self) -> None:
        """Eagerly start the persistent session, degrading gracefully on init errors.

        Unlike E2B/Daytona (which gate eager init on an ``api_key``), AWS credentials are
        resolved lazily by boto3, so a missing region/credentials would otherwise surface
        as a raw botocore error inside the constructor. Failures after the session was
        created (package install, file upload — raised as ``ToolExecutionException``) stop
        the partially initialized session so it does not linger until its AWS TTL. In both
        cases persistent mode is disabled so each ``execute`` transparently creates and
        stops its own session — this avoids both crashing construction and leaking sessions
        (the base ``execute`` would otherwise keep creating unstored, never-stopped sessions
        while ``persistent_sandbox`` stays True).
        """
        try:
            super()._initialize_persistent_sandbox()
        except (BotoCoreError, ClientError, AgentCoreThrottlingError, ToolExecutionException) as e:
            logger.warning(
                f"Tool {self.name} - {self.id}: Could not eagerly initialize persistent AgentCore "
                f"session ({e}); falling back to a fresh session per execution (non-persistent)."
            )
            if self._sandbox is not None:
                try:
                    self._get_client().stop_session(self._sandbox)
                except Exception as stop_error:
                    logger.warning(
                        f"Tool {self.name} - {self.id}: Failed to stop partially initialized " f"session: {stop_error}"
                    )
            self._sandbox = None
            self.persistent_sandbox = False

    def _get_client(self) -> AgentCoreCodeInterpreterClient:
        if self._client is None:
            self._client = AgentCoreCodeInterpreterClient(self.connection)
        return self._client

    def _create_sandbox(self) -> AgentCoreSession:
        return self._get_client().start_session(
            identifier=self.code_interpreter_identifier,
            session_timeout_seconds=self.timeout,
        )

    def _get_sandbox_id(self, sandbox: AgentCoreSession) -> str:
        return sandbox.session_id

    def _get_sandbox_host(self, sandbox: AgentCoreSession) -> str | None:
        return None

    def _execute_python_code(
        self, code: str, sandbox: AgentCoreSession, params: dict | None = None, timeout: int | None = None
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")

        if params:
            vars_code = "\n# Tool params variables\n"
            for key, value in params.items():
                resolved = self._resolve_param_value(value, sandbox)
                vars_code += f"{key} = {repr(resolved)}\n"
            code = vars_code + "\n" + code

        try:
            logger.info(f"Executing Python code: {code}")
            result = self._get_client().invoke(sandbox, "executeCode", {"code": code, "language": "python"})

            if result.is_error:
                error_text = result.error_text
                if "NameError" in error_text and self.persistent_sandbox:
                    logger.debug(f"Tool {self.name}: Recoverable NameError in persistent session: {error_text}")
                raise ToolExecutionException(
                    f"Error during Python code execution: {error_text}",
                    recoverable=True,
                )

            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"[stderr] {result.stderr}")

            return "\n".join(output_parts) if output_parts else ""

        except ToolExecutionException:
            raise
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)

    def _compose_shell_command(self, command: str, env: dict | None = None, cwd: str | None = None) -> str:
        """Compose cd/export into the command since AgentCore executeCommand has no env/cwd args."""
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
        sandbox: AgentCoreSession,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            composed = self._compose_shell_command(command, env=env, cwd=cwd)
            result = self._get_client().invoke(sandbox, "executeCommand", {"command": composed})
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        if result.is_error or result.exit_code != 0:
            raise ToolExecutionException(f"Error during shell command execution: {result.error_text}", recoverable=True)
        return result.stdout

    def _install_packages(self, sandbox: AgentCoreSession, packages: str) -> None:
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            # Quote each requirement: executeCommand runs through a shell, so an unquoted
            # version pin like "pandas>=2.0" would be parsed as a redirection and silently
            # install the latest version instead. Split on commas and whitespace so both
            # "a,b" and "a b" work; shlex.quote leaves plain names untouched.
            requirements = [shlex.quote(pkg) for pkg in packages.replace(",", " ").split()]
            if not requirements:
                return
            try:
                result = self._get_client().invoke(
                    sandbox, "executeCommand", {"command": f"pip install -qq {' '.join(requirements)}"}
                )
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if result.is_error or result.exit_code != 0:
                raise ToolExecutionException(
                    f"Error during package installation: {result.error_text}", recoverable=True
                )

    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: AgentCoreSession) -> str:
        normalized = normalize_sandbox_path(target_path)
        # Transport-level failures (throttling, botocore errors) are wrapped as recoverable,
        # matching the code/shell/package paths: agents only retry RecoverableAgentException,
        # so a raw error here would abort the run instead of allowing another attempt.
        try:
            parent = posixpath.dirname(normalized)
            if parent and parent != ".":
                # AgentCore writeFiles does not create parent directories; mkdir -p is idempotent.
                self._get_client().invoke(sandbox, "executeCommand", {"command": f"mkdir -p {shlex.quote(parent)}"})
            content = {"path": normalized, "blob": file.read()}
            result = self._get_client().invoke(sandbox, "writeFiles", {"content": [content]})
        except Exception as e:
            raise ToolExecutionException(f"Error during file upload: {e}", recoverable=True)

        if result.is_error:
            raise ToolExecutionException(f"Error during file upload: {result.error_text}", recoverable=True)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file to: {normalized}")
        return normalized

    def _download_file_bytes(self, file_path: str, sandbox: AgentCoreSession) -> bytes:
        client = self._get_client()
        try:
            result = client.invoke(sandbox, "readFiles", {"paths": [normalize_sandbox_path(file_path)]})
        except Exception as e:
            raise ToolExecutionException(f"Error during file download: {e}", recoverable=True)

        if result.is_error:
            raise ToolExecutionException(f"Error during file download: {result.error_text}", recoverable=True)
        try:
            return client.extract_file_bytes(result)
        except FileNotFoundError as e:
            raise ToolExecutionException(f"Error during file download: {e}", recoverable=True)

    def _run_shell_command(self, command: str, sandbox: AgentCoreSession) -> tuple[int, str]:
        result = self._get_client().invoke(sandbox, "executeCommand", {"command": command})
        return result.exit_code, result.stdout or ""

    def _reconnect_sandbox(self, sandbox_id: str) -> AgentCoreSession:
        status = self._get_client().get_session_status(self.code_interpreter_identifier, sandbox_id)
        if status != "READY":
            raise ValueError(
                f"AgentCore session {sandbox_id} is {status or 'unavailable'}; "
                "terminated sessions cannot be restarted"
            )
        return AgentCoreSession(identifier=self.code_interpreter_identifier, session_id=sandbox_id)

    def _destroy_sandbox(self, sandbox: AgentCoreSession) -> None:
        self._get_client().stop_session(sandbox)

    def set_timeout(self, timeout: int) -> None:
        """Update the tool-level timeout. AgentCore fixes the session timeout at start."""
        super().set_timeout(timeout)
        if self._sandbox and self.persistent_sandbox:
            logger.debug(
                f"Tool {self.name} - {self.id}: AgentCore session timeout is fixed at session start; "
                f"new value {timeout}s applies to future sessions"
            )

    def close(self) -> None:
        """Close the persistent sandbox session if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            try:
                self._get_client().stop_session(self._sandbox)
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to stop session: {e}")
            self._sandbox = None
