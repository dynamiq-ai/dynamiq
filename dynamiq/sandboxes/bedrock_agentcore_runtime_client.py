"""Data-plane client for the AWS Bedrock AgentCore **Runtime** API.

Credentials come from the shared :class:`~dynamiq.connections.AWS` connection; this module
only owns the wire protocol, so it lives beside the sandbox rather than under
``connections/``. AgentCore's Runtime and Code Interpreter APIs share a boto3 client name
(``bedrock-agentcore``) but nothing else: the envelope, the errors and the file-transfer
story are all different.

Behaviours below were established empirically against the live API and are not obvious
from the AWS documentation:

- ``body.command`` is tokenised and ``exec``'d directly into the container. It is **not**
  interpreted by a shell, despite the API docs saying "run shell commands". Every command
  is therefore wrapped as ``bash -c '<command>'``; see :func:`wrap_shell_command`.
- ``timeout`` is validated server-side at 1..3600 and rejects out-of-range values with a
  ``ValidationException`` rather than coercing them, so it is clamped here.
- A timed-out command reports ``exitCode: -1`` with ``status: TIMED_OUT`` (not 124, which
  is the coreutils ``timeout`` convention the Code Interpreter backend relies on).
- Errors arrive both as raised exceptions and as modeled members inside the event stream.
"""

import hashlib
import posixpath
import re
import shlex
import uuid

from botocore.config import Config
from pydantic import BaseModel, Field

from dynamiq.connections import AWS as AWSConnection
from dynamiq.utils.logger import logger

# Hard service limits (verified against the live API 2026-08-13).
MAX_COMMAND_BYTES = 65536
MIN_COMMAND_TIMEOUT_SECONDS = 1
MAX_COMMAND_TIMEOUT_SECONDS = 3600
DEFAULT_COMMAND_TIMEOUT_SECONDS = 300
MIN_SESSION_ID_LENGTH = 33
MAX_SESSION_ID_LENGTH = 256

# Sandbox defaults. They live here rather than in the sandbox module so the node can import
# them without importing the sandbox module itself — see that module's import note.
DEFAULT_BASE_PATH = "/mnt/workspace"
# The session env file lives on the session-storage mount rather than under ``base_path``,
# because views of one session differ only in base_path and must all see the same file.
DEFAULT_SESSION_ENV_PATH = f"{DEFAULT_BASE_PATH}/.dqenv"
# Inline base64 costs 4/3 in size and must leave room for the bash -c wrapper and the
# mkdir/decode boilerplate inside the 64 KB command budget.
DEFAULT_INLINE_TRANSFER_MAX_BYTES = 40_000

# Transient AWS error codes, matched case-insensitively so both the REST codes
# (``ThrottlingException``) and the event-stream member names delivered mid-invocation
# (``throttlingException``) are covered.
RETRYABLE_ERROR_CODES = frozenset(
    code.lower()
    for code in (
        "ThrottlingException",
        "TooManyRequestsException",
        "ThrottledException",
        "ServiceQuotaExceededException",
    )
)
# 409 while a session provisions or tears down. The service explicitly asks for backoff.
CONFLICT_ERROR_CODES = frozenset(code.lower() for code in ("RetryableConflictException", "ConflictException"))
RESOURCE_NOT_FOUND_CODE = "resourcenotfoundexception"

_UNSAFE_SESSION_ID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")


class AgentCoreRuntimeThrottlingError(Exception):
    """Raised when AWS throttles or quota-limits an AgentCore Runtime request."""


class AgentCoreRuntimeConflictError(Exception):
    """Raised on ``RetryableConflictException`` — a session is provisioning or tearing down."""


class CommandTooLargeError(ValueError):
    """Raised when a command exceeds the 64 KB service limit, before it is sent."""


def _error_code(error: Exception) -> str:
    """Extract the AWS error code from a botocore error, tolerating a ``None`` response."""
    response = getattr(error, "response", None) or {}
    return (response.get("Error") or {}).get("Code", "") or ""


def wrap_shell_command(command: str) -> str:
    """Wrap a command so it is interpreted by a shell.

    ``InvokeAgentRuntimeCommand`` splits ``body.command`` into argv and execs it directly
    (containerd ``ctr exec``) — no shell is involved. Unwrapped, ``echo a; echo b`` prints
    the literal ``a; echo b``, pipes and redirects are inert, ``$VAR`` is not expanded, and
    a command whose ``argv[0]`` is a shell builtin or missing binary fails with an opaque
    ``ctr: OCI runtime exec failed`` instead of 127.

    The wrapper costs ~10 bytes against the 64 KB command budget.
    """
    return f"bash -c {shlex.quote(command)}"


def format_env_file(env: dict[str, str]) -> str:
    """Render ``env`` as shell-sourceable ``KEY='value'`` lines.

    Values are ``shlex.quote``d, so a value containing quotes, ``$``, ``;`` or a newline
    survives ``set -a; . file`` without breaking out of its assignment.

    Exported because the writer of the env file is not always this process: with
    :func:`build_env_bootstrap_command`'s ``param_name`` form the content is produced by
    whoever populates the parameter, and it has to agree on this format.
    """
    return "".join(f"{key}={shlex.quote(str(value))}\n" for key, value in env.items())


def build_env_bootstrap_command(
    path: str,
    env: dict[str, str] | None = None,
    param_name: str | None = None,
    region: str | None = None,
) -> str:
    """Build the command that (re)writes the session env file.

    Two ways to source the content, and the choice decides where the secret is exposed:

    - ``env`` inlines the values, so they appear in this one command string — and therefore
      in whatever logs AgentCore keeps of it.
    - ``param_name`` fetches from SSM Parameter Store using the runtime's execution role, so
      only the parameter *name* ever appears in a command string. Prefer it for secrets.

    ``param_name`` wins if both are given.

    The write is atomic (``> tmp && mv``) because views of one session share this file and
    each holds its own lock, so nothing serialises two concurrent bootstraps against each
    other; a reader must never observe a half-written file. ``$$`` is left unquoted so each
    invocation gets its own temp path.
    """
    if param_name:
        producer = (
            f"aws ssm get-parameter --name {shlex.quote(param_name)} --with-decryption "
            f"--query Parameter.Value --output text"
        )
        if region:
            producer += f" --region {shlex.quote(region)}"
    else:
        producer = f"printf '%s' {shlex.quote(format_env_file(env or {}))}"

    quoted_path = shlex.quote(path)
    tmp_path = f"{quoted_path}.tmp.$$"
    parent = posixpath.dirname(path)
    mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent and parent != "/" else ""
    return f"umask 077 && {mkdir}{producer} > {tmp_path} && chmod 600 {tmp_path} && mv {tmp_path} {quoted_path}"


def build_env_prefix(path: str) -> str:
    """Build the prefix that sources the session env file into a command's environment.

    ``set -a`` marks the assignments for export so child processes inherit them; the
    existence test keeps a command runnable in a session whose bootstrap never happened,
    rather than failing it with a confusing "No such file" from ``.``.

    Costs ~60 bytes of the 64 KB command budget. That is accounted for automatically:
    :meth:`AgentCoreRuntimeClient.invoke_command` measures the *wrapped* payload, and the
    prefix is part of the command by then.
    """
    quoted = shlex.quote(path)
    return f"set -a; [ -f {quoted} ] && . {quoted}; set +a; "


def clamp_command_timeout(timeout: int | None) -> int:
    """Clamp to the service's 1..3600 range; out-of-range is a hard 400, not a coerce."""
    if timeout is None:
        return DEFAULT_COMMAND_TIMEOUT_SECONDS
    return max(MIN_COMMAND_TIMEOUT_SECONDS, min(int(timeout), MAX_COMMAND_TIMEOUT_SECONDS))


def derive_session_id(seed: str) -> str:
    """Derive a deterministic, service-valid ``runtimeSessionId`` from an arbitrary seed.

    AgentCore lets the caller choose the session id (33–256 chars), which inverts E2B's
    model: rather than storing an assigned id, the id is recomputed from the conversation,
    so reconnection survives a process restart and needs no mapping table.

    A hash suffix is appended rather than padding with zeros so that distinct seeds cannot
    collide after sanitisation (``a/b`` and ``a-b`` would otherwise both become ``a-b``).
    """
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]
    sanitized = _UNSAFE_SESSION_ID_CHARS.sub("-", seed).strip("-")
    candidate = f"dq-{sanitized}-{digest}" if sanitized else f"dq-{digest}"
    if len(candidate) > MAX_SESSION_ID_LENGTH:
        # Keep the hash (which carries the identity) and trim the human-readable prefix.
        keep = MAX_SESSION_ID_LENGTH - len(digest) - 4
        candidate = f"dq-{sanitized[:keep]}-{digest}"
    return candidate.ljust(MIN_SESSION_ID_LENGTH, "0")


def normalize_sandbox_path(path: str) -> str:
    """Normalize a path for the Runtime backend.

    Unlike the Code Interpreter backend, Runtime paths are ordinary absolute filesystem
    paths inside the container, so ``/`` is preserved rather than collapsed to ``.``.
    """
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return "."
    return normalized


class AgentCoreRuntimeCommandResult(BaseModel):
    """Normalized result of one ``InvokeAgentRuntimeCommand`` invocation."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    status: str | None = None

    @property
    def timed_out(self) -> bool:
        return self.status == "TIMED_OUT"

    @property
    def is_success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def error_text(self) -> str:
        if self.timed_out:
            return self.stderr or "Command timed out"
        return self.stderr or self.stdout or "Unknown execution error"


class AgentCoreRuntimeSession(BaseModel):
    """Identity of a Runtime session: everything needed to address it again."""

    agent_runtime_arn: str
    session_id: str
    qualifier: str | None = Field(
        default=None,
        description="Endpoint alias pinning a runtime version. Persist it per conversation.",
    )


class AgentCoreRuntimeClient:
    """Thin wrapper over the boto3 ``bedrock-agentcore`` Runtime data plane."""

    def __init__(self, connection: AWSConnection, read_timeout: int = 3660):
        session = connection.get_boto3_session()
        # Disable botocore's built-in retries: InvokeAgentRuntimeCommand carries no
        # idempotency token, so a transient socket error must not silently re-run user
        # code. Retryable conditions are handled at the application level instead.
        self._client = session.client(
            "bedrock-agentcore",
            config=Config(read_timeout=read_timeout, retries={"total_max_attempts": 1}),
        )

    @staticmethod
    def _translate_error(error: Exception) -> Exception:
        code = _error_code(error).lower()
        if code in RETRYABLE_ERROR_CODES:
            return AgentCoreRuntimeThrottlingError(str(error))
        if code in CONFLICT_ERROR_CODES:
            return AgentCoreRuntimeConflictError(str(error))
        return error

    @staticmethod
    def _raise_for_stream_error(member: str, payload: dict) -> None:
        """Translate a modeled error member delivered inside the event stream."""
        message = (payload or {}).get("message") or member
        lowered = member.lower()
        if lowered in RETRYABLE_ERROR_CODES:
            raise AgentCoreRuntimeThrottlingError(message)
        if lowered in CONFLICT_ERROR_CODES:
            raise AgentCoreRuntimeConflictError(message)
        raise RuntimeError(f"AgentCore Runtime {member}: {message}")

    def invoke_command(
        self,
        agent_runtime_arn: str,
        session_id: str,
        command: str,
        timeout: int | None = None,
        qualifier: str | None = None,
    ) -> AgentCoreRuntimeCommandResult:
        """Run one shell command in a Runtime session and collect its streamed output.

        The session is created implicitly on first use — no bootstrap ``InvokeAgentRuntime``
        call is required.
        """
        payload = wrap_shell_command(command)
        encoded_length = len(payload.encode("utf-8"))
        if encoded_length > MAX_COMMAND_BYTES:
            raise CommandTooLargeError(
                f"Command is {encoded_length} bytes after shell wrapping, over the AgentCore "
                f"limit of {MAX_COMMAND_BYTES}. Write the payload to a file instead "
                f"(the sandbox switches to an S3 relay above its inline transfer threshold)."
            )

        params = {
            "agentRuntimeArn": agent_runtime_arn,
            "runtimeSessionId": session_id,
            "contentType": "application/json",
            "accept": "application/vnd.amazon.eventstream",
            "body": {"command": payload, "timeout": clamp_command_timeout(timeout)},
        }
        if qualifier:
            params["qualifier"] = qualifier

        try:
            response = self._client.invoke_agent_runtime_command(**params)
            result = AgentCoreRuntimeCommandResult()
            stdout: list[str] = []
            stderr: list[str] = []
            for event in response.get("stream", []):
                chunk = event.get("chunk") or {}
                for member, member_payload in chunk.items():
                    if member == "contentDelta":
                        if out := (member_payload or {}).get("stdout"):
                            stdout.append(out)
                        if err := (member_payload or {}).get("stderr"):
                            stderr.append(err)
                    elif member == "contentStop":
                        result.exit_code = (member_payload or {}).get("exitCode")
                        result.status = (member_payload or {}).get("status")
                    elif member != "contentStart":
                        self._raise_for_stream_error(member, member_payload)
            result.stdout = "".join(stdout)
            result.stderr = "".join(stderr)
            return result
        except (AgentCoreRuntimeThrottlingError, AgentCoreRuntimeConflictError):
            raise
        except Exception as e:
            raise self._translate_error(e) from e

    def stop_session(self, agent_runtime_arn: str, session_id: str, qualifier: str | None = None) -> None:
        """Stop a Runtime session, flushing session storage durably.

        Blocks for several seconds (measured 3.4 s, occasionally 12 s), so keep it off
        interactive paths — call it on conversation end.
        """
        params = {
            "agentRuntimeArn": agent_runtime_arn,
            "runtimeSessionId": session_id,
            # clientToken carries the same 33-character minimum as runtimeSessionId.
            "clientToken": uuid.uuid4().hex.ljust(MIN_SESSION_ID_LENGTH, "0"),
        }
        if qualifier:
            params["qualifier"] = qualifier
        try:
            self._client.stop_runtime_session(**params)
        except Exception as e:
            if _error_code(e).lower() == RESOURCE_NOT_FOUND_CODE:
                logger.debug(f"AgentCore Runtime session {session_id} already stopped")
                return
            raise self._translate_error(e) from e
