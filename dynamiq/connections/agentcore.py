import posixpath
import uuid
from dataclasses import dataclass, field

from dynamiq.connections import AWS as AWSConnection
from dynamiq.utils.logger import logger

DEFAULT_CODE_INTERPRETER_IDENTIFIER = "aws.codeinterpreter.v1"
MAX_SESSION_TIMEOUT_SECONDS = 28800
# Transient, retry-worthy AWS error codes. Matched case-insensitively so both the
# REST error codes (``ThrottlingException``) and the event-stream member names
# delivered mid-invocation (``throttlingException``) are covered. Service-quota
# exhaustion is included because concurrent-session limits are the most likely
# failure under parallel (multi-tenant) session creation and often clear on backoff.
RETRYABLE_ERROR_CODES = frozenset(
    code.lower()
    for code in (
        "ThrottlingException",
        "TooManyRequestsException",
        "ThrottledException",
        "ServiceQuotaExceededException",
    )
)
RESOURCE_NOT_FOUND_CODE = "resourcenotfoundexception"


class AgentCoreThrottlingError(Exception):
    """Raised when AWS throttles or rate/quota-limits Bedrock AgentCore requests.

    Used as the retry signal for session-lifecycle backoff; covers both
    ``ThrottlingException`` and ``ServiceQuotaExceededException``.
    """


def _error_code(error: Exception) -> str:
    """Extract the AWS error code from a botocore error, tolerating a ``None`` response."""
    response = getattr(error, "response", None) or {}
    return (response.get("Error") or {}).get("Code", "") or ""


@dataclass
class AgentCoreSession:
    """Handle for a running AgentCore code interpreter session."""

    identifier: str
    session_id: str


@dataclass
class AgentCoreInvocationResult:
    """Normalized result of an AgentCore code interpreter invocation."""

    content: list[dict] = field(default_factory=list)
    structured: dict = field(default_factory=dict)
    is_error: bool = False

    @property
    def stdout(self) -> str:
        stdout = self.structured.get("stdout")
        return stdout if stdout is not None else self.text

    @property
    def stderr(self) -> str:
        return self.structured.get("stderr") or ""

    @property
    def exit_code(self) -> int:
        exit_code = self.structured.get("exitCode")
        if exit_code is None:
            return 1 if self.is_error else 0
        return exit_code

    @property
    def text(self) -> str:
        """Joined text of all text content blocks."""
        return "\n".join(block["text"] for block in self.content if block.get("type") == "text" and block.get("text"))

    @property
    def error_text(self) -> str:
        return self.stderr or self.text or "Unknown execution error"


class AgentCoreCodeInterpreterClient:
    """Thin wrapper over the boto3 ``bedrock-agentcore`` data-plane client.

    Handles session lifecycle, invocation event-stream normalization, and
    translation of AWS throttling errors so callers can retry on a typed exception.
    """

    def __init__(self, connection: AWSConnection, read_timeout: int = 910):
        from botocore.config import Config

        session = connection.get_boto3_session()
        # Disable botocore's built-in retries: the default legacy mode silently re-sends
        # failed requests up to 5 times, which for the non-idempotent InvokeCodeInterpreter
        # (executeCode/executeCommand carry no idempotency token) could execute user code
        # multiple times on a transient socket error. Transient throttling on the idempotent
        # session-lifecycle calls is instead retried at the application level via tenacity.
        self._client = session.client(
            "bedrock-agentcore",
            config=Config(read_timeout=read_timeout, retries={"total_max_attempts": 1}),
        )

    @staticmethod
    def _translate_error(error: Exception) -> Exception:
        if _error_code(error).lower() in RETRYABLE_ERROR_CODES:
            return AgentCoreThrottlingError(str(error))
        return error

    def start_session(
        self,
        identifier: str,
        name: str | None = None,
        session_timeout_seconds: int | None = None,
    ) -> AgentCoreSession:
        params = {"codeInterpreterIdentifier": identifier}
        params["name"] = name or f"dynamiq-session-{uuid.uuid4().hex[:8]}"
        if session_timeout_seconds is not None:
            params["sessionTimeoutSeconds"] = min(max(session_timeout_seconds, 1), MAX_SESSION_TIMEOUT_SECONDS)
        try:
            response = self._client.start_code_interpreter_session(**params)
        except Exception as e:
            raise self._translate_error(e) from e
        return AgentCoreSession(identifier=response["codeInterpreterIdentifier"], session_id=response["sessionId"])

    def get_session_status(self, identifier: str, session_id: str) -> str:
        try:
            response = self._client.get_code_interpreter_session(
                codeInterpreterIdentifier=identifier, sessionId=session_id
            )
        except Exception as e:
            raise self._translate_error(e) from e
        return response.get("status", "")

    def stop_session(self, session: AgentCoreSession) -> None:
        try:
            self._client.stop_code_interpreter_session(
                codeInterpreterIdentifier=session.identifier, sessionId=session.session_id
            )
        except Exception as e:
            if _error_code(e).lower() == RESOURCE_NOT_FOUND_CODE:
                logger.debug(f"AgentCore session {session.session_id} already stopped")
                return
            raise self._translate_error(e) from e

    def invoke(self, session: AgentCoreSession, tool_name: str, arguments: dict) -> AgentCoreInvocationResult:
        try:
            response = self._client.invoke_code_interpreter(
                codeInterpreterIdentifier=session.identifier,
                sessionId=session.session_id,
                name=tool_name,
                arguments=arguments,
            )
            result = AgentCoreInvocationResult()
            for event in response["stream"]:
                event_result = event.get("result")
                if event_result is None:
                    continue
                result.content.extend(event_result.get("content") or [])
                self._merge_structured(result.structured, event_result.get("structuredContent"))
                result.is_error = result.is_error or bool(event_result.get("isError"))
            return result
        except Exception as e:
            raise self._translate_error(e) from e

    @staticmethod
    def _merge_structured(target: dict, structured: dict | None) -> None:
        """Merge one event's structuredContent into the accumulator field-wise.

        stdout/stderr are concatenated across events (large output can span multiple
        result events); scalar fields (exitCode, taskId, taskStatus, executionTime)
        take the latest non-null value.
        """
        if not structured:
            return
        for key, value in structured.items():
            if key in ("stdout", "stderr") and value:
                target[key] = (target.get(key) or "") + value
            elif value is not None:
                target[key] = value

    @staticmethod
    def extract_file_bytes(result: AgentCoreInvocationResult) -> bytes:
        """Extract file content bytes from a ``readFiles`` invocation result."""
        for block in result.content:
            if block.get("type") == "resource":
                resource = block.get("resource") or {}
                if resource.get("blob") is not None:
                    blob = resource["blob"]
                    return blob if isinstance(blob, bytes) else str(blob).encode("utf-8")
                if resource.get("text") is not None:
                    return resource["text"].encode("utf-8")
            elif block.get("type") == "text" and block.get("text") is not None:
                return block["text"].encode("utf-8")
        raise FileNotFoundError("No file content found in AgentCore response")


def normalize_sandbox_path(path: str) -> str:
    """Normalize a path for AgentCore file APIs, which expect workspace-relative paths."""
    normalized = posixpath.normpath(path)
    if normalized in (".", "/"):
        return "."
    return normalized
