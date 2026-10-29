import asyncio
import json
import posixpath
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.credentials import ReadOnlyCredentials
from pydantic import BaseModel, Field

from dynamiq.connections import AWS as AWSConnection
from dynamiq.utils.logger import logger
from dynamiq.utils.user_agent import VERSION

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


class AgentCoreSession(BaseModel):
    """Handle for a running AgentCore code interpreter session."""

    identifier: str
    session_id: str


class AgentCoreInvocationResult(BaseModel):
    """Normalized result of an AgentCore code interpreter invocation."""

    content: list[dict] = Field(default_factory=list)
    structured: dict = Field(default_factory=dict)
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
            if key in ("stdout", "stderr"):
                # Concatenate; an empty string in a later event must not wipe accumulated output.
                if value:
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


GATEWAY_SIGNING_SERVICE = "bedrock-agentcore"
# Offered in initialize. The gateway answers with a version it supports, and later requests carry that one.
GATEWAY_MCP_PROTOCOL_VERSION = "2025-06-18"
# Gateway requests carry the connection's signed credentials, so they may only go to AWS hosts.
AWS_HOST_SUFFIXES = (".amazonaws.com", ".amazonaws.com.cn", ".api.aws")
GATEWAY_HOST_REGION = re.compile(r"\.bedrock-agentcore\.([a-z0-9-]+)\.")
MAX_TOOL_LIST_PAGES = 50


class AgentCoreGatewayError(Exception):
    """Raised when an AgentCore Gateway request fails or its reply cannot be decoded."""


def normalize_gateway_url(url: str) -> str:
    """Return the MCP endpoint of an AgentCore Gateway URL, adding the /mcp path when the URL has none.

    Raises:
        ValueError: If the URL is not HTTPS on an AWS host.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(AWS_HOST_SUFFIXES):
        raise ValueError(f"Gateway URL must be an HTTPS URL on an AWS host, got '{url}'.")
    if parsed.path in ("", "/"):
        parsed = parsed._replace(path="/mcp")
    return parsed.geturl()


class AgentCoreGatewayClient:
    """Calls the tools of an AgentCore Gateway that authorizes callers with AWS IAM.

    Every request is signed with SigV4, so the connection's credentials need bedrock-agentcore:InvokeGateway on the
    gateway. A session holds no state once it is closed, so one client serves concurrent runs.
    """

    def __init__(self, connection: AWSConnection, url: str, timeout: float):
        self.url = normalize_gateway_url(url)
        self.timeout = timeout
        # Loading CA certificates blocks, so every session reuses one context instead of building its own.
        self._ssl_context = httpx.create_ssl_context()
        self._boto3_session = connection.get_boto3_session()
        match = GATEWAY_HOST_REGION.search(urlparse(self.url).hostname or "")
        self.region = match.group(1) if match else self._boto3_session.region_name
        if not self.region:
            raise ValueError("Set the region on the AWS connection: the gateway URL does not name one.")

    def get_credentials(self) -> ReadOnlyCredentials:
        """Return credentials to sign with, refreshing temporary ones that are about to expire."""
        credentials = self._boto3_session.get_credentials()
        if credentials is None:
            raise AgentCoreGatewayError("No AWS credentials found to sign the gateway requests.")
        return credentials.get_frozen_credentials()

    @contextmanager
    def session(self) -> Iterator["AgentCoreGatewaySession"]:
        """Open an MCP session with the gateway and complete its handshake."""
        with httpx.Client(timeout=self.timeout, verify=self._ssl_context) as http:
            session = AgentCoreGatewaySession(self, self.get_credentials(), http)
            session.initialize()
            yield session

    @asynccontextmanager
    async def async_session(self) -> AsyncIterator["AsyncAgentCoreGatewaySession"]:
        """Open an MCP session with the gateway and complete its handshake."""
        # Resolving credentials can read files or call STS.
        credentials = await asyncio.to_thread(self.get_credentials)
        async with httpx.AsyncClient(timeout=self.timeout, verify=self._ssl_context) as http:
            session = AsyncAgentCoreGatewaySession(self, credentials, http)
            await session.initialize()
            yield session


class _GatewaySessionBase:
    """Builds and decodes the JSON-RPC messages of one MCP session over streamable HTTP."""

    def __init__(self, client: AgentCoreGatewayClient, credentials: ReadOnlyCredentials):
        self._client = client
        self._credentials = credentials
        self._last_id = 0
        self._session_id: str | None = None
        self._protocol_version: str | None = None

    def _prepare(self, method: str, params: dict[str, Any] | None) -> tuple[int | None, bytes, dict[str, str]]:
        """Return the request id, body and signed headers of a message. A notification takes no id."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        request_id = None
        if not method.startswith("notifications/"):
            self._last_id += 1
            request_id = message["id"] = self._last_id
        if params is not None:
            message["params"] = params
        body = json.dumps(message).encode()

        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._protocol_version:
            headers["Mcp-Protocol-Version"] = self._protocol_version
        request = AWSRequest(method="POST", url=self._client.url, data=body, headers=headers)
        SigV4Auth(self._credentials, GATEWAY_SIGNING_SERVICE, self._client.region).add_auth(request)
        return request_id, body, dict(request.headers.items())

    def _read(self, response: httpx.Response, request_id: int | None) -> dict[str, Any]:
        """Return the result of the reply to a request, or an empty dict for a notification."""
        self._session_id = response.headers.get("mcp-session-id") or self._session_id
        if response.is_error:
            raise AgentCoreGatewayError(
                f"Gateway request failed with HTTP {response.status_code}: {response.text[:500]}"
            )
        if request_id is None:
            return {}

        for message in _json_rpc_messages(response):
            # Requests and notifications the server sends on the stream are not replies.
            if not isinstance(message, dict) or "method" in message:
                continue
            # A JSON-RPC error may carry a null id, so any error is reported.
            if error := message.get("error"):
                raise AgentCoreGatewayError(
                    f"Gateway returned JSON-RPC error {error.get('code')}: {error.get('message')}"
                )
            if message.get("id") == request_id:
                return message.get("result") or {}
        raise AgentCoreGatewayError(f"Gateway sent no reply to request {request_id}.")

    def _initialize_params(self) -> dict[str, Any]:
        return {
            "protocolVersion": GATEWAY_MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "dynamiq", "version": VERSION},
        }

    def _set_protocol_version(self, initialize_result: dict[str, Any]) -> None:
        self._protocol_version = initialize_result.get("protocolVersion") or GATEWAY_MCP_PROTOCOL_VERSION

    @staticmethod
    def _collect_tools(result: dict[str, Any], tools: dict[str, dict[str, Any]]) -> str | None:
        """Add the input schemas of a tools/list page to tools by name and return the cursor of the next page."""
        for tool in result.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                input_schema = tool.get("inputSchema")
                tools[tool["name"]] = input_schema if isinstance(input_schema, dict) else {}
        return result.get("nextCursor") or None


class AgentCoreGatewaySession(_GatewaySessionBase):
    """An initialized MCP session with an AgentCore Gateway."""

    def __init__(self, client: AgentCoreGatewayClient, credentials: ReadOnlyCredentials, http: httpx.Client):
        super().__init__(client, credentials)
        self._http = http

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id, body, headers = self._prepare(method, params)
        response = self._http.post(self._client.url, content=body, headers=headers)
        return self._read(response, request_id)

    def initialize(self) -> None:
        self._set_protocol_version(self.request("initialize", self._initialize_params()))
        self.request("notifications/initialized")

    def list_tools(self) -> dict[str, dict[str, Any]]:
        """Return the input schema of every tool on the gateway by tool name."""
        tools: dict[str, dict[str, Any]] = {}
        cursor = None
        for _ in range(MAX_TOOL_LIST_PAGES):
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            cursor = self._collect_tools(result, tools)
            if not cursor:
                return tools
        raise AgentCoreGatewayError(f"Gateway tools/list did not end within {MAX_TOOL_LIST_PAGES} pages.")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})


class AsyncAgentCoreGatewaySession(_GatewaySessionBase):
    """An initialized MCP session with an AgentCore Gateway, for async code."""

    def __init__(self, client: AgentCoreGatewayClient, credentials: ReadOnlyCredentials, http: httpx.AsyncClient):
        super().__init__(client, credentials)
        self._http = http

    async def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        request_id, body, headers = self._prepare(method, params)
        response = await self._http.post(self._client.url, content=body, headers=headers)
        return self._read(response, request_id)

    async def initialize(self) -> None:
        self._set_protocol_version(await self.request("initialize", self._initialize_params()))
        await self.request("notifications/initialized")

    async def list_tools(self) -> dict[str, dict[str, Any]]:
        """Return the input schema of every tool on the gateway by tool name."""
        tools: dict[str, dict[str, Any]] = {}
        cursor = None
        for _ in range(MAX_TOOL_LIST_PAGES):
            result = await self.request("tools/list", {"cursor": cursor} if cursor else {})
            cursor = self._collect_tools(result, tools)
            if not cursor:
                return tools
        raise AgentCoreGatewayError(f"Gateway tools/list did not end within {MAX_TOOL_LIST_PAGES} pages.")

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.request("tools/call", {"name": name, "arguments": arguments})


def _json_rpc_messages(response: httpx.Response) -> list[Any]:
    """Decode the JSON-RPC messages of a reply, which is either a JSON body or a server-sent event stream."""
    if not response.headers.get("content-type", "").lower().startswith("text/event-stream"):
        try:
            return [response.json()]
        except ValueError as e:
            raise AgentCoreGatewayError(f"Could not decode the gateway reply: {response.text[:200]}") from e

    messages = []
    for data in _sse_data(response.text):
        try:
            messages.append(json.loads(data))
        except ValueError:
            # Some events carry no message, such as the empty event that opens a resumable stream.
            continue
    return messages


def _sse_data(text: str) -> Iterator[str]:
    """Yield the data of each server-sent event. An event's data lines are joined with newlines."""
    lines: list[str] = []
    # Only CR and LF end an SSE line: splitlines() would also split JSON at characters such as U+2028.
    for line in [*re.split(r"\r\n|\r|\n", text), ""]:
        if line.startswith("data:"):
            lines.append(line[len("data:") :].removeprefix(" "))
        elif not line and lines:
            yield "\n".join(lines)
            lines = []
