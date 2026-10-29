import json
import re
from datetime import datetime, timezone

import httpx
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials
from pydantic import ValidationError

from dynamiq import Workflow, connections
from dynamiq.connections import agentcore
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import BedrockAgentCoreWebSearchTool, SubAgentTool
from dynamiq.runnables import RunnableStatus

GATEWAY_URL = "https://websearch-gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
CREDENTIALS = ReadOnlyCredentials("AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "session-token")
SIGNING_TIME = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)
SEARCH_PAYLOAD = {
    "id": "search-1",
    "results": [
        {
            "title": "Weekly influenza surveillance report",
            "url": "https://www.cdc.gov/fluview/index.html",
            "publishedDate": "2026-10-23",
            "text": "Seasonal influenza activity is low nationally.",
        },
        {"text": "A passage the index has no title or URL for."},
    ],
}


def tool_result(payload: dict | str, is_error: bool = False) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class FakeGateway:
    """An AgentCore Gateway that checks every request's SigV4 signature and answers like the Web Search target."""

    def __init__(self):
        self.tool_pages = [["web-search___WebSearch"]]
        self.call_result = tool_result(SEARCH_PAYLOAD)
        self.event_stream = False
        self.error_status = None
        self.requests: list[httpx.Request] = []

    @property
    def messages(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self._assert_signed(request)
        if self.error_status:
            error = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32002, "message": "Insufficient permissions"}}
            return httpx.Response(self.error_status, json=error)

        message = json.loads(request.content)
        if message["method"] == "notifications/initialized":
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "gw"}}
            return self._reply(message, result, headers={"mcp-session-id": "session-1"})
        if message["method"] == "tools/list":
            page = int(message["params"].get("cursor", 0))
            result = {"tools": [{"name": name} for name in self.tool_pages[page]]}
            if page + 1 < len(self.tool_pages):
                result["nextCursor"] = str(page + 1)
            return self._reply(message, result)
        return self._reply(message, self.call_result)

    def _reply(self, message: dict, result: dict, headers: dict | None = None) -> httpx.Response:
        reply = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}, ensure_ascii=False)
        if not self.event_stream:
            return httpx.Response(200, text=reply, headers={"content-type": "application/json", **(headers or {})})
        # A resumable stream opens with an empty event, and the server may send its own requests and notifications,
        # whose ids can repeat the client's, before the reply.
        ping = json.dumps({"jsonrpc": "2.0", "id": message["id"], "method": "ping"})
        notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "hi"}})
        events = ["id: event-0\ndata:", f"data: {ping}", f"event: message\ndata: {notification}", f"data: {reply}"]
        stream = "\n\n".join(events) + "\n\n"
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream", **(headers or {})})

    @staticmethod
    def _assert_signed(request: httpx.Request) -> None:
        """Re-sign what was actually sent and compare: the signature has to cover the request as it arrived."""
        authorization = request.headers["authorization"]
        assert "/us-east-1/bedrock-agentcore/aws4_request" in authorization
        signed_headers = re.search(r"SignedHeaders=([^,]+)", authorization).group(1).split(";")
        expected = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={name: request.headers[name] for name in signed_headers if name != "host"},
        )
        SigV4Auth(CREDENTIALS, "bedrock-agentcore", "us-east-1").add_auth(expected)
        assert expected.headers["Authorization"] == authorization


@pytest.fixture
def gateway(mocker):
    fake = FakeGateway()
    transport = httpx.MockTransport(fake.handle)
    client, async_client = httpx.Client, httpx.AsyncClient
    mocker.patch.object(agentcore.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    mocker.patch.object(agentcore.httpx, "AsyncClient", lambda **kwargs: async_client(transport=transport, **kwargs))
    mocker.patch("botocore.auth.get_current_datetime", return_value=SIGNING_TIME)
    return fake


def aws_connection(**kwargs) -> connections.AWS:
    # The region differs from the gateway's on purpose: requests must be signed for the gateway's region.
    return connections.AWS(
        access_key_id=CREDENTIALS.access_key,
        secret_access_key=CREDENTIALS.secret_key,
        session_token=CREDENTIALS.token,
        region="eu-west-1",
        **kwargs,
    )


def web_search_tool(**kwargs) -> BedrockAgentCoreWebSearchTool:
    return BedrockAgentCoreWebSearchTool(connection=aws_connection(), gateway_url=GATEWAY_URL, **kwargs)


def test_search_returns_cited_results(gateway):
    tool = web_search_tool(target_name="web-search", max_results=5, is_optimized_for_agents=True)

    result = tool.run({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert "### Result 1: Weekly influenza surveillance report" in content
    assert "- URL: [https://www.cdc.gov/fluview/index.html](https://www.cdc.gov/fluview/index.html)" in content
    assert "- Published: 2026-10-23" in content
    assert "- Content: Seasonal influenza activity is low nationally." in content
    assert "### Result 2: Untitled result" in content
    assert result.output["urls"] == ["https://www.cdc.gov/fluview/index.html"]
    assert result.output["sources"][0]["title"] == "Weekly influenza surveillance report"

    assert [message["method"] for message in gateway.messages] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert gateway.messages[-1]["params"] == {
        "name": "web-search___WebSearch",
        "arguments": {"query": "influenza activity this week", "maxResults": 5},
    }
    assert "mcp-protocol-version" not in gateway.requests[0].headers
    for request in gateway.requests[1:]:
        assert request.headers["mcp-protocol-version"] == "2025-03-26"
        assert request.headers["mcp-session-id"] == "session-1"


@pytest.mark.asyncio
async def test_async_search_reads_event_stream_replies(gateway):
    # JSON may carry these characters unescaped, and only CR and LF end a line of an event stream.
    payload = {**SEARCH_PAYLOAD, "results": [{**SEARCH_PAYLOAD["results"][0], "text": "Low\u2028activity\u0085."}]}
    gateway.event_stream = True
    gateway.call_result = tool_result(payload)
    tool = web_search_tool(target_name="web-search")

    result = await tool.run_async({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert content["raw_response"] == payload
    assert content["urls"] == ["https://www.cdc.gov/fluview/index.html"]
    assert content["sources_with_url"][0] == (
        "- [Weekly influenza surveillance report](https://www.cdc.gov/fluview/index.html)"
    )


def test_finds_the_web_search_tool_when_target_name_is_unset(gateway):
    gateway.tool_pages = [["kb___Retrieve"], ["search___WebSearch", "search___Other"]]

    result = web_search_tool().run({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.SUCCESS
    tool_lists = [message for message in gateway.messages if message["method"] == "tools/list"]
    assert [message["params"] for message in tool_lists] == [{}, {"cursor": "1"}]
    assert gateway.messages[-1]["params"]["name"] == "search___WebSearch"


@pytest.mark.parametrize(
    ("tool_names", "error"),
    [
        (["kb___Retrieve"], "The gateway has no Web Search tool"),
        (["a___WebSearch", "b___WebSearch"], "several Web Search tools (a___WebSearch, b___WebSearch)"),
    ],
)
def test_tool_lookup_needs_exactly_one_web_search_tool(gateway, tool_names, error):
    gateway.tool_pages = [tool_names]

    result = web_search_tool().run({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.FAILURE
    assert error in result.error.message
    assert "tools/call" not in [message["method"] for message in gateway.messages]


def test_filters_are_sent_only_when_set(gateway):
    tool = web_search_tool(target_name="web-search", include_domains=["cdc.gov"], exclude_domains=[])

    tool.run({"query": "influenza activity this week", "include_domains": []})
    assert gateway.messages[-1]["params"]["arguments"] == {"query": "influenza activity this week"}

    tool.run(
        {
            "query": "influenza activity this week",
            "exclude_domains": ["example.com"],
            "start_published_date": "2026-10-01",
            "end_published_date": "2026-10-31",
        }
    )
    assert gateway.messages[-1]["params"]["arguments"]["filters"] == {
        "domainFilter": {"include": ["cdc.gov"], "exclude": ["example.com"]},
        "publishedDateFilter": {"from": "2026-10-01T00:00:00Z", "to": "2026-10-31T23:59:59Z"},
    }


@pytest.mark.parametrize(
    "input_data",
    [
        {"query": "q" * 201},
        {"query": "   "},
        {"query": "influenza", "max_results": 26},
        {"query": "influenza", "include_domains": [f"site{index}.org" for index in range(101)]},
    ],
)
def test_input_outside_the_service_limits_is_rejected_before_any_request(gateway, input_data):
    result = web_search_tool(target_name="web-search").run(input_data)

    assert result.status == RunnableStatus.FAILURE
    assert gateway.requests == []


@pytest.mark.parametrize(
    ("gateway_settings", "error"),
    [
        ({"error_status": 403}, "HTTP 403"),
        ({"call_result": tool_result("Query rejected", is_error=True)}, "Query rejected"),
        ({"call_result": tool_result("not json")}, "not a JSON object: not json"),
    ],
)
def test_gateway_failures_are_recoverable_tool_errors(gateway, gateway_settings, error):
    vars(gateway).update(gateway_settings)

    result = web_search_tool(target_name="web-search").run({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.FAILURE
    assert result.error.recoverable
    assert error in result.error.message


def test_missing_credentials_fail_the_search(gateway, mocker):
    mocker.patch("boto3.session.Session.get_credentials", return_value=None)

    result = web_search_tool(target_name="web-search").run({"query": "influenza activity this week"})

    assert result.status == RunnableStatus.FAILURE
    assert "No AWS credentials found" in result.error.message
    assert gateway.requests == []


@pytest.mark.parametrize(
    "gateway_url",
    [
        "http://websearch-gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp",
        "https://gateway.example.com/mcp",
        "https://websearch-gw-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com.example.com/mcp",
    ],
)
def test_gateway_url_must_be_https_on_an_aws_host(gateway_url):
    with pytest.raises(ValidationError, match="HTTPS URL on an AWS host"):
        BedrockAgentCoreWebSearchTool(connection=aws_connection(), gateway_url=gateway_url)


def test_gateway_url_without_a_path_points_at_the_mcp_endpoint():
    tool = BedrockAgentCoreWebSearchTool(connection=aws_connection(), gateway_url=GATEWAY_URL.removesuffix("/mcp"))

    assert tool.gateway_url == GATEWAY_URL


def test_yaml_round_trip_keeps_the_gateway_settings(gateway, tmp_path):
    tool = web_search_tool(
        id="web-search", target_name="web-search", max_results=5, include_domains=["cdc.gov"], timeout=10
    )
    first = tmp_path / "search.yaml"
    Workflow(flow=Flow(nodes=[tool])).to_yaml_file(first)
    loaded = Workflow.from_yaml_file(str(first), init_components=True)
    second = tmp_path / "search_again.yaml"
    loaded.to_yaml_file(second)
    reloaded = Workflow.from_yaml_file(str(second), init_components=True)

    reloaded_tool = reloaded.flow.nodes[0]
    assert isinstance(reloaded_tool, BedrockAgentCoreWebSearchTool)
    assert (reloaded_tool.gateway_url, reloaded_tool.target_name, reloaded_tool.max_results) == (
        GATEWAY_URL,
        "web-search",
        5,
    )
    assert (reloaded_tool.include_domains, reloaded_tool.timeout) == (["cdc.gov"], 10)
    assert reloaded_tool.connection.secret_access_key == CREDENTIALS.secret_key
    assert reloaded_tool.run({"query": "influenza activity this week"}).status == RunnableStatus.SUCCESS


def test_sub_agent_factory_rebuilds_the_tool_from_its_dict(gateway):
    # Agent chats hand their web search tool to sub-agents as a dict, which is resolved like workflow YAML.
    llm = OpenAI(connection=connections.OpenAI(api_key="k"), model="gpt-4o-mini", is_postponed_component_init=True)
    tool = web_search_tool(target_name="web-search", is_optimized_for_agents=True)
    sub_agent = SubAgentTool(
        name="universal-agent",
        description="Researches a task on the web.",
        validate_factory=False,
        agent_factory={"name": "Universal", "llm": llm.to_dict(), "tools": [tool.to_dict()]},
    )

    rebuilt = sub_agent.get_or_create_agent().tools[0]

    assert isinstance(rebuilt, BedrockAgentCoreWebSearchTool)
    assert rebuilt.run({"query": "influenza activity this week"}).status == RunnableStatus.SUCCESS
