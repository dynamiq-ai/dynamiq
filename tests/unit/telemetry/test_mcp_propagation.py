"""MCP tools send the trace context of their own span to the server in the request ``_meta``."""

import contextlib
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, TextContent

from dynamiq.connections import MCPSse
from dynamiq.nodes.tools.mcp import MCPTool, ServerMetadata
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import span_named

pytest.importorskip("opentelemetry.sdk")

pytestmark = pytest.mark.usefixtures("no_span_leaks")


class StubSession:
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments=None, *args, **kwargs):
        StubSession.calls.append({"name": name, "arguments": arguments, "kwargs": kwargs})
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


@contextlib.asynccontextmanager
async def fake_connect(self, headers=None):
    yield (object(), object())


@pytest.fixture
def stub_server():
    StubSession.calls = []
    with patch.object(MCPSse, "connect", new=fake_connect), patch("mcp.ClientSession", StubSession):
        yield StubSession


def build_tool() -> MCPTool:
    return MCPTool(
        name="create_issue",
        description="Opens a GitHub issue.",
        json_input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
        connection=MCPSse(url="https://example.com/"),
        server_metadata=ServerMetadata(id="srv", name="github", description="GitHub MCP"),
    )


def test_request_meta_carries_the_trace_of_the_tool_span(stub_server, traced, exporter):
    result = build_tool().run(input_data={"title": "bug"}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    (call,) = stub_server.calls
    assert call["arguments"] == {"title": "bug"}
    span = span_named(exporter, "execute_tool create_issue")
    traceparent = call["kwargs"]["meta"]["traceparent"]
    version, trace_id, span_id, flags = traceparent.split("-")
    assert trace_id == f"{span.context.trace_id:032x}"
    assert span_id == f"{span.context.span_id:016x}"


@pytest.mark.asyncio
async def test_request_meta_on_the_async_path(stub_server, traced, exporter):
    result = await build_tool().run_async(input_data={"title": "bug"}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    span = span_named(exporter, "execute_tool create_issue")
    traceparent = stub_server.calls[0]["kwargs"]["meta"]["traceparent"]
    assert traceparent.split("-")[1] == f"{span.context.trace_id:032x}"


def test_no_meta_when_tracing_is_off(stub_server):
    config = RunnableConfig(tracing=TracingConfig(enabled=False))

    result = build_tool().run(input_data={"title": "bug"}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    (call,) = stub_server.calls
    assert call["kwargs"].get("meta") is None
