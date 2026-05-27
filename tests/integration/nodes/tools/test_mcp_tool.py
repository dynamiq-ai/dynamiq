import contextlib
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import Field, create_model

from dynamiq import connections
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools import FileListTool, FileReadTool
from dynamiq.nodes.tools.mcp import MCPServer, MCPSse, MCPTool, extract_text_from_mcp_content


def assert_tool_matches(tool, expected, connection):
    assert tool.name == expected["name"]
    assert tool.description == expected["description"]
    assert tool.connection == connection
    assert set(tool.input_schema.__annotations__.items()) == expected["schema"]


def mock_get_input_schema(schema_dict: dict[str, Any]):
    """
    Creates an BaseClass based on provided JSON schema.

    Args:
        schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
    """
    fields = {}
    props = schema_dict.get("properties", {})
    required = set(schema_dict.get("required", []))
    json_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict",
        "array": "list",
    }
    for field_name, field_spec in props.items():
        field_type = json_mapping.get(field_spec.get("type", "string"))
        default = ... if field_name in required else None
        description = field_spec.get("description", None)
        fields[field_name] = (field_type, Field(default, description=description))
    return create_model(schema_dict.get("title", "MCPToolSchema"), **fields)


@pytest.fixture
def llm_model():
    connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key")
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=connection)


@pytest.fixture
def sse_server_connection():
    connection = MCPSse(url="https://example.com/")
    yield connection
    # Cleanup: ensure any async resources are closed
    if hasattr(connection, "close") and callable(connection.close):
        try:
            connection.close()
        except Exception:
            pass


@pytest.fixture
def mock_mcp_tools(sse_server_connection):
    with patch.object(MCPTool, "get_input_schema", side_effect=mock_get_input_schema):
        tools = {
            "add": MCPTool(
                name="add",
                description="Add two numbers",
                json_input_schema={
                    "title": "AddSchema",
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
            "multiply": MCPTool(
                name="multiply",
                description="Multiply two numbers",
                json_input_schema={
                    "title": "MultiplySchema",
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
            "subtract": MCPTool(
                name="subtract",
                description="Subtract two numbers",
                json_input_schema={
                    "title": "SubtractSchema",
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
        }
        yield tools


@pytest.fixture
def mcp_server_tool(sse_server_connection, mock_mcp_tools):
    tool = MCPServer(connection=sse_server_connection)
    tool._mcp_tools = mock_mcp_tools
    yield tool
    # Cleanup: ensure any async resources are closed
    if hasattr(tool, "close") and callable(tool.close):
        try:
            tool.close()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_get_mcp_tools(mcp_server_tool, sse_server_connection):
    with patch.object(MCPServer, "initialize_tools", new=AsyncMock()) as mock_init:
        tools = mcp_server_tool.get_mcp_tools()
        mock_init.assert_not_awaited()

        expected_tools = [
            {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
            {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
            {"name": "subtract", "description": "Subtract two numbers", "schema": {("a", "float"), ("b", "float")}},
        ]

        assert len(tools) == len(expected_tools)

        for tool, expected in zip(tools, expected_tools):
            assert_tool_matches(tool, expected, sse_server_connection)


@pytest.mark.asyncio
async def test_mock_tool_execute(mcp_server_tool):
    tool = mcp_server_tool._mcp_tools["add"]  # "add"

    mocked_result = {"content": {"result": 42}}

    with patch("dynamiq.nodes.tools.mcp.MCPTool.execute", return_value=mocked_result) as mock_exec:
        result = tool.execute(tool.input_schema(a=20, b=22))
        mock_exec.assert_called_once()
        assert result == mocked_result

    with patch.object(MCPTool, "execute_async", new_callable=AsyncMock, return_value=mocked_result) as mock_exec:
        result = await tool.run(input_data={"a": 20, "b": 22})
        mock_exec.assert_called_once()
        assert result.output == mocked_result


def test_mcp_tool_filter_names(mcp_server_tool, mock_mcp_tools):
    mcp_server_tool._mcp_tools = mock_mcp_tools

    mcp_server_tool.include_tools = ["add", "multiply"]
    mcp_server_tool.exclude_tools = []
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 2
    assert tools[0].name == "add" and tools[1].name == "multiply"

    mcp_server_tool.include_tools = []
    mcp_server_tool.exclude_tools = ["add", "multiply"]
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "subtract"

    mcp_server_tool.include_tools = ["multiply"]
    mcp_server_tool.exclude_tools = []
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "multiply"

    mcp_server_tool.include_tools = ["add", "multiply"]
    mcp_server_tool.exclude_tools = ["add"]
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "multiply"

    mcp_server_tool.include_tools = []
    mcp_server_tool.exclude_tools = ["multiply"]
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 2
    assert tools[0].name == "add" and tools[1].name == "subtract"

    mcp_server_tool.include_tools = ["add"]
    mcp_server_tool.exclude_tools = ["add"]
    tools = mcp_server_tool.get_mcp_tools()
    assert len(tools) == 0

    tools = mcp_server_tool.get_mcp_tools(select_all=True)
    assert len(tools) == 3


def test_agent_integration_with_mcp_tools(sse_server_connection, mock_mcp_tools, llm_model):
    mcp_server = MCPServer(connection=sse_server_connection)
    mcp_server._mcp_tools = {"add": mock_mcp_tools["add"], "multiply": mock_mcp_tools["multiply"]}
    mcp_tool = mock_mcp_tools["subtract"]

    agent = Agent(llm=llm_model, tools=[mcp_server, mcp_tool])

    agent_tools = [tool for tool in agent.tools if not isinstance(tool, (FileReadTool, FileListTool))]
    expected_tools = [
        {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
        {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
        {"name": "subtract", "description": "Subtract two numbers", "schema": {("a", "float"), ("b", "float")}},
    ]

    assert len(agent_tools) == 3
    assert all(isinstance(tool, MCPTool) for tool in agent_tools)
    for tool, expected in zip(agent_tools, expected_tools):
        assert_tool_matches(tool, expected, sse_server_connection)

    dict_tools = [
        tool
        for tool in agent.to_dict()["tools"]
        if tool["type"] not in ["dynamiq.nodes.tools.FileReadTool", "dynamiq.nodes.tools.FileListTool"]
    ]
    assert len(dict_tools) == 2
    assert dict_tools[0]["name"] == "subtract"
    assert dict_tools[0]["type"] == "dynamiq.nodes.tools.MCPTool"
    assert dict_tools[1]["type"] == "dynamiq.nodes.tools.MCPServer"


def test_extract_text_from_mcp_content():
    content = [
        TextContent(type="text", text="line one"),
        ImageContent(type="image", data="aGk=", mimeType="image/png"),
        TextContent(type="text", text="line two"),
    ]
    assert extract_text_from_mcp_content(content) == "line one\nline two"
    assert extract_text_from_mcp_content([]) == ""


def _patch_session_with_result(result: CallToolResult):
    """Patch the connection + ClientSession so call_tool yields `result`."""

    @contextlib.asynccontextmanager
    async def fake_connect(self):
        yield (object(), object())

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, args):
            return result

    return (
        patch.object(MCPSse, "connect", new=fake_connect),
        patch("dynamiq.nodes.tools.mcp.ClientSession", return_value=FakeSession()),
    )


@pytest.mark.asyncio
async def test_execute_async_optimized_returns_structured_content(mcp_server_tool):
    tool = mcp_server_tool._mcp_tools["add"]
    tool.is_optimized_for_agents = True  # the agent sets this on its tools at init
    payload = {"data": {"result": 42}, "meta": {"mode": "sandbox"}}
    result = CallToolResult(
        content=[TextContent(type="text", text="{...}")],
        structuredContent=payload,
    )

    conn_patch, session_patch = _patch_session_with_result(result)
    with conn_patch, session_patch:
        output = await tool.execute_async(tool.input_schema(a=20, b=22))

    assert output["content"] == payload
    assert output["raw_response"]["structuredContent"] == payload
    assert output["raw_response"]["content"] == [{"type": "text", "text": "{...}"}]
    assert output["raw_response"]["isError"] is False


@pytest.mark.asyncio
async def test_execute_async_optimized_falls_back_to_text(mcp_server_tool):
    tool = mcp_server_tool._mcp_tools["add"]
    tool.is_optimized_for_agents = True
    result = CallToolResult(content=[TextContent(type="text", text="plain text result")])

    conn_patch, session_patch = _patch_session_with_result(result)
    with conn_patch, session_patch:
        output = await tool.execute_async(tool.input_schema(a=20, b=22))

    assert output["content"] == "plain text result"
    assert output["raw_response"]["content"] == [{"type": "text", "text": "plain text result"}]
    assert output["raw_response"]["isError"] is False


@pytest.mark.asyncio
async def test_execute_async_not_optimized_returns_full_dump(mcp_server_tool):
    tool = mcp_server_tool._mcp_tools["add"]
    tool.is_optimized_for_agents = False
    payload = {"data": {"result": 42}}
    result = CallToolResult(
        content=[TextContent(type="text", text="{...}")],
        structuredContent=payload,
    )

    conn_patch, session_patch = _patch_session_with_result(result)
    with conn_patch, session_patch:
        output = await tool.execute_async(tool.input_schema(a=20, b=22))

    assert output["content"]["structuredContent"] == payload
    assert output["content"]["isError"] is False
    assert "content" in output["content"]


@pytest.mark.asyncio
async def test_execute_async_raises_on_is_error(mcp_server_tool):
    tool = mcp_server_tool._mcp_tools["add"]
    result = CallToolResult(
        content=[TextContent(type="text", text="API rate limit exceeded")],
        isError=True,
    )

    conn_patch, session_patch = _patch_session_with_result(result)
    with conn_patch, session_patch:
        with pytest.raises(ToolExecutionException, match="API rate limit exceeded"):
            await tool.execute_async(tool.input_schema(a=20, b=22))
