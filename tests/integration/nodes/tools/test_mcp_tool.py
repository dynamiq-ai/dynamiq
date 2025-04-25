import uuid
from unittest.mock import AsyncMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter, MCPTool, MPCConnection


def assert_tool_matches(tool, expected, connection):
    assert tool.name == expected["name"]
    assert tool.description == expected["description"]
    assert tool.connection == connection
    assert set(tool.input_schema.__annotations__.items()) == expected["schema"]


@pytest.fixture
def model():
    connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key")
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=connection)


@pytest.fixture
def sse_server_connection():
    return MPCConnection(url="mock_url.py")


@pytest.fixture
def mock_mcp_tools(sse_server_connection):
    return {
        "add": MCPTool(
            name="add",
            description="Add two numbers",
            input_schema={
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
            input_schema={
                "title": "MultiplySchema",
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
            connection=sse_server_connection,
        ),
    }


@pytest.fixture
def mcp_adapter_tool(sse_server_connection, mock_mcp_tools):
    tool = MCPServerAdapter(connection=sse_server_connection)
    tool.mcp_tools = mock_mcp_tools
    return tool


@pytest.mark.asyncio
async def test_get_mcp_tools(mcp_adapter_tool, sse_server_connection):
    with patch.object(MCPServerAdapter, "initialize_tools", new=AsyncMock()) as mock_init:
        tools = mcp_adapter_tool.get_mcp_tools()
        mock_init.assert_not_awaited()

        expected_tools = [
            {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
            {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
        ]

        assert len(tools) == len(expected_tools)

        for tool, expected in zip(tools, expected_tools):
            assert_tool_matches(tool, expected, sse_server_connection)


@pytest.mark.asyncio
async def test_agent_with_mcp_tool(mcp_adapter_tool, model, sse_server_connection):
    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=model,
        tools=[mcp_adapter_tool],
        max_loops=10,
    )

    tools = agent.tools
    expected_tools = [
        {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
        {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
    ]

    assert len(tools) == len(expected_tools)

    for tool, expected in zip(tools, expected_tools):
        assert_tool_matches(tool, expected, sse_server_connection)


@pytest.mark.asyncio
async def test_mock_tool_execute(mcp_adapter_tool):
    tool = mcp_adapter_tool.mcp_tools["add"]  # "add"

    mocked_result = {"content": {"result": 42}}

    with patch("dynamiq.nodes.tools.mcp_adapter.MCPTool.execute", return_value=mocked_result) as mock_exec:
        result = tool.execute(tool.input_schema(a=20, b=22))
        mock_exec.assert_called_once()
        assert result == mocked_result

    with patch("dynamiq.nodes.tools.mcp_adapter.MCPTool.execute", return_value=mocked_result) as mock_exec:
        result = await tool.run(input_data={"a": 20, "b": 22})
        mock_exec.assert_called_once()
        assert result.output == mocked_result
