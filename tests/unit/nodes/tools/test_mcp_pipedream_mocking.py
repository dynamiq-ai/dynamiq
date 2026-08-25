"""Mock-execution coverage for the two tool families most likely to have real side effects.

MCP servers and Pipedream actions reach external systems — GitHub, Slack, databases, SaaS
apps — so they are the tools a dry run most needs to hold back. MCP has a wrinkle: the
``MCPServer`` node on a canvas is a discovery wrapper that never executes, and the tools that
do execute are constructed by it at discovery time.
"""

import contextlib
from unittest.mock import patch

import pytest

from dynamiq.connections.connections import PipedreamOAuth2
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.mcp import MCPServer, MCPSse, MCPTool, ServerMetadata
from dynamiq.nodes.tools.pipedream import Pipedream
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.mocking import MockConfig, MockPolicy, RunMockConfig


class FakeMCPTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} description"
        self.inputSchema = {"type": "object", "properties": {"q": {"type": "string"}}}


def patch_discovery(*tool_names):
    """Patch the connection and session so `initialize_tools` discovers `tool_names`."""

    @contextlib.asynccontextmanager
    async def fake_connect(self):
        yield (object(), object())

    class FakeToolList:
        tools = [FakeMCPTool(name) for name in tool_names]

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return FakeToolList()

    return (
        patch.object(MCPSse, "connect", new=fake_connect),
        patch("dynamiq.nodes.tools.mcp.ClientSession", return_value=FakeSession()),
    )


def build_mcp_tool(**kwargs) -> MCPTool:
    return MCPTool(
        name="create_issue",
        description="Opens a GitHub issue.",
        json_input_schema={"type": "object", "properties": {"title": {"type": "string"}}},
        connection=MCPSse(url="https://example.com/"),
        server_metadata=ServerMetadata(id="srv", name="github", description="GitHub MCP"),
        **kwargs,
    )


class TestMCPToolMocking:
    def test_mocked_mcp_tool_never_reaches_the_server(self):
        tool = build_mcp_tool(mock=MockConfig(enabled=True, output="issue #1 created"))

        with patch.object(MCPTool, "execute", side_effect=AssertionError("MCP server was contacted")):
            result = tool.run(input_data={"title": "bug"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "[MOCKED] issue #1 created"
        assert result.output["is_mocked"] is True

    @pytest.mark.asyncio
    async def test_mocked_mcp_tool_is_suppressed_on_the_async_path(self):
        """MCPTool has a native execute_async, so it takes the async seam."""
        tool = build_mcp_tool(mock=MockConfig(enabled=True))

        with patch.object(MCPTool, "execute_async", side_effect=AssertionError("MCP server was contacted")):
            result = await tool.run_async(input_data={"title": "bug"})

        assert result.output["is_mocked"] is True

    def test_run_level_policy_sweeps_in_mcp_tools(self):
        tool = build_mcp_tool()

        with patch.object(MCPTool, "execute", side_effect=AssertionError("MCP server was contacted")):
            result = tool.run(
                input_data={"title": "bug"},
                config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            )

        assert result.output["is_mocked"] is True


class TestMCPServerMockPropagation:
    """The server never runs, so a mock configured on it has to reach the tools it discovers."""

    @staticmethod
    def discover(server: MCPServer, *names: str):
        connect_patch, session_patch = patch_discovery(*names)
        with connect_patch, session_patch:
            return server.get_mcp_tools()

    def test_server_mock_governs_every_discovered_tool(self):
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, locked=True, output="suppressed"),
        )

        tools = self.discover(server, "create_issue", "merge_pr")

        assert len(tools) == 2
        for tool in tools:
            resolved = tool.resolve_mock(RunnableConfig())
            assert resolved is not None, f"{tool.name} would have executed for real"
            assert resolved.output == "suppressed"
            assert resolved.locked is True

    def test_a_mock_set_after_discovery_still_applies(self):
        """The server is read when the tool runs, not snapshotted when it was discovered."""
        server = MCPServer(connection=MCPSse(url="https://example.com/"), name="github-mcp")
        (tool,) = self.discover(server, "create_issue")
        assert tool.resolve_mock(RunnableConfig()) is None

        server.mock = MockConfig(enabled=True, locked=True, output="pinned late")

        resolved = tool.resolve_mock(RunnableConfig())
        assert resolved is not None, "a mock configured after discovery must still suppress the tool"
        assert resolved.output == "pinned late"

    def test_a_tools_own_mock_wins_over_the_servers(self):
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, output="from server"),
        )
        (tool,) = self.discover(server, "create_issue")
        tool.mock = MockConfig(enabled=True, output="from tool")

        assert tool.resolve_mock(RunnableConfig()).output == "from tool"

    @staticmethod
    def _pinned_server_with_a_tool_level_mock():
        """A server pinned off, one of whose tools carries its own (unlocked) mock."""
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, locked=True, output="never fires"),
        )
        (tool,) = TestMCPServerMockPropagation.discover(server, "create_issue")
        tool.mock = MockConfig(enabled=True, output="from tool")
        return tool

    def test_a_tools_own_mock_keeps_the_servers_pin(self):
        """Authoring a payload for one tool is not opting out of the server's pin.

        The tool's own config wins on output, but `locked` belongs to whoever pinned the
        server: without it carried across, MockPolicy.NONE would reach the real server.
        """
        tool = self._pinned_server_with_a_tool_level_mock()

        resolved = tool.resolve_mock(RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)))

        assert resolved is not None, "the pinned server would have been contacted for real"
        assert resolved.output == "from tool", "the tool's own payload must still win"

    def test_excluding_a_tool_cannot_unpin_its_server(self):
        """An exclusion is a run-level request, and a pin outranks every one of those."""
        tool = self._pinned_server_with_a_tool_level_mock()

        resolved = tool.resolve_mock(RunnableConfig(mock=RunMockConfig(exclude_ids={tool.id})))

        assert resolved is not None, "the pinned server would have been contacted for real"
        assert resolved.output == "from tool"

    def test_an_unpinned_server_does_not_pin_its_tools(self):
        """Only `locked` is inherited; an ordinary server mock stays unpinnned."""
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, output="from server"),
        )
        (tool,) = self.discover(server, "create_issue")
        tool.mock = MockConfig(enabled=True, output="from tool")

        assert tool.resolve_mock(RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE))) is None

    def test_the_server_can_be_excluded_by_its_own_id(self):
        """Tool names are discovered at runtime; the server is the node the operator configured."""
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"), name="github-mcp", mock=MockConfig(enabled=True)
        )
        (tool,) = self.discover(server, "create_issue")

        config = RunnableConfig(mock=RunMockConfig(exclude_ids={server.id}))

        assert tool.resolve_mock(config) is None, "excluding the server must reach the tools it owns"

    def test_the_server_can_be_excluded_by_its_own_name(self):
        server = MCPServer(
            connection=MCPSse(url="https://example.com/"), name="github-mcp", mock=MockConfig(enabled=True)
        )
        (tool,) = self.discover(server, "create_issue")

        config = RunnableConfig(mock=RunMockConfig(exclude_names={"github-mcp"}))

        assert tool.resolve_mock(config) is None

    def test_an_unmocked_server_leaves_its_tools_live(self):
        server = MCPServer(connection=MCPSse(url="https://example.com/"), name="github-mcp")
        connect_patch, session_patch = patch_discovery("create_issue")

        with connect_patch, session_patch:
            (tool,) = server.get_mcp_tools()

        assert tool.mock.enabled is False
        assert tool.resolve_mock(RunnableConfig()) is None


class TestPipedreamMocking:
    @staticmethod
    def build(**kwargs) -> Pipedream:
        return Pipedream(
            connection=PipedreamOAuth2(access_token="t", project_id="p"),
            external_user_id="user-1",
            action_id="slack-send-message",
            input_props={},
            configurable_props={},
            **kwargs,
        )

    def test_mocked_pipedream_action_does_not_fire(self):
        tool = self.build(mock=MockConfig(enabled=True, output="message sent to #general"))

        with patch.object(Pipedream, "execute", side_effect=AssertionError("Pipedream action ran")):
            result = tool.run(input_data={})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "[MOCKED] message sent to #general"

    def test_run_level_policy_sweeps_in_pipedream(self):
        tool = self.build()

        with patch.object(Pipedream, "execute", side_effect=AssertionError("Pipedream action ran")):
            result = tool.run(
                input_data={},
                config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            )

        assert result.output["is_mocked"] is True

    def test_locked_pipedream_action_survives_a_real_run(self):
        tool = self.build(mock=MockConfig(enabled=True, locked=True))

        with patch.object(Pipedream, "execute", side_effect=AssertionError("Pipedream action ran")):
            result = tool.run(input_data={}, config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)))

        assert result.status == RunnableStatus.SUCCESS


class TestHumanFeedbackToolMocking:
    """Mocking the human is how a human-in-the-loop workflow gets evaluated unattended.

    Distinct from `Node.approval`, which a mocked node skips: HumanFeedbackTool is an
    ordinary tool the agent chooses to call, so mocking it supplies a canned human answer.
    """

    def test_a_mocked_human_never_gets_prompted(self):
        tool = HumanFeedbackTool(mock=MockConfig(enabled=True, output="yes, ship it"))

        with patch.object(HumanFeedbackTool, "execute", side_effect=AssertionError("a human was prompted")):
            result = tool.run(input_data={"input": "Ship the release?"})

        assert result.status == RunnableStatus.SUCCESS
        assert result.output["content"] == "[MOCKED] yes, ship it"

    def test_the_canned_answer_can_depend_on_what_was_asked(self):
        tool = HumanFeedbackTool(mock=MockConfig(enabled=True, output="approved: {{ input.input }}"))

        with patch.object(HumanFeedbackTool, "execute", side_effect=AssertionError("a human was prompted")):
            result = tool.run(input_data={"input": "Ship the release?"})

        assert "approved: Ship the release?" in result.output["content"]

    def test_run_level_policy_sweeps_in_the_human(self):
        tool = HumanFeedbackTool()

        with patch.object(HumanFeedbackTool, "execute", side_effect=AssertionError("a human was prompted")):
            result = tool.run(
                input_data={"input": "Ship the release?"},
                config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            )

        assert result.output["is_mocked"] is True


def patch_discovery_failure():
    """Patch the connection so every discovery attempt fails like an unreachable server."""

    @contextlib.asynccontextmanager
    async def dead_connect(self):
        raise ConnectionError("connection refused")
        yield  # pragma: no cover

    return patch.object(MCPSse, "connect", new=dead_connect)


class TestUnreachableMCPServer:
    """A mocked server contributes no real calls, so unreachability must not kill the run."""

    def test_a_mocked_unreachable_server_degrades_instead_of_raising(self):
        server = MCPServer(
            connection=MCPSse(url="https://gone.example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, locked=True),
        )

        with patch_discovery_failure():
            tools = server.get_mcp_tools()

        assert tools == [], "no schemas means no tools, but the run must continue"

    def test_an_unmocked_unreachable_server_still_fails_loudly(self):
        """A real run must not silently lose its tools."""
        from dynamiq.nodes.agents.exceptions import ToolExecutionException

        server = MCPServer(connection=MCPSse(url="https://gone.example.com/"), name="github-mcp")

        with patch_discovery_failure():
            with pytest.raises(ToolExecutionException):
                server.get_mcp_tools()

    def test_an_agent_constructs_over_a_dead_mocked_server(self):
        """The whole point: a canvas with a pinned-off MCP server must load and run."""
        from dynamiq.connections import OpenAI as OpenAIConnection
        from dynamiq.nodes.agents import Agent
        from dynamiq.nodes.llms import OpenAI

        llm = OpenAI(connection=OpenAIConnection(api_key="test-key"), model="gpt-4o-mini")
        server = MCPServer(
            connection=MCPSse(url="https://gone.example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, locked=True),
        )

        with patch_discovery_failure():
            agent = Agent(name="ops", llm=llm, role="operate", tools=[server])

        assert agent.tools == [], "the dead mocked server contributes no tools, and nothing crashed"


class TestServerMockAcrossMapCloning:
    """Map clones an agent's whole subtree; a cloned MCPTool must still answer to its server.

    The owner link is a private attribute, so this guards against a clone implementation
    that drops private state and silently un-mocks every discovered tool per iteration.
    """

    def test_a_cloned_tool_still_reads_the_original_servers_mock(self):
        from dynamiq.nodes.cloning import regenerate_node_ids

        server = MCPServer(
            connection=MCPSse(url="https://example.com/"),
            name="github-mcp",
            mock=MockConfig(enabled=True, locked=True, output="held"),
        )
        connect_patch, session_patch = patch_discovery("create_issue")
        with connect_patch, session_patch:
            (tool,) = server.get_mcp_tools()

        clone = regenerate_node_ids(tool.clone(), {})

        assert clone.id != tool.id
        resolved = clone.resolve_mock(RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)))
        assert resolved is not None, "the clone lost its owner link — Map iterations would run for real"
        assert resolved.output == "held"

    def test_a_mock_set_on_the_server_after_cloning_reaches_existing_clones(self):
        from dynamiq.nodes.cloning import regenerate_node_ids

        server = MCPServer(connection=MCPSse(url="https://example.com/"), name="github-mcp")
        connect_patch, session_patch = patch_discovery("create_issue")
        with connect_patch, session_patch:
            (tool,) = server.get_mcp_tools()
        clone = regenerate_node_ids(tool.clone(), {})
        assert clone.resolve_mock(RunnableConfig()) is None

        server.mock = MockConfig(enabled=True, output="pinned late")

        assert clone.resolve_mock(RunnableConfig()) is not None, "owner is read live, not snapshotted"
