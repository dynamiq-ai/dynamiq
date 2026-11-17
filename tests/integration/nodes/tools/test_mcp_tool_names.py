"""The names an agent offers the model its tools under, and the tool each name reaches."""

import contextlib
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from dynamiq import Workflow
from dynamiq.connections.connections import MCPSse
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.hooks import ToolPolicyHook
from dynamiq.nodes.tools.mcp import MCPServer
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from tests.helpers.agent_hooks import (
    ALL_MODES,
    EmailTool,
    build_agent,
    final_step,
    make_llm,
    run_scripted,
    scripted,
    tool_step,
)

FC = InferenceMode.FUNCTION_CALLING
GITHUB, GITLAB, LINEAR = "https://github.example/sse", "https://gitlab.example/sse", "https://linear.example/sse"
QUERY_SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}


class FakeServers:
    """MCP servers by URL: each lists the tools it is given and records every call as (url, tool, arguments)."""

    def __init__(self, tools_by_url: dict[str, list[str]]):
        self.tools_by_url = tools_by_url
        self.calls: list[tuple[str, str, dict]] = []

    @contextlib.contextmanager
    def serving(self):
        servers = self

        @contextlib.asynccontextmanager
        async def connect(connection, headers=None):
            yield (connection.url, None)

        class Session:
            def __init__(self, url, _write, *args, **kwargs):
                self.url = url

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                return None

            async def list_tools(self, params=None):
                return ListToolsResult(
                    tools=[
                        Tool(name=name, description=f"{name} on {self.url}", inputSchema=QUERY_SCHEMA)
                        for name in servers.tools_by_url[self.url]
                    ]
                )

            async def call_tool(self, name, arguments):
                servers.calls.append((self.url, name, arguments))
                return CallToolResult(content=[TextContent(type="text", text=f"{name} done")])

        with patch.object(MCPSse, "connect", new=connect), patch("mcp.ClientSession", new=Session):
            yield self


def mcp_server(name: str, url: str, **kwargs) -> MCPServer:
    return MCPServer(name=name, connection=MCPSse(url=url), **kwargs)


def function_names(agent: Agent) -> list[str]:
    return [
        schema["function"]["name"] for schema in agent._tools if schema["function"]["name"] != "provide_final_answer"
    ]


def test_a_tool_name_two_servers_share_is_offered_apart_and_each_call_reaches_its_own_server():
    fake = FakeServers({GITHUB: ["search", "create_issue"], LINEAR: ["search"]})
    with fake.serving():
        agent = Agent(
            name="a",
            llm=make_llm(),
            tools=[mcp_server("github", GITHUB), mcp_server("linear", LINEAR)],
            inference_mode=FC,
        )
        result, _ = run_scripted(agent, FC, [tool_step("linear_search", q="bug"), final_step("done")])

    assert result.status == RunnableStatus.SUCCESS
    assert sorted(function_names(agent)) == ["create_issue", "github_search", "linear_search"]
    assert fake.calls == [(LINEAR, "search", {"q": "bug"})]
    assert [tool.name for tool in agent.tools] == ["search", "create_issue", "search"]


def test_tools_whose_names_collide_with_nothing_keep_the_names_they_had():
    fake = FakeServers({GITHUB: ["search", "list_repos"]})
    specialist = Agent(name="Billing Specialist", llm=make_llm(), role="answer billing questions")
    with fake.serving():
        agent = Agent(
            name="a", llm=make_llm(), tools=[mcp_server("github", GITHUB), EmailTool(), specialist], inference_mode=FC
        )

    assert function_names(agent) == [agent.sanitize_tool_name(tool.name) for tool in agent.tools]
    assert function_names(agent) == ["search", "list_repos", "send-email", "Billing-Specialist"]
    assert [line.split(":")[0] for line in agent.tool_description.splitlines()] == [
        "- search",
        "- list_repos",
        "- send-email",
        "- Billing Specialist",
    ]


def test_a_name_longer_than_function_calling_accepts_is_cut_and_still_calls_the_tool_by_its_own_name():
    long_name = "workspace_" + "very_long_segment_" * 5 + "create"
    fake = FakeServers({GITHUB: [long_name]})
    with fake.serving():
        agent = Agent(name="a", llm=make_llm(), tools=[mcp_server("github", GITHUB)], inference_mode=FC)
        [offered] = function_names(agent)
        run_scripted(agent, FC, [tool_step(offered, q="x"), final_step("done")])
        text_mode = Agent(name="a", llm=make_llm(), tools=[mcp_server("github", GITHUB)])

    assert len(offered) <= 64
    assert agent.sanitize_tool_name(offered) == offered
    assert fake.calls == [(GITHUB, long_name, {"q": "x"})]
    # Only function names have a length limit; a text prompt names the tool in full, as before.
    assert text_mode.tool_names == long_name


def test_a_tool_policy_hides_the_tool_of_every_server_that_has_it():
    fake = FakeServers({GITHUB: ["delete_repo", "search"], GITLAB: ["delete_repo"]})
    hook = ToolPolicyHook(deny=True, tools=["delete_repo"])
    with fake.serving():
        agent = build_agent(
            [hook], tools=[mcp_server("github", GITHUB), mcp_server("gitlab", GITLAB)], inference_mode=FC
        )

    assert function_names(agent) == ["search"]


@pytest.mark.parametrize("mode", ALL_MODES, ids=lambda mode: mode.value)
def test_a_tool_policy_rules_a_tool_the_model_knows_by_another_name(mode):
    fake = FakeServers({GITHUB: ["delete_repo"], GITLAB: ["delete_repo"]})
    hook = ToolPolicyHook(tools=["delete_repo"], allow_if={"metadata.role": "admin"}, message="Admins only.")
    steps = [tool_step("github_delete_repo", q="prod-db"), final_step("done")]
    with fake.serving():
        agent = build_agent(
            [hook], tools=[mcp_server("github", GITHUB), mcp_server("gitlab", GITLAB)], inference_mode=mode
        )
        _, viewer = run_scripted(agent, mode, steps, trusted={"metadata": {"role": "viewer"}})
        refused = list(fake.calls)
        run_scripted(agent, mode, steps, trusted={"metadata": {"role": "admin"}})

    assert refused == []
    assert "Admins only." in viewer.sent_text(1)
    assert fake.calls == [(GITHUB, "delete_repo", {"q": "prod-db"})]


def test_an_agent_sharing_tool_objects_with_another_keeps_its_own_names():
    fake = FakeServers({GITHUB: ["search"], LINEAR: ["search"]})
    with fake.serving():
        github, linear = mcp_server("github", GITHUB), mcp_server("linear", LINEAR)
        researcher = Agent(name="researcher", llm=make_llm(), tools=[github], inference_mode=FC)
        manager = Agent(name="manager", llm=make_llm(), tools=[github, linear, researcher], inference_mode=FC)
        result, _ = run_scripted(researcher, FC, [tool_step("search", q="bug"), final_step("done")])

    assert sorted(function_names(manager)) == ["github_search", "linear_search", "researcher"]
    assert manager.tool_by_names["github_search"] is researcher.tool_by_names["search"]
    assert function_names(researcher) == ["search"]
    assert result.status == RunnableStatus.SUCCESS
    assert fake.calls == [(GITHUB, "search", {"q": "bug"})]


def test_listed_tools_saved_to_yaml_are_offered_and_called_the_same_after_loading(tmp_path):
    fake = FakeServers({GITHUB: ["search"], LINEAR: ["search"]})
    path = tmp_path / "workflow.yaml"
    with fake.serving():
        tools = mcp_server("github", GITHUB).get_mcp_tools() + mcp_server("linear", LINEAR).get_mcp_tools()
        agent = Agent(id="agent", name="a", llm=make_llm(), tools=tools, inference_mode=FC)
        Workflow(id="wf", flow=Flow(id="flow", nodes=[agent])).to_yaml_file(path)
        loaded = Workflow.from_yaml_file(str(path), init_components=True).flow.nodes[0]
        run_scripted(loaded, FC, [tool_step("linear_search", q="bug"), final_step("done")])

    assert sorted(function_names(loaded)) == sorted(function_names(agent)) == ["github_search", "linear_search"]
    assert [tool.name for tool in loaded.tools] == ["search", "search"]
    assert fake.calls == [(LINEAR, "search", {"q": "bug"})]


def test_three_servers_left_at_the_default_name_each_get_a_name_of_their_own():
    urls = [f"https://s{i}.example/sse" for i in (1, 2, 3)]
    fake = FakeServers({url: ["search"] for url in urls})

    def build() -> Agent:
        servers = [MCPServer(id=f"mcp-server-{i}", connection=MCPSse(url=url)) for i, url in enumerate(urls, 1)]
        return Agent(name="a", llm=make_llm(), tools=servers, inference_mode=FC)

    with fake.serving():
        agent = build()
        names = function_names(agent)
        for name in names:
            run_scripted(agent, FC, [tool_step(name, q=name), final_step("done")])
        rebuilt = function_names(build())

    assert len(set(names)) == 3
    assert all(name.startswith("mcp_search") for name in names)
    assert sorted(url for url, _, _ in fake.calls) == urls
    assert rebuilt == names


def test_a_tool_listed_for_one_run_leaves_the_names_of_the_agents_own_tools_alone():
    fake = FakeServers({GITHUB: ["search"], LINEAR: ["search"]})
    with fake.serving():
        agent = Agent(
            name="a",
            llm=make_llm(),
            tools=[mcp_server("github", GITHUB), mcp_server("linear", LINEAR, discover_per_run=True)],
            inference_mode=FC,
        )
        run_scripted(agent, FC, [tool_step("linear_search", q="bug"), final_step("done")])
        run_scripted(agent, FC, [tool_step("search", q="bug"), final_step("done")])

    assert function_names(agent) == ["search"]
    assert fake.calls == [(LINEAR, "search", {"q": "bug"}), (GITHUB, "search", {"q": "bug"})]


def test_a_sub_agent_with_a_long_name_keeps_its_memory_scope():
    # The scope is made from the sub-agent's name; shortening it like a function name would orphan saved memory.
    child = Agent(
        name="Customer Support Escalation Specialist for Enterprise Billing Disputes",
        llm=make_llm(),
        role="settle billing disputes",
        memory=Memory(backend=InMemory()),
    )
    parent = build_agent(tools=[child], inference_mode=FC)
    [offered] = function_names(parent)

    with scripted(child, InferenceMode.DEFAULT, [final_step("Refund approved.")]):
        result, _ = run_scripted(
            parent,
            FC,
            [tool_step(offered, input="Dispute 42"), final_step("done")],
            input_extra={"user_id": "u1", "session_id": "s1"},
        )

    scope = "Customer-Support-Escalation-Specialist-for-Enterprise-Billing-Disputes"
    assert result.status == RunnableStatus.SUCCESS
    assert len(offered) <= 64
    saved = child.memory.get_all()
    assert saved
    assert {(m.metadata["user_id"], m.metadata["session_id"]) for m in saved} == {(f"u1:{scope}", f"s1:{scope}")}
