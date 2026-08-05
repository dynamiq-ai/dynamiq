"""Node-level `tool_params_from_input` on an Agent: bind agent input keys to tool parameters."""

from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


class DocsSearchInputSchema(BaseModel):
    query: str = Field(default="", description="What to look up.")
    user: str = Field(
        default="anonymous",
        description="Caller identity, supplied by the application.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    limit: int = Field(default=1, description="How many documents to return.")


class DocsSearchTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "docs-search"
    description: str = "Searches internal docs."
    input_schema: ClassVar[type[DocsSearchInputSchema]] = DocsSearchInputSchema

    def execute(
        self, input_data: DocsSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        return {"content": f"docs for {input_data.user}"}


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def build_agent(test_llm, bindings: dict | None = None) -> Agent:
    return Agent(
        name="docs-assistant",
        llm=test_llm,
        role="answer questions",
        tools=[DocsSearchTool()],
        tool_params_from_input=bindings or {},
    )


def resolve(agent: Agent, run_input: dict, inherited: ToolParams | None = None) -> ToolParams | None:
    """Run the agent's tool-param resolution the way execute() does."""
    input_data = agent.validate_input_schema(run_input)
    return agent._resolve_tool_params(input_data, inherited)


class TestBindings:
    def test_binds_input_key_to_tool_param(self, test_llm):
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user"}}})

        params = resolve(agent, {"input": "q", "user": "alice@corp.com"})

        assert params.by_name_params["docs-search"] == {"user": "alice@corp.com"}

    def test_binds_under_a_different_name(self, test_llm):
        """The tool's parameter and the input key need not share a name."""
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "caller_email"}}})

        params = resolve(agent, {"input": "q", "caller_email": "alice@corp.com"})

        assert params.by_name_params["docs-search"] == {"user": "alice@corp.com"}

    def test_value_type_is_preserved(self, test_llm):
        agent = build_agent(test_llm, {"global": {"limit": "limit", "opts": "opts"}})

        params = resolve(agent, {"input": "q", "limit": 5, "opts": {"deep": True}})

        assert params.global_params == {"limit": 5, "opts": {"deep": True}}

    def test_dotted_key_walks_into_nested_input(self, test_llm):
        agent = build_agent(test_llm, {"global": {"tenant": "metadata.tenant"}})

        params = resolve(agent, {"input": "q", "metadata": {"tenant": "acme"}})

        assert params.global_params == {"tenant": "acme"}

    def test_binds_a_dependency_output(self, test_llm):
        """A value from another node, once input_mapping/transformer put it in the agent input."""
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "auth_user"}}})
        dep = RunnableResult(status=RunnableStatus.SUCCESS, input={}, output={"user": "alice@corp.com"})
        run_input = agent.transform_input(
            input_data={"input": "q"},
            depends_result={"auth_1": dep},
        ) | {"auth_user": "alice@corp.com"}

        params = agent._resolve_tool_params(agent.validate_input_schema(run_input))

        assert params.by_name_params["docs-search"] == {"user": "alice@corp.com"}

    def test_absent_input_key_leaves_the_param_unset(self, test_llm):
        """No `user` in the agent input -> the tool's own default stands."""
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user"}}})

        params = resolve(agent, {"input": "q"})

        assert params.by_name_params["docs-search"] == {}

    def test_no_bindings_and_no_input_yields_nothing(self, test_llm):
        assert resolve(build_agent(test_llm), {"input": "q"}) is None

    def test_run_input_overrides_bindings(self, test_llm):
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user", "limit": "limit"}}})

        params = resolve(
            agent,
            {
                "input": "q",
                "user": "alice@corp.com",
                "limit": 3,
                "tool_params": {"by_name": {"docs-search": {"user": "override@corp.com"}}},
            },
        )

        assert params.by_name_params["docs-search"] == {"user": "override@corp.com", "limit": 3}

    def test_inherited_params_survive_an_empty_run_input(self, test_llm):
        """Params handed down by a parent agent must not be wiped by the child's empty input."""
        inherited = ToolParams(**{"global": {"from_parent": "yes"}})

        params = resolve(build_agent(test_llm), {"input": "q"}, inherited=inherited)

        assert params.global_params == {"from_parent": "yes"}

    def test_inherited_params_override_bindings_and_lose_to_run_input(self, test_llm):
        agent = build_agent(test_llm, {"global": {"source": "declared_source", "only_declared": "declared_only"}})
        inherited = ToolParams(**{"global": {"source": "parent", "only_parent": 2}})

        params = resolve(
            agent,
            {
                "input": "q",
                "declared_source": "declared",
                "declared_only": 1,
                "tool_params": {"global": {"source": "run_input"}},
            },
            inherited=inherited,
        )

        assert params.global_params == {"source": "run_input", "only_declared": 1, "only_parent": 2}


class TestBoundValuesReachTheTool:
    def test_merged_into_tool_input(self, test_llm):
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user"}}})
        tool = agent.tools[0]

        params = resolve(agent, {"input": "q", "user": "alice@corp.com"})
        content, _, _ = agent._run_tool(tool=tool, tool_input={"query": "margins"}, config=None, tool_params=params)

        assert content == "docs for alice@corp.com"

    def test_llm_supplied_value_cannot_override_the_binding(self, test_llm):
        """`user` is hidden from the agent; a value the LLM invents is replaced by the bound one."""
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user"}}})
        tool = agent.tools[0]

        params = resolve(agent, {"input": "q", "user": "alice@corp.com"})
        content, _, _ = agent._run_tool(
            tool=tool,
            tool_input={"query": "margins", "user": "mallory@evil.com"},
            config=None,
            tool_params=params,
        )

        assert content == "docs for alice@corp.com"


class TestSerialization:
    def test_to_dict_uses_aliases_and_round_trips(self, test_llm):
        bindings = {"by_name": {"docs-search": {"user": "user"}}, "global": {"tenant": "metadata.tenant"}}
        agent = build_agent(test_llm, bindings)

        dumped = agent.to_dict()["tool_params_from_input"]

        assert dumped["by_name"] == {"docs-search": {"user": "user"}}
        assert dumped["global"] == {"tenant": "metadata.tenant"}
        assert ToolParams.model_validate(dumped) == agent.tool_params_from_input
