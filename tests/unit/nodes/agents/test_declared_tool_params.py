"""Node-level `tool_params_from_input` on an Agent: bind agent input keys to tool parameters."""

import json
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel, Field

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node, NodeDependency, NodeOutputReference
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


class DocsSearchInputSchema(BaseModel):
    query: str = Field(default="", description="What to look up.")
    user: str = Field(
        default="anonymous",
        description="Caller identity, supplied by the application.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    limit: int = Field(default=1, description="How many documents to return.")


TOOL_CALLS: list[DocsSearchInputSchema] = []


class DocsSearchTool(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "docs-search"
    description: str = "Searches internal docs."
    input_schema: ClassVar[type[DocsSearchInputSchema]] = DocsSearchInputSchema

    def execute(
        self, input_data: DocsSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        TOOL_CALLS.append(input_data)
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
        """input_mapping lifts another node's output into the agent input; the binding names it."""
        auth = DocsSearchTool(id="auth-1", name="auth-lookup")
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "auth_user"}}})
        agent.depends = [NodeDependency(auth)]
        agent.input_mapping = {"auth_user": NodeOutputReference(node=auth, output_key="user")}

        dep = RunnableResult(status=RunnableStatus.SUCCESS, input={}, output={"user": "alice@corp.com"})
        run_input = agent.transform_input(input_data={"input": "q"}, depends_result={auth.id: dep})

        assert run_input["auth_user"] == "alice@corp.com", "input_mapping must produce the key the binding names"

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


class TestBindingsAppliedDuringARun:
    """End-to-end through agent.run(), so the wiring inside execute() is covered too."""

    @staticmethod
    def _mock_llm(responses: list[str]):
        stream = iter(responses)

        def run(**kwargs):
            result = MagicMock(spec=RunnableResult)
            result.status = RunnableStatus.SUCCESS
            result.output = {"content": next(stream)}
            return result

        return run

    def _run(self, agent: Agent, run_input: dict) -> RunnableResult:
        agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
        agent.init_components()
        responses = [
            json.dumps({"thought": "search", "action": "docs-search", "action_input": {"query": "margins"}}),
            json.dumps({"thought": "done", "action": "finish", "action_input": "Q3 margin was 41%"}),
        ]
        TOOL_CALLS.clear()
        with patch.object(agent.llm, "run", side_effect=self._mock_llm(responses)):
            return agent.run(input_data=run_input)

    def test_bound_value_reaches_the_tool(self, test_llm):
        agent = build_agent(test_llm, {"by_name": {"docs-search": {"user": "user"}}})

        result = self._run(agent, {"input": "Q3 margins?", "user": "alice@corp.com"})

        assert result.status == RunnableStatus.SUCCESS
        assert [call.user for call in TOOL_CALLS] == ["alice@corp.com"]

    def test_without_bindings_the_tool_keeps_its_default(self, test_llm):
        agent = build_agent(test_llm)

        self._run(agent, {"input": "Q3 margins?", "user": "alice@corp.com"})

        assert [call.user for call in TOOL_CALLS] == ["anonymous"], "no binding -> input key must not leak in"


class TestSerialization:
    def test_to_dict_uses_aliases_and_round_trips(self, test_llm):
        bindings = {"by_name": {"docs-search": {"user": "user"}}, "global": {"tenant": "metadata.tenant"}}
        agent = build_agent(test_llm, bindings)

        dumped = agent.to_dict()["tool_params_from_input"]

        assert dumped["by_name"] == {"docs-search": {"user": "user"}}
        assert dumped["global"] == {"tenant": "metadata.tenant"}
        assert ToolParams.model_validate(dumped) == agent.tool_params_from_input

    def test_yaml_roundtrip_keeps_bindings_working(self, tmp_path):
        """Dump a workflow, reload it, and check the reloaded agent still binds its input."""
        llm = OpenAI(id="llm-1", connection=OpenAIConnection(id="conn-1", api_key="test-key"), model="gpt-4o")
        agent = Agent(
            id="docs-agent",
            name="docs-assistant",
            llm=llm,
            role="answer questions",
            tools=[DocsSearchTool(id="docs-tool")],
            tool_params_from_input={
                "by_name": {"docs-search": {"user": "user"}},
                "global": {"tenant": "metadata.tenant"},
            },
        )
        workflow = Workflow(id="wf", flow=Flow(id="flow", nodes=[agent]))

        yaml_path = tmp_path / "workflow.yaml"
        workflow.to_yaml_file(yaml_path)

        raw = yaml.safe_load(yaml_path.read_text())["nodes"]["docs-agent"]["tool_params_from_input"]
        assert raw == {
            "global": {"tenant": "metadata.tenant"},
            "by_name": {"docs-search": {"user": "user"}},
            "by_id": {},
        }, "YAML must carry the aliases, not the field names"

        loaded_agent = Workflow.from_yaml_file(str(yaml_path), init_components=True).flow.nodes[0]
        assert loaded_agent.tool_params_from_input == agent.tool_params_from_input

        run_input = {"input": "q", "user": "alice@corp.com", "metadata": {"tenant": "acme"}}
        params = loaded_agent._resolve_tool_params(loaded_agent.validate_input_schema(run_input))

        assert params.by_name_params["docs-search"] == {"user": "alice@corp.com"}
        assert params.global_params == {"tenant": "acme"}
