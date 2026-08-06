"""A tool's own `input_transformer`, resolved by the agent that calls it.

Inside an agent a tool has no dependencies of its own, so the agent resolves the tool's selector
against its own input (run input + dependency results) and merges the result over the arguments the
model produced.
"""

import json
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.components import schema_generator
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import InputTransformer, Node, NodeDependency
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus

TOOL_CALLS: list["DocsSearchInputSchema"] = []


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
        TOOL_CALLS.append(input_data)
        return {"content": f"docs for {input_data.user}"}


class AuthLookupSchema(BaseModel):
    input: str = Field(default="")


class AuthLookup(Node):
    """Upstream node whose output a tool wants to reach."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "auth-lookup"
    description: str = "Resolves the caller."
    input_schema: ClassVar[type[AuthLookupSchema]] = AuthLookupSchema

    def execute(self, input_data: AuthLookupSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        # A realistically chunky output: none of it should be retained by the agent.
        return {"user": "alice@corp.com", "documents": [{"text": "x" * 500} for _ in range(20)]}


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


@pytest.fixture(autouse=True)
def clear_tool_calls():
    TOOL_CALLS.clear()
    yield
    TOOL_CALLS.clear()


def mocked_llm(agent: Agent, tool_input: dict | None = None):
    """Drive one docs-search call, then finish."""
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    stream = iter(
        [
            json.dumps(
                {
                    "thought": "search",
                    "action": "docs-search",
                    "action_input": tool_input if tool_input is not None else {"query": "margins"},
                }
            ),
            json.dumps({"thought": "done", "action": "finish", "action_input": "Q3 margin was 41%"}),
        ]
    )

    def run(**kwargs):
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(stream)}
        return result

    return patch.object(agent.llm, "run", side_effect=run)


class TestToolSelectorInAWorkflow:
    """Full Workflow runs — the agent node is driven the way the flow drives it."""

    @staticmethod
    def build(test_llm, tool_selector: dict | None, agent_selector: dict | None = None) -> tuple[Workflow, Agent]:
        auth = AuthLookup(id="auth")
        tool = DocsSearchTool(
            input_transformer=InputTransformer(selector=tool_selector) if tool_selector else InputTransformer()
        )
        agent = Agent(
            id="agent",
            name="docs-assistant",
            llm=test_llm,
            role="answer questions",
            tools=[tool],
            depends=[NodeDependency(auth)],
            input_transformer=InputTransformer(selector=agent_selector) if agent_selector else InputTransformer(),
        )
        return Workflow(flow=Flow(nodes=[auth, agent])), agent

    def test_tool_pulls_a_value_from_another_node(self, test_llm):
        workflow, agent = self.build(test_llm, {"user": "$.auth.output.user"})

        with mocked_llm(agent):
            result = workflow.run(input_data={"input": "Q3 margins?"})

        assert result.status == RunnableStatus.SUCCESS
        assert [call.user for call in TOOL_CALLS] == ["alice@corp.com"]
        assert [call.query for call in TOOL_CALLS] == ["margins"], "the model's own argument must survive"

    def test_agent_selector_does_not_hide_the_dependency_from_the_tool(self, test_llm):
        """The agent narrows its own input to `input`; the tool still reaches `auth`."""
        workflow, agent = self.build(
            test_llm,
            tool_selector={"user": "$.auth.output.user"},
            agent_selector={"input": "$.input"},
        )

        with mocked_llm(agent):
            result = workflow.run(input_data={"input": "Q3 margins?"})

        assert result.status == RunnableStatus.SUCCESS
        assert [call.user for call in TOOL_CALLS] == ["alice@corp.com"]

    def test_tool_pulls_a_value_from_the_run_input(self, test_llm):
        workflow, agent = self.build(test_llm, {"user": "$.user"})

        with mocked_llm(agent):
            result = workflow.run(input_data={"input": "Q3 margins?", "user": "bob@corp.com"})

        assert result.status == RunnableStatus.SUCCESS
        assert [call.user for call in TOOL_CALLS] == ["bob@corp.com"]

    def test_only_selected_values_are_retained(self, test_llm):
        """The upstream node returns a chunky payload; the agent keeps only what was selected."""
        workflow, agent = self.build(test_llm, {"user": "$.auth.output.user"})

        with mocked_llm(agent):
            workflow.run(input_data={"input": "Q3 margins?"})

        assert list(agent._tool_input_overrides.values()) == [{"user": "alice@corp.com"}]

    def test_unmatched_path_leaves_the_tool_default(self, test_llm):
        workflow, agent = self.build(test_llm, {"user": "$.auth.output.nope"})

        with mocked_llm(agent):
            workflow.run(input_data={"input": "Q3 margins?"})

        assert [call.user for call in TOOL_CALLS] == ["anonymous"], "unmatched path must not inject None"

    def test_tool_without_a_selector_is_untouched(self, test_llm):
        workflow, agent = self.build(test_llm, tool_selector=None)

        with mocked_llm(agent):
            workflow.run(input_data={"input": "Q3 margins?", "user": "bob@corp.com"})

        assert [call.user for call in TOOL_CALLS] == ["anonymous"], "an input key must not leak into a tool"
        assert agent._tool_input_overrides == {}, "nothing is resolved or retained without a selector"

    def test_model_supplied_value_is_overridden(self, test_llm):
        """`user` is hidden from the agent; a value the model invents loses to the selector."""
        workflow, agent = self.build(test_llm, {"user": "$.auth.output.user"})

        with mocked_llm(agent, tool_input={"query": "margins", "user": "mallory@evil.com"}):
            workflow.run(input_data={"input": "Q3 margins?"})

        assert [call.user for call in TOOL_CALLS] == ["alice@corp.com"]

    def test_selector_target_is_neither_visible_nor_settable_by_the_model(self, test_llm):
        """A selector-filled parameter stays out of the schemas the model sees, and stays unset
        when the model invents it and no selector fills it."""
        workflow, agent = self.build(test_llm, tool_selector=None)
        tools = agent._runtime_tools

        advertised = json.dumps(
            [
                schema_generator.generate_input_formats(tools, agent.sanitize_tool_name),
                schema_generator.generate_structured_output_schemas(tools, agent.sanitize_tool_name, False),
                schema_generator.generate_function_calling_schemas(tools, False, agent.sanitize_tool_name),
            ],
            default=str,
        )
        assert "query" in advertised, "control: a visible parameter does appear in the schemas"
        assert "user" not in advertised, "a hidden parameter must not appear in any schema the model sees"

        with mocked_llm(agent, tool_input={"query": "margins", "user": "mallory@evil.com"}):
            workflow.run(input_data={"input": "Q3 margins?"})

        assert [call.user for call in TOOL_CALLS] == ["anonymous"], "a model-supplied hidden value must be dropped"


class TestPrecedence:
    def test_run_input_tool_params_win_over_the_selector(self, test_llm):
        tool = DocsSearchTool(input_transformer=InputTransformer(selector={"user": "$.user", "limit": "$.page_size"}))
        agent = Agent(name="docs-assistant", llm=test_llm, role="answer questions", tools=[tool])
        workflow = Workflow(flow=Flow(nodes=[agent]))

        with mocked_llm(agent):
            workflow.run(
                input_data={
                    "input": "Q3 margins?",
                    "user": "bob@corp.com",
                    "page_size": 7,
                    "tool_params": {"by_name": {"docs-search": {"user": "override@corp.com"}}},
                }
            )

        assert [(call.user, call.limit) for call in TOOL_CALLS] == [
            ("override@corp.com", 7)
        ], "tool_params override the selector; untouched selector values remain"
