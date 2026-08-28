"""Mocked tools, driven by a real agent loop.

The point of these tests is the agent's perspective: a mocked tool must be indistinguishable
from a real one at the reasoning layer, while never touching the outside world.
"""

import json
from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms.base import FallbackConfig
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.mocking import DEFAULT_MOCK_MARKER, MockConfig, MockPolicy, RunMockConfig

CHARGES: list[float] = []


class ChargeCardSchema(BaseModel):
    amount: float = Field(default=0.0, description="Amount to charge.")


class ChargeCardTool(Node):
    """Stands in for any tool with a real, irreversible side effect."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "charge-card"
    description: str = "Charges a customer's card."
    input_schema: ClassVar[type[ChargeCardSchema]] = ChargeCardSchema

    def execute(self, input_data: ChargeCardSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        CHARGES.append(input_data.amount)
        return {"content": f"charged {input_data.amount}"}


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


@pytest.fixture(autouse=True)
def clear_charges():
    CHARGES.clear()
    yield
    CHARGES.clear()


def drive_agent(agent: Agent, seen: list[str]):
    """Make the agent call charge-card once, then finish. Records every prompt it sees."""
    agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
    stream = iter(
        [
            json.dumps({"thought": "charge", "action": "charge-card", "action_input": {"amount": 42}}),
            json.dumps({"thought": "done", "action": "finish", "action_input": "All set."}),
        ]
    )

    def run(**kwargs):
        seen.append(str(kwargs.get("prompt")))
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": next(stream)}
        return result

    return patch.object(agent.llm, "run", side_effect=run)


def build_agent(test_llm, tool: ChargeCardTool) -> Agent:
    return Agent(id="agent", name="billing", llm=test_llm, role="handle billing", tools=[tool])


class TestAgentWithMockedTool:
    def test_agent_completes_without_the_tool_ever_firing(self, test_llm):
        agent = build_agent(test_llm, ChargeCardTool(mock=MockConfig(enabled=True)))
        seen: list[str] = []

        with drive_agent(agent, seen):
            result = agent.run(input_data={"input": "charge the customer 42"})

        assert result.status == RunnableStatus.SUCCESS
        assert CHARGES == [], "a mocked tool must not reach the outside world"
        assert result.output["content"] == "All set."

    def test_the_mocked_observation_reaches_the_model(self, test_llm):
        tool = ChargeCardTool(mock=MockConfig(enabled=True, output="payment accepted, ref TEST-1"))
        agent = build_agent(test_llm, tool)
        seen: list[str] = []

        with drive_agent(agent, seen):
            agent.run(input_data={"input": "charge the customer 42"})

        follow_up = "\n".join(seen[1:])
        assert "payment accepted, ref TEST-1" in follow_up
        assert DEFAULT_MOCK_MARKER in follow_up

    def test_mock_response_can_reference_the_models_arguments(self, test_llm):
        tool = ChargeCardTool(mock=MockConfig(enabled=True, output="would charge {{ input.amount }}"))
        agent = build_agent(test_llm, tool)
        seen: list[str] = []

        with drive_agent(agent, seen):
            agent.run(input_data={"input": "charge the customer 42"})

        assert "would charge 42" in "\n".join(seen[1:])

    def test_injected_tool_error_is_recoverable_for_the_agent(self, test_llm):
        tool = ChargeCardTool(mock=MockConfig(enabled=True, error="card network timeout"))
        agent = build_agent(test_llm, tool)
        seen: list[str] = []

        with drive_agent(agent, seen):
            result = agent.run(input_data={"input": "charge the customer 42"})

        assert result.status == RunnableStatus.SUCCESS, "the agent should recover and finish"
        assert CHARGES == []
        assert "card network timeout" in "\n".join(seen[1:])


class TestMockedLLMDoesNotFallBack:
    """An injected LLM failure must not spend money on the fallback provider."""

    def test_fallback_is_skipped_when_the_failure_was_simulated(self, test_llm):
        fallback = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o-mini")
        primary = OpenAI(
            connection=OpenAIConnection(api_key="test-api-key"),
            model="gpt-4o",
            fallback=FallbackConfig(enabled=True, llm=fallback),
            mock=MockConfig(enabled=True, error="rate limited"),
        )

        with patch.object(fallback, "run_sync") as fallback_run:
            result = primary.run(input_data={}, prompt=Prompt(messages=[Message(role="user", content="hi")]))

        assert result.status == RunnableStatus.FAILURE
        fallback_run.assert_not_called()

    def test_a_real_failure_still_falls_back(self, test_llm):
        """The guard must be narrow: only simulated failures skip the fallback."""
        fallback = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o-mini")
        primary = OpenAI(
            connection=OpenAIConnection(api_key="test-api-key"),
            model="gpt-4o",
            fallback=FallbackConfig(enabled=True, llm=fallback),
        )

        with patch.object(primary, "execute", side_effect=RuntimeError("real outage")):
            with patch.object(fallback, "run_sync") as fallback_run:
                fallback_run.return_value = RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "ok"})
                primary.run(input_data={}, prompt=Prompt(messages=[Message(role="user", content="hi")]))

        fallback_run.assert_called_once()


class TestMockedSubAgentTool:
    """Mocking the delegation wrapper must stop the whole sub-agent."""

    @staticmethod
    def build(test_llm, mock: MockConfig) -> tuple[Agent, OpenAI]:
        """The child gets its own LLM, so 'the sub-agent never ran' is directly observable."""
        child_llm = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o-mini")
        child = Agent(id="child", name="specialist", llm=child_llm, role="do the work", tools=[ChargeCardTool()])
        wrapper = SubAgentTool(agent=child, name="specialist", description="Delegates billing work.", mock=mock)
        parent = Agent(id="parent", name="coordinator", llm=test_llm, role="delegate", tools=[wrapper])
        return parent, child_llm

    def drive_delegation(self, agent: Agent, seen: list[str]):
        agent.inference_mode = InferenceMode.STRUCTURED_OUTPUT
        stream = iter(
            [
                json.dumps({"thought": "delegate", "action": "specialist", "action_input": {"input": "charge 42"}}),
                json.dumps({"thought": "done", "action": "finish", "action_input": "Delegated."}),
            ]
        )

        def run(**kwargs):
            seen.append(str(kwargs.get("prompt")))
            result = MagicMock(spec=RunnableResult)
            result.status = RunnableStatus.SUCCESS
            result.output = {"content": next(stream)}
            return result

        return patch.object(agent.llm, "run", side_effect=run)

    def test_mocked_sub_agent_tool_suppresses_the_whole_delegation(self, test_llm):
        parent, child_llm = self.build(test_llm, MockConfig(enabled=True, output="sub-agent skipped"))
        seen: list[str] = []

        with patch.object(child_llm, "run") as child_llm_run:
            with self.drive_delegation(parent, seen):
                result = parent.run(input_data={"input": "delegate the charge"})

        assert result.status == RunnableStatus.SUCCESS
        child_llm_run.assert_not_called()
        assert CHARGES == [], "the sub-agent must not reason at all, nor its own tools fire"
        assert "sub-agent skipped" in "\n".join(seen[1:])

    def test_locked_sub_agent_mock_survives_a_none_policy_run(self, test_llm):
        parent, child_llm = self.build(test_llm, MockConfig(enabled=True, locked=True))

        with patch.object(child_llm, "run") as child_llm_run:
            with self.drive_delegation(parent, []):
                parent.run(
                    input_data={"input": "delegate the charge"},
                    config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)),
                )

        child_llm_run.assert_not_called()
        assert CHARGES == []


class TestMockedDelegationDoesNotEchoCallerContext:
    """Per-agent ids and metadata are injected for the sub-agent to consume, not to be shown.

    A mocked delegation runs on the wrapper, whose schema is `extra="allow"`, so anything
    injected would be echoed back into the parent's transcript and trace.
    """

    CONTEXT = {
        "user_id": "user-8842",
        "session_id": "sess-privileged-7f31",
        "metadata": {"tenant": "acme", "internal_billing_ref": "BR-99213"},
    }

    def build(self, test_llm, mock: MockConfig) -> tuple[Agent, Agent, SubAgentTool]:
        child_llm = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o-mini")
        child = Agent(id="child", name="specialist", llm=child_llm, role="do the work")
        wrapper = SubAgentTool(agent=child, name="specialist", description="Delegates.", mock=mock)
        parent = Agent(id="parent", name="coordinator", llm=test_llm, role="delegate", tools=[wrapper])
        parent._current_call_context = dict(self.CONTEXT)
        return parent, child, wrapper

    def test_the_suppressed_call_description_carries_no_injected_context(self, test_llm):
        """Zero-config mock: `describe_skipped_call` renders the input the node would have run with."""
        parent, _child, wrapper = self.build(test_llm, MockConfig(enabled=True))

        content, _files, meta = parent._run_tool(wrapper, {"input": "go"}, RunnableConfig(), tool_run_id="rid")

        assert meta["is_mocked"] is True
        for leaked in ("user_id", "session_id", "user-8842", "sess-privileged-7f31", "BR-99213"):
            assert leaked not in content, f"{leaked} reached the parent transcript"
        assert "'input': 'go'" in content, "the model's own arguments are still described"

    def test_a_real_delegation_still_receives_the_injected_context(self, test_llm):
        """The guard must be narrow: an unmocked delegation is unchanged."""
        parent, _child, wrapper = self.build(test_llm, MockConfig())

        with patch.object(Agent, "run") as child_run:
            child_run.return_value = RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "did it"})
            parent._run_tool(wrapper, {"input": "go"}, RunnableConfig(), tool_run_id="rid")

        delegated_input = child_run.call_args.kwargs["input_data"]
        assert delegated_input["user_id"] == "user-8842:specialist"
        assert delegated_input["session_id"] == "sess-privileged-7f31:specialist"
        assert delegated_input["metadata"] == self.CONTEXT["metadata"]


class TestMockedSubAgentToolInFactoryMode:
    """Factory mode is the only mode that allows parallel delegation, so A/B runs land here."""

    BLUEPRINT = {
        "connections": {"openai-conn": {"type": "dynamiq.connections.OpenAI", "api_key": "test-key"}},
        "name": "Researcher",
        "llm": {
            "id": "openai-conn",
            "type": "dynamiq.nodes.llms.OpenAI",
            "connection": "openai-conn",
            "model": "gpt-4o",
        },
        "role": "You are a research agent.",
        "tools": [],
    }

    def build(self, test_llm, mock: MockConfig) -> tuple[Agent, SubAgentTool]:
        wrapper = SubAgentTool(name="specialist", description="Delegates.", agent_factory=self.BLUEPRINT, mock=mock)
        parent = Agent(id="parent", name="coordinator", llm=test_llm, role="delegate", tools=[wrapper])
        return parent, wrapper

    def test_a_mocked_factory_delegation_does_not_crash(self, test_llm):
        """The factory clone path must not be handed the sub-agent that mocking never built."""
        parent, wrapper = self.build(test_llm, MockConfig(enabled=True, output="simulated delegation"))

        content, _files, meta = parent._run_tool(wrapper, {"input": "go"}, RunnableConfig(), tool_run_id="rid")

        assert content == "[MOCKED] simulated delegation"
        assert meta["is_mocked"] is True

    def test_an_unmocked_factory_delegation_still_builds_its_agent(self, test_llm):
        """The guard must be narrow — factory mode is untouched when nothing is mocked."""
        parent, wrapper = self.build(test_llm, MockConfig())

        with patch.object(Agent, "run") as child_run:
            child_run.return_value = RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "did it"})
            content, _files, _meta = parent._run_tool(wrapper, {"input": "go"}, RunnableConfig(), tool_run_id="rid")

        child_run.assert_called_once()
        assert content == "did it"


class TestAgentABComparison:
    """The A/B case: one workflow definition, two runs, one of which touches nothing."""

    def run_once(self, test_llm, config: RunnableConfig | None) -> RunnableResult:
        agent = build_agent(test_llm, ChargeCardTool())
        with drive_agent(agent, []):
            return agent.run(input_data={"input": "charge the customer 42"}, config=config)

    def test_control_run_executes_the_tool(self, test_llm):
        result = self.run_once(test_llm, None)

        assert result.status == RunnableStatus.SUCCESS
        assert CHARGES == [42.0]

    def test_variant_run_under_all_policy_executes_nothing(self, test_llm):
        result = self.run_once(test_llm, RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)))

        assert result.status == RunnableStatus.SUCCESS
        assert CHARGES == []

    def test_all_policy_leaves_the_agents_own_llm_alone(self, test_llm):
        """Only tools are swept in by default, so the agent still reasons for real."""
        seen: list[str] = []
        agent = build_agent(test_llm, ChargeCardTool())

        with drive_agent(agent, seen):
            agent.run(
                input_data={"input": "charge the customer 42"},
                config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL)),
            )

        assert len(seen) == 2, "the agent's LLM must still be called for each reasoning step"

    def test_id_based_exclusion_survives_tool_cloning(self, test_llm):
        """Parallel tool calls clone the tool and regenerate its id; the exclusion must follow."""
        tool = ChargeCardTool()
        agent = build_agent(test_llm, tool)
        config = RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_ids={tool.id}))

        cloned, cloned_config = agent._clone_tool_for_execution(tool, config)

        assert cloned.id != tool.id, "the clone is expected to get a fresh id"
        assert cloned.resolve_mock(cloned_config) is None, "the excluded tool must still run for real"

    def test_id_based_exclusion_survives_cloning_of_a_nested_tool(self, test_llm):
        """Cloning a sub-agent re-ids its tools too; an exclusion on one must follow it down."""
        inner = ChargeCardTool()
        child = Agent(id="child", name="specialist", llm=test_llm, role="work", tools=[inner])
        wrapper = SubAgentTool(agent=child, name="specialist", description="Delegates.")
        parent = build_agent(test_llm, wrapper)
        config = RunnableConfig(mock=RunMockConfig(policy=MockPolicy.ALL, exclude_ids={inner.id}))

        cloned, cloned_config = parent._clone_tool_for_execution(child, config)

        cloned_inner = cloned.tools[0]
        assert cloned_inner.id != inner.id
        assert cloned_inner.resolve_mock(cloned_config) is None, "the nested exclusion must be carried over"

    def test_mocks_hold_when_the_agent_runs_tools_in_parallel(self, test_llm):
        """Parallel batches clone tools, so the mock has to survive the clone."""
        tool = ChargeCardTool(mock=MockConfig(enabled=True, locked=True, output="not charged"))
        agent = build_agent(test_llm, tool)
        config = RunnableConfig()

        cloned, cloned_config = agent._clone_tool_for_execution(tool, config)
        result = cloned.run(input_data={"amount": 42}, config=cloned_config)

        assert result.status == RunnableStatus.SUCCESS
        assert CHARGES == []
        assert result.output["content"] == "[MOCKED] not charged"

    def test_a_locked_tool_stays_inert_in_the_control_run(self, test_llm):
        agent = build_agent(test_llm, ChargeCardTool(mock=MockConfig(enabled=True, locked=True)))

        with drive_agent(agent, []):
            result = agent.run(
                input_data={"input": "charge the customer 42"},
                config=RunnableConfig(mock=RunMockConfig(policy=MockPolicy.NONE)),
            )

        assert result.status == RunnableStatus.SUCCESS
        assert CHARGES == []
