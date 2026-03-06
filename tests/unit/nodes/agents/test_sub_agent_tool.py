import re

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.agent_tool import SubAgentTool, SubAgentToolInputSchema
from dynamiq.nodes.agents.components import schema_generator
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode


def _sanitize_tool_name(s: str) -> str:
    s = s.replace(" ", "-")
    return re.sub(r"[^a-zA-Z0-9_-]", "", s)


@pytest.fixture
def test_llm():
    connection = OpenAIConnection(api_key="test-api-key")
    return OpenAI(
        connection=connection,
        model="gpt-4o",
        max_tokens=100,
        temperature=0,
    )


@pytest.fixture
def child_agent(test_llm):
    return Agent(
        name="Researcher",
        id="researcher_agent",
        description="Performs web research",
        llm=test_llm,
        role="You are a research agent.",
        tools=[],
        max_loops=3,
    )


@pytest.fixture
def child_agent_no_description(test_llm):
    return Agent(
        name="Writer",
        id="writer_agent",
        llm=test_llm,
        role="You are a writing agent.",
        tools=[],
        max_loops=3,
    )


# --- SubAgentTool creation ---


class TestSubAgentToolCreation:
    def test_initialized_mode(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="Performs web research")

        assert tool.agent is child_agent
        assert tool.agent_factory is None
        assert tool.is_factory_mode is False
        assert tool.name == "Researcher"
        assert tool.description == "Performs web research"

    def test_factory_mode(self, test_llm):
        factory = lambda: Agent(name="Researcher", llm=test_llm, role="research", tools=[])
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory=factory,
        )

        assert tool.agent is None
        assert tool.agent_factory is factory
        assert tool.is_factory_mode is True

    def test_dict_factory_mode(self, test_llm):
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={
                "name": "Researcher",
                "llm": test_llm,
                "role": "You are a research agent.",
                "tools": [],
            },
        )

        assert tool.agent is None
        assert isinstance(tool.agent_factory, dict)
        assert tool.is_factory_mode is True
        assert tool.is_parallel_execution_allowed is True

        agent = tool.get_or_create_agent()
        assert isinstance(agent, Agent)
        assert agent.name == "Researcher"

        agent2 = tool.get_or_create_agent()
        assert agent2 is not agent

    def test_dict_factory_with_tools(self, test_llm):
        python_tool = Python(code="def run(input_data): return input_data")
        researcher_tool = SubAgentTool(
            name="Researcher",
            description="Searches the web",
            agent_factory={
                "name": "Researcher",
                "llm": test_llm,
                "role": "You are a research agent.",
                "tools": [python_tool],
            },
        )

        programmer_tool = SubAgentTool(
            name="Programmer",
            description="Writes code",
            agent_factory={
                "name": "Programmer",
                "llm": test_llm,
                "role": "You are a programming agent.",
                "tools": [],
            },
        )

        researcher = researcher_tool.get_or_create_agent()
        programmer = programmer_tool.get_or_create_agent()

        assert researcher.name == "Researcher"
        assert programmer.name == "Programmer"
        assert len(researcher.tools) == 1
        assert len(programmer.tools) == 0
        assert researcher.role == "You are a research agent."
        assert programmer.role == "You are a programming agent."

    def test_requires_agent_or_factory(self):
        with pytest.raises(ValueError, match="requires either"):
            SubAgentTool(name="Test", description="Test")

    def test_rejects_both_agent_and_factory(self, child_agent, test_llm):
        with pytest.raises(ValueError, match="cannot have both"):
            SubAgentTool(
                name="Test",
                description="Test",
                agent=child_agent,
                agent_factory=lambda: Agent(name="X", llm=test_llm, role="x", tools=[]),
            )

    def test_name_is_required(self, child_agent):
        with pytest.raises(Exception):
            SubAgentTool(agent=child_agent, description="desc")

    def test_description_is_required(self, child_agent):
        with pytest.raises(Exception):
            SubAgentTool(agent=child_agent, name="name")


# --- Parallel execution flag ---


class TestParallelExecutionFlag:
    def test_initialized_mode_not_parallel(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="research")
        assert tool.is_parallel_execution_allowed is False

    def test_factory_mode_is_parallel(self, test_llm):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory=lambda: Agent(name="Researcher", llm=test_llm, role="r", tools=[]),
        )
        assert tool.is_parallel_execution_allowed is True


# --- get_or_create_agent ---


class TestGetOrCreateAgent:
    def test_initialized_returns_same_instance(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="research")

        agent1 = tool.get_or_create_agent()
        agent2 = tool.get_or_create_agent()

        assert agent1 is child_agent
        assert agent2 is child_agent
        assert agent1 is agent2

    def test_init_components_rejects_bad_factory(self):
        tool = SubAgentTool(
            name="Bad",
            description="returns a string",
            agent_factory=lambda: "not an agent",
        )
        with pytest.raises(TypeError, match="agent_factory must return an Agent"):
            tool.init_components()

    def test_factory_returns_new_instance_each_time(self, test_llm):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory=lambda: Agent(name="Researcher", llm=test_llm, role="r", tools=[]),
        )

        agent1 = tool.get_or_create_agent()
        agent2 = tool.get_or_create_agent()

        assert isinstance(agent1, Agent)
        assert isinstance(agent2, Agent)
        assert agent1 is not agent2


# --- Auto-wrapping in Agent.__init__ ---


class TestAutoWrapping:
    def test_agent_in_tools_gets_wrapped(self, test_llm, child_agent):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent],
            max_loops=3,
        )

        agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1

        wrapper = agent_tools[0]
        assert wrapper.name == "Researcher"
        assert wrapper.description == "Performs web research"
        assert wrapper.agent is child_agent

    def test_agent_no_description_gets_empty_string(self, test_llm, child_agent_no_description):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent_no_description],
            max_loops=3,
        )

        agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        wrapper = agent_tools[0]
        assert wrapper.description == ""

    def test_non_agent_tools_not_wrapped(self, test_llm):
        python_tool = Python(code="def run(input_data): return input_data")
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[python_tool],
            max_loops=3,
        )

        sub_agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(sub_agent_tools) == 0

    def test_explicit_sub_agent_tool_not_double_wrapped(self, test_llm, child_agent):
        explicit_tool = SubAgentTool(agent=child_agent, name="Researcher", description="research")
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[explicit_tool],
            max_loops=3,
        )

        agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1
        assert agent_tools[0] is explicit_tool

    def test_mixed_tools(self, test_llm, child_agent):
        python_tool = Python(code="def run(input_data): return input_data")
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent, python_tool],
            max_loops=3,
        )

        sub_agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(sub_agent_tools) == 1
        assert sub_agent_tools[0].name == "Researcher"


# --- Tool lookup by name ---


class TestToolLookup:
    def test_tool_by_names_includes_wrapped_agent(self, test_llm, child_agent):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent],
            max_loops=3,
        )

        tool_map = parent.tool_by_names
        assert "Researcher" in tool_map
        assert isinstance(tool_map["Researcher"], SubAgentTool)


# --- to_dict serialization ---


class TestSerialization:
    def test_to_dict_delegates_to_agent(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="research")
        result = tool.to_dict()

        assert result["name"] == "Researcher"
        assert result["type"] == child_agent.type

    def test_to_dict_factory_mode_uses_default(self, test_llm):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory=lambda: Agent(name="Researcher", llm=test_llm, role="r", tools=[]),
        )
        result = tool.to_dict()

        assert result["name"] == "Researcher"
        assert "SubAgentTool" in result.get("type", "")


# --- Schema generation ---


class TestSchemaGeneration:
    def test_function_calling_schema_detects_sub_agent_tool(self, test_llm, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="Performs web research")
        schemas = schema_generator.generate_function_calling_schemas(
            tools=[tool],
            delegation_allowed=True,
            sanitize_tool_name=_sanitize_tool_name,
            llm=test_llm,
        )

        tool_schema = next(s for s in schemas if s["function"]["name"] == "Researcher")
        assert "delegate_final" in tool_schema["function"]["parameters"]["properties"]["action_input"]["description"]

    def test_structured_output_schema_detects_sub_agent_tool(self, test_llm, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="Performs web research")
        schema = schema_generator.generate_structured_output_schemas(
            tools=[tool],
            sanitize_tool_name=_sanitize_tool_name,
            delegation_allowed=True,
        )

        action_input_desc = schema["json_schema"]["schema"]["properties"]["action_input"]["description"]
        assert "delegate_final" in action_input_desc


# --- _should_delegate_final ---


class TestDelegation:
    def test_delegate_final_with_sub_agent_tool(self, test_llm, child_agent):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent],
            delegation_allowed=True,
            inference_mode=InferenceMode.XML,
            max_loops=3,
        )

        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))
        result = parent._should_delegate_final(wrapper, {"input": "task", "delegate_final": True})
        assert result is True

    def test_delegate_final_without_flag(self, test_llm, child_agent):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent],
            delegation_allowed=True,
            inference_mode=InferenceMode.XML,
            max_loops=3,
        )

        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))
        result = parent._should_delegate_final(wrapper, {"input": "task"})
        assert result is False

    def test_delegate_final_delegation_not_allowed(self, test_llm, child_agent):
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[child_agent],
            delegation_allowed=False,
            inference_mode=InferenceMode.XML,
            max_loops=3,
        )

        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))
        result = parent._should_delegate_final(wrapper, {"input": "task", "delegate_final": True})
        assert result is False

    def test_delegate_final_with_explicit_sub_agent_tool(self, test_llm, child_agent):
        explicit_tool = SubAgentTool(agent=child_agent, name="Researcher", description="Performs web research")
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[explicit_tool],
            delegation_allowed=True,
            inference_mode=InferenceMode.XML,
            max_loops=3,
        )

        tool = parent.tools[0]
        assert isinstance(tool, SubAgentTool)
        result = parent._should_delegate_final(tool, {"input": "task", "delegate_final": True})
        assert result is True

    def test_delegate_final_non_agent_tool(self, test_llm):
        python_tool = Python(code="def run(input_data): return input_data")
        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="You are a manager.",
            tools=[python_tool],
            delegation_allowed=True,
            inference_mode=InferenceMode.XML,
            max_loops=3,
        )

        result = parent._should_delegate_final(python_tool, {"input": "task", "delegate_final": True})
        assert result is False
