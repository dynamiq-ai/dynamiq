import re
from unittest.mock import MagicMock, patch

import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.agents.components import schema_generator
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.sandboxes.e2b import E2BSandbox


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


class TestSubAgentToolCreation:
    def test_initialized_mode(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Researcher", description="Performs web research")

        assert tool.agent is child_agent
        assert tool.agent_factory is None
        assert tool.is_factory_mode is False
        assert tool.name == "Researcher"
        assert tool.description.startswith("Performs web research")
        assert SubAgentTool.INITIALIZED_HINT in tool.description

    def test_factory_mode(self, test_llm):
        def factory():
            return Agent(name="Researcher", llm=test_llm, role="research", tools=[])

        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory=factory,
        )

        assert tool.agent is None
        assert tool.agent_factory is factory
        assert tool.is_factory_mode is True

    def test_dict_factory_mode(self):
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={
                "connections": {
                    "openai-conn": {
                        "type": "dynamiq.connections.OpenAI",
                        "api_key": "test-key",
                    },
                },
                "name": "Researcher",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "connection": "openai-conn",
                    "model": "gpt-4o",
                },
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

    def test_validate_factory_false_skips_trial(self):
        tool = SubAgentTool(
            name="Bad",
            description="returns a string",
            agent_factory=lambda: "not an agent",
            validate_factory=False,
        )
        tool.init_components()

        with pytest.raises(TypeError, match="agent_factory must return an Agent"):
            tool.get_or_create_agent()

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

    def test_factory_always_generates_unique_id(self):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory={
                "connections": {
                    "openai-conn": {
                        "type": "dynamiq.connections.OpenAI",
                        "api_key": "test-key",
                    },
                },
                "name": "Researcher",
                "id": "fixed-id",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "connection": "openai-conn",
                    "model": "gpt-4o",
                },
                "role": "r",
                "tools": [],
            },
        )

        agents = [tool.get_or_create_agent() for _ in range(5)]
        ids = [a.id for a in agents]

        assert all(aid != "fixed-id" for aid in ids), "Factory agents must not keep the explicit ID"
        assert len(set(ids)) == 5, "Every factory agent must have a unique ID"


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
        assert wrapper.description.startswith("Performs web research")
        assert SubAgentTool.INITIALIZED_HINT in wrapper.description
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
        assert SubAgentTool.INITIALIZED_HINT in wrapper.description

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


class TestSerialization:
    def test_to_dict_preserves_wrapper_and_nests_agent(self, child_agent):
        tool = SubAgentTool(agent=child_agent, name="Research Tool", description="Delegates to researcher")
        result = tool.to_dict()

        assert "SubAgentTool" in result["type"]
        assert result["name"] == "Research Tool"
        assert "Delegates to researcher" in result["description"]
        assert SubAgentTool.INITIALIZED_HINT in result["description"]
        assert "agent" in result
        assert result["agent"]["name"] == "Researcher"
        assert result["agent"]["type"] == child_agent.type

    def test_to_dict_callable_factory_placeholder(self, test_llm):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory=lambda: Agent(name="Researcher", llm=test_llm, role="r", tools=[]),
        )

        result = tool.to_dict()
        assert "agent_factory" in result
        assert result["agent_factory"]["_type"] == "callable"
        assert "_repr" in result["agent_factory"]

    def test_to_dict_dict_factory_serializes(self):
        tool = SubAgentTool(
            name="Researcher",
            description="research",
            agent_factory={
                "connections": {
                    "openai-conn": {
                        "type": "dynamiq.connections.OpenAI",
                        "api_key": "test-key",
                    },
                },
                "name": "Researcher",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "connection": "openai-conn",
                    "model": "gpt-4o",
                },
                "role": "You are a research agent.",
                "tools": [],
            },
        )
        result = tool.to_dict()

        assert result["name"] == "Researcher"
        assert "SubAgentTool" in result.get("type", "")
        assert "agent" not in result
        assert "agent_factory" in result
        assert isinstance(result["agent_factory"], dict)
        assert result["agent_factory"]["name"] == "Researcher"


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


class TestYamlRoundtrip:
    def test_sub_agent_tool_yaml_roundtrip(self, tmp_path):
        """Roundtrip: to_yaml_file -> from_yaml_file -> to_yaml_file -> from_yaml_file.

        Ensures an explicit SubAgentTool wrapping a child Agent survives
        serialization and is not duplicated after reload.
        """
        openai_conn = OpenAIConnection(id="openai-conn", api_key="test-key")
        parent_llm = OpenAI(id="parent-llm", connection=openai_conn, model="gpt-4o")
        child_llm = OpenAI(id="child-llm", connection=openai_conn, model="gpt-4o")

        child_agent = Agent(
            id="researcher-agent",
            name="Researcher Agent",
            description="I am a research agent",
            llm=child_llm,
            role="You are a research agent.",
            tools=[],
            max_loops=3,
        )
        sub_agent_tool = SubAgentTool(
            agent=child_agent,
            name="Research Tool",
            description="Delegates to researcher",
        )

        parent = Agent(
            id="manager-agent",
            name="Manager",
            llm=parent_llm,
            role="You are a manager.",
            tools=[sub_agent_tool],
            max_loops=5,
        )

        wf = Workflow(
            id="sub-agent-wf",
            flow=Flow(id="sub-agent-flow", name="SubAgent Flow", nodes=[parent]),
            version="1",
        )

        yaml_path = tmp_path / "sub_agent_tool.yaml"
        wf.to_yaml_file(yaml_path)

        loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
        assert len(loaded.flow.nodes) == 1
        loaded_agent = loaded.flow.nodes[0]
        assert isinstance(loaded_agent, Agent)

        agent_tools = [t for t in loaded_agent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1, "First load should have exactly one SubAgentTool"
        wrapper = agent_tools[0]
        assert wrapper.name == "Research Tool"
        assert "Delegates to researcher" in wrapper.description
        assert SubAgentTool.INITIALIZED_HINT in wrapper.description
        assert wrapper.agent.name == "Researcher Agent"
        assert wrapper.agent.description == "I am a research agent"

        rt_path = tmp_path / "sub_agent_tool_rt.yaml"
        loaded.to_yaml_file(rt_path)

        rt = Workflow.from_yaml_file(str(rt_path), init_components=True)
        rt_agent = rt.flow.nodes[0]
        assert isinstance(rt_agent, Agent)

        rt_tools = [t for t in rt_agent.tools if isinstance(t, SubAgentTool)]
        assert len(rt_tools) == 1, "After roundtrip, SubAgentTool must not be duplicated"
        rt_wrapper = rt_tools[0]
        assert rt_wrapper.name == "Research Tool"
        assert "Delegates to researcher" in rt_wrapper.description
        assert SubAgentTool.INITIALIZED_HINT in rt_wrapper.description
        assert rt_wrapper.agent.name == "Researcher Agent"
        assert rt_wrapper.agent.description == "I am a research agent"

    def test_agent_as_tool_yaml_roundtrip_backward_compat(self, tmp_path):
        """Roundtrip with a raw Agent passed as a tool (backward-compatible auto-wrap).

        The parent Agent.__init__ auto-wraps child Agents into SubAgentTool
        using the agent's own name/description. After YAML serialization the
        SubAgentTool is preserved, so the wrapper keeps the agent's values.
        """
        openai_conn = OpenAIConnection(id="openai-conn", api_key="test-key")
        parent_llm = OpenAI(id="parent-llm", connection=openai_conn, model="gpt-4o")
        child_llm = OpenAI(id="child-llm", connection=openai_conn, model="gpt-4o")

        child_agent = Agent(
            id="researcher-agent",
            name="Researcher",
            description="Performs web research",
            llm=child_llm,
            role="You are a research agent.",
            tools=[],
            max_loops=3,
        )

        parent = Agent(
            id="manager-agent",
            name="Manager",
            llm=parent_llm,
            role="You are a manager.",
            tools=[child_agent],
            max_loops=5,
        )

        wf = Workflow(
            id="compat-wf",
            flow=Flow(id="compat-flow", name="Compat Flow", nodes=[parent]),
            version="1",
        )

        yaml_path = tmp_path / "agent_as_tool.yaml"
        wf.to_yaml_file(yaml_path)

        loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
        assert len(loaded.flow.nodes) == 1
        loaded_agent = loaded.flow.nodes[0]
        assert isinstance(loaded_agent, Agent)

        agent_tools = [t for t in loaded_agent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1, "First load should have exactly one SubAgentTool"
        wrapper = agent_tools[0]
        assert wrapper.name == "Researcher"
        assert "Performs web research" in wrapper.description
        assert SubAgentTool.INITIALIZED_HINT in wrapper.description
        assert wrapper.agent.name == "Researcher"
        assert wrapper.agent.description == "Performs web research"

        rt_path = tmp_path / "agent_as_tool_rt.yaml"
        loaded.to_yaml_file(rt_path)

        rt = Workflow.from_yaml_file(str(rt_path), init_components=True)
        rt_agent = rt.flow.nodes[0]
        assert isinstance(rt_agent, Agent)

        rt_tools = [t for t in rt_agent.tools if isinstance(t, SubAgentTool)]
        assert len(rt_tools) == 1, "After roundtrip, SubAgentTool must not be duplicated"
        rt_wrapper = rt_tools[0]
        assert rt_wrapper.name == "Researcher"
        assert "Performs web research" in rt_wrapper.description
        assert SubAgentTool.INITIALIZED_HINT in rt_wrapper.description
        assert rt_wrapper.agent.name == "Researcher"
        assert rt_wrapper.agent.description == "Performs web research"

    def test_dict_factory_yaml_roundtrip(self, tmp_path):
        """Roundtrip for SubAgentTool in YAML-string factory mode.

        Verifies that to_yaml -> from_yaml preserves factory semantics:
        the reloaded tool should still be in factory mode (agent is None,
        agent_factory is a YAML string) and get_or_create_agent() should produce
        distinct Agent instances with isolated sandbox and tools.
        """
        openai_conn = OpenAIConnection(id="openai-conn", api_key="test-key")
        parent_llm = OpenAI(id="parent-llm", connection=openai_conn, model="gpt-4o")

        sub_agent_tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={
                "connections": {
                    "openai-conn": {
                        "type": "dynamiq.connections.OpenAI",
                        "api_key": "test-key",
                    },
                    "e2b-conn": {
                        "type": "dynamiq.connections.E2B",
                        "api_key": "test-key",
                    },
                },
                "name": "Researcher",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "connection": "openai-conn",
                    "model": "gpt-4o",
                },
                "role": "You are a research agent.",
                "tools": [
                    {
                        "type": "dynamiq.nodes.tools.python.Python",
                        "code": "def run(input_data): return input_data",
                    },
                ],
                "max_loops": 3,
                "sandbox": {
                    "enabled": True,
                    "backend": {
                        "type": "dynamiq.sandboxes.e2b.E2BSandbox",
                        "connection": "e2b-conn",
                    },
                },
            },
        )

        parent = Agent(
            id="manager-agent",
            name="Manager",
            llm=parent_llm,
            role="You are a manager.",
            tools=[sub_agent_tool],
            max_loops=5,
        )

        wf = Workflow(
            id="dict-factory-wf",
            flow=Flow(id="dict-factory-flow", name="Dict Factory Flow", nodes=[parent]),
            version="1",
        )

        yaml_path = tmp_path / "dict_factory.yaml"
        wf.to_yaml_file(yaml_path)

        loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
        assert len(loaded.flow.nodes) == 1
        loaded_parent = loaded.flow.nodes[0]
        assert isinstance(loaded_parent, Agent)

        agent_tools = [t for t in loaded_parent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1, "Should have exactly one SubAgentTool"
        wrapper = agent_tools[0]
        assert wrapper.name == "Researcher"
        assert "Performs web research" in wrapper.description
        assert SubAgentTool.FACTORY_HINT in wrapper.description
        assert wrapper.is_factory_mode is True
        assert wrapper.agent is None
        assert isinstance(wrapper.agent_factory, dict)

        agent_a = wrapper.get_or_create_agent()
        agent_b = wrapper.get_or_create_agent()
        assert isinstance(agent_a, Agent)
        assert isinstance(agent_b, Agent)
        assert agent_a is not agent_b, "Factory must produce distinct instances"
        assert agent_a.name == "Researcher"
        assert agent_b.name == "Researcher"
        assert agent_a.role == "You are a research agent."
        assert len(agent_a.tools) >= 1

        assert agent_a.sandbox is not None, "Factory-created agent must have sandbox"
        assert agent_a.sandbox.enabled is True
        assert isinstance(agent_a.sandbox.backend, E2BSandbox)
        assert agent_b.sandbox is not None
        assert agent_a.sandbox is not agent_b.sandbox, "Each factory agent must get its own sandbox"
        assert agent_a.sandbox.backend is not agent_b.sandbox.backend, "Each factory agent must get its own backend"

        a_py_tools = [t for t in agent_a.tools if isinstance(t, Python)]
        b_py_tools = [t for t in agent_b.tools if isinstance(t, Python)]
        assert len(a_py_tools) >= 1 and len(b_py_tools) >= 1
        assert a_py_tools[0] is not b_py_tools[0], "Each factory agent must get its own tool instances"

        # Second roundtrip
        rt_path = tmp_path / "dict_factory_rt.yaml"
        loaded.to_yaml_file(rt_path)

        rt = Workflow.from_yaml_file(str(rt_path), init_components=True)
        rt_parent = rt.flow.nodes[0]
        rt_tools = [t for t in rt_parent.tools if isinstance(t, SubAgentTool)]
        assert len(rt_tools) == 1, "After roundtrip, SubAgentTool must not be duplicated"
        rt_wrapper = rt_tools[0]
        assert rt_wrapper.is_factory_mode is True
        assert rt_wrapper.agent is None
        assert isinstance(rt_wrapper.agent_factory, dict)

        rt_agent = rt_wrapper.get_or_create_agent()
        assert isinstance(rt_agent, Agent)
        assert rt_agent.name == "Researcher"

        assert rt_agent.sandbox is not None, "Roundtrip factory-created agent must have sandbox"
        assert rt_agent.sandbox.enabled is True
        assert isinstance(rt_agent.sandbox.backend, E2BSandbox)

    def test_example_yaml_files_load(self, tmp_path):
        """Verify example YAML files load and factory roundtrips correctly."""
        import os

        examples_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "..",
            "examples",
            "components",
            "core",
            "dag",
        )

        # --- agents_as_tools.yaml (initialized agent mode) ---
        agents_path = os.path.join(examples_dir, "agents_as_tools.yaml")
        if not os.path.exists(agents_path):
            pytest.skip("Example YAML not found")

        loaded = Workflow.from_yaml_file(agents_path, init_components=False)
        assert len(loaded.flow.nodes) == 1
        parent = loaded.flow.nodes[0]
        assert isinstance(parent, Agent)
        agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1
        assert agent_tools[0].agent.name == "Coder Agent"

        # --- subagent_factory.yaml (YAML-string factory mode + roundtrip) ---
        factory_path = os.path.join(examples_dir, "subagent_factory.yaml")
        if not os.path.exists(factory_path):
            pytest.skip("Example YAML not found")

        loaded = Workflow.from_yaml_file(factory_path, init_components=True)
        assert len(loaded.flow.nodes) == 1
        parent = loaded.flow.nodes[0]
        assert isinstance(parent, Agent)

        agent_tools = [t for t in parent.tools if isinstance(t, SubAgentTool)]
        assert len(agent_tools) == 1
        wrapper = agent_tools[0]
        assert wrapper.name == "Researcher"
        assert wrapper.is_factory_mode is True
        assert isinstance(wrapper.agent_factory, dict)

        agent_a = wrapper.get_or_create_agent()
        agent_b = wrapper.get_or_create_agent()
        assert isinstance(agent_a, Agent)
        assert agent_a is not agent_b, "Factory must produce distinct instances"
        assert agent_a.name == "Researcher"
        assert agent_a.sandbox is not None
        assert isinstance(agent_a.sandbox.backend, E2BSandbox)
        assert agent_a.sandbox.backend is not agent_b.sandbox.backend

        rt_path = tmp_path / "subagent_factory_rt.yaml"
        loaded.to_yaml_file(rt_path)

        rt = Workflow.from_yaml_file(str(rt_path), init_components=True)
        rt_wrapper = [t for t in rt.flow.nodes[0].tools if isinstance(t, SubAgentTool)][0]
        assert rt_wrapper.is_factory_mode is True
        assert isinstance(rt_wrapper.agent_factory, dict)

        rt_agent = rt_wrapper.get_or_create_agent()
        assert isinstance(rt_agent, Agent)
        assert rt_agent.name == "Researcher"
        assert rt_agent.sandbox is not None
        assert isinstance(rt_agent.sandbox.backend, E2BSandbox)


# --- ToolParams resolution through SubAgentTool wrapper ---


def _make_successful_run_result():
    """Return a mock RunnableResult that looks like a successful tool run."""
    result = MagicMock(spec=RunnableResult)
    result.status = RunnableStatus.SUCCESS
    result.output = {"content": "ok"}
    return result


class TestToolParamsResolution:
    def test_flat_params_applied_by_agent_name_and_id(self, test_llm):
        """Flat params keyed by the original agent name/ID are applied to merged_input."""
        child = Agent(
            name="Coder",
            id="coder-42",
            llm=test_llm,
            role="code",
            tools=[],
            max_loops=3,
        )
        parent = Agent(
            name="Boss",
            llm=test_llm,
            role="manage",
            tools=[child],
            max_loops=3,
        )
        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))
        assert wrapper.id != "coder-42", "SubAgentTool must have its own auto-generated ID"

        tp = ToolParams(
            by_name={"Coder": {"name_param": "from_name"}},
            by_id={"coder-42": {"id_param": "from_id"}},
        )

        with patch.object(type(child), "run", return_value=_make_successful_run_result()) as mock_run:
            parent._run_tool(tool=wrapper, tool_input={"input": "x"}, config=None, tool_params=tp)
            kwargs = mock_run.call_args[1]

        data = kwargs["input_data"]
        assert data["name_param"] == "from_name", "by_name via resolved_agent.name"
        assert data["id_param"] == "from_id", "by_id via resolved_agent.id"

    def test_nested_tool_params_propagated_to_child(self, test_llm):
        """Nested ToolParams keyed by wrapper ID are propagated into child_kwargs."""
        child = Agent(
            name="Coder",
            id="coder-42",
            llm=test_llm,
            role="code",
            tools=[],
            max_loops=3,
        )
        parent = Agent(
            name="Boss",
            llm=test_llm,
            role="manage",
            tools=[child],
            max_loops=3,
        )
        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))

        nested_tp = ToolParams(
            **{
                "global": {"nested_global": "yes"},
            }
        )
        tp = ToolParams(
            by_id={"coder-42": nested_tp},
        )

        with patch.object(type(child), "run", return_value=_make_successful_run_result()) as mock_run:
            parent._run_tool(tool=wrapper, tool_input={"input": "x"}, config=None, tool_params=tp)
            kwargs = mock_run.call_args[1]

        passed_tp = kwargs.get("tool_params")
        assert passed_tp is not None, "Nested ToolParams should be propagated"
        assert passed_tp.global_params.get("nested_global") == "yes"


class TestSubAgentStreaming:
    def test_streaming_propagated_and_delivered(self, test_llm, child_agent):
        """Callbacks propagate to the child and stream events arrive at the parent's handler."""
        from dynamiq.callbacks import BaseCallbackHandler
        from dynamiq.runnables import RunnableConfig
        from dynamiq.types.streaming import StreamingConfig

        child_agent.streaming = StreamingConfig(enabled=True)

        parent = Agent(
            name="Manager",
            llm=test_llm,
            role="manage",
            tools=[child_agent],
            max_loops=3,
        )
        wrapper = next(t for t in parent.tools if isinstance(t, SubAgentTool))

        stream_chunks = []
        forwarded_config = {}

        class CapturingCallback(BaseCallbackHandler):
            def on_node_execute_stream(self, node_data, chunk, **kwargs):
                stream_chunks.append(chunk)

        capturing = CapturingCallback()
        config = RunnableConfig(callbacks=[capturing])

        def fake_run(input_data, config, **kwargs):
            forwarded_config["value"] = config
            child_agent.stream_content(
                content="sub-agent answer",
                source=child_agent.name,
                step="answer",
                config=config,
            )
            return _make_successful_run_result()

        with patch.object(type(child_agent), "run", side_effect=fake_run):
            parent._run_tool(tool=wrapper, tool_input={"input": "query"}, config=config)

        assert (
            capturing in forwarded_config["value"].callbacks
        ), "Parent callback must appear in the config forwarded to the sub-agent"
        assert len(stream_chunks) >= 1, "At least one stream chunk from the child must reach the callback"
        delta = stream_chunks[0]["choices"][0]["delta"]
        assert delta["content"] == "sub-agent answer"
        assert delta["source"] == child_agent.name
        assert delta["step"] == "answer"
