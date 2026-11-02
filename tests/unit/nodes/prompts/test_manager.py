import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.prompts import get_prompt_constant
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager
from dynamiq.nodes.agents.prompts.orchestrators.graph import (
    PROMPT_TEMPLATE_GRAPH_ASSIGN,
    PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT,
    PROMPT_TEMPLATE_GRAPH_PLAN,
)
from dynamiq.nodes.agents.prompts.overrides.gpt import (
    REACT_BLOCK_INSTRUCTIONS_SINGLE as GPT_REACT_BLOCK_INSTRUCTIONS_SINGLE,
)
from dynamiq.nodes.agents.prompts.overrides.gpt import (
    REACT_BLOCK_XML_INSTRUCTIONS_SINGLE as GPT_REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
)
from dynamiq.nodes.agents.prompts.secondary_instructions import DELEGATION_INSTRUCTIONS, DELEGATION_INSTRUCTIONS_XML
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode


@pytest.fixture
def test_llm():
    """Provides a reusable LLM instance for testing."""
    connection = OpenAIConnection(api_key="api-key")

    return OpenAI(
        connection=connection,
        model="gpt-5.1",
        max_tokens=100,
        temperature=0,
    )


def test_variables_refreshed_on_reset():
    """Test that prompt variables are refreshed on reset."""
    manager = AgentPromptManager()
    manager._prompt_variables["block1"] = "Content1"

    manager.reset()

    assert "block1" not in manager._prompt_variables


@pytest.mark.parametrize(
    "inference_mode,expected_delegation_text",
    [
        (InferenceMode.DEFAULT, DELEGATION_INSTRUCTIONS),
        (InferenceMode.XML, DELEGATION_INSTRUCTIONS_XML),
        (InferenceMode.FUNCTION_CALLING, DELEGATION_INSTRUCTIONS),
        (InferenceMode.STRUCTURED_OUTPUT, DELEGATION_INSTRUCTIONS),
    ],
)
def test_agent_initialization_with_model_specific_prompts(test_llm, inference_mode, expected_delegation_text):
    """Test that Agent properly initializes with GPT model-specific prompts for all inference modes."""
    python_tool = Python(code="def run(input_data): return input_data")
    agent = Agent(
        name="TestAgent",
        id="test_agent",
        llm=test_llm,
        role="You are a helpful assistant with special capabilities.",
        tools=[python_tool],
        inference_mode=inference_mode,
        verbose=False,
        delegation_allowed=True,
    )

    assert agent.system_prompt_manager is not None
    assert isinstance(agent.system_prompt_manager, AgentPromptManager)

    # gpt-5.1 from test_llm fixture
    assert agent.system_prompt_manager.model_name == "openai/gpt-5.1"

    assert "role" in agent.system_prompt_manager._prompt_blocks
    assert "helpful assistant" in agent.system_prompt_manager._prompt_blocks["role"].lower()

    prompt = agent.system_prompt_manager.generate_prompt()
    assert prompt is not None

    if expected_delegation_text:
        assert expected_delegation_text in prompt, (
            f"Expected delegation instructions not found in {inference_mode} mode for GPT model. "
            f"Looking for: {expected_delegation_text[:100]}..."
        )

    if inference_mode == InferenceMode.XML:
        assert GPT_REACT_BLOCK_XML_INSTRUCTIONS_SINGLE == get_prompt_constant(
            "gpt-5.1", "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE", None
        )
    if inference_mode == InferenceMode.DEFAULT:
        assert GPT_REACT_BLOCK_INSTRUCTIONS_SINGLE == get_prompt_constant(
            "gpt-5.1", "REACT_BLOCK_INSTRUCTIONS_SINGLE", None
        )


def test_graph_manager_prompt_blocks_setup(test_llm):
    """Test that GraphManager properly sets up orchestrator-specific prompt blocks."""

    graph_manager = GraphAgentManager(
        name="TestGraphManager",
        id="test_graph_manager",
        llm=test_llm,
    )

    assert graph_manager.system_prompt_manager is not None
    assert isinstance(graph_manager.system_prompt_manager, AgentPromptManager)

    assert graph_manager.system_prompt_manager._prompt_blocks
    assert graph_manager.system_prompt_manager._prompt_blocks["plan"] == PROMPT_TEMPLATE_GRAPH_PLAN
    assert graph_manager.system_prompt_manager._prompt_blocks["assign"] == PROMPT_TEMPLATE_GRAPH_ASSIGN
    assert graph_manager.system_prompt_manager._prompt_blocks["handle_input"] == PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT


def test_file_tools_persist_across_resets(test_llm):
    """Test that dynamically added file tools persist in tool_description after reset.

    This is a regression test for a bug where:
    1. File tools are added to self.tools (permanent)
    2. tool_description variable is updated
    3. reset() restores variables from _initial_variables
    4. tool_description loses the file tools
    5. Subsequent runs have outdated tool description
    """
    from io import BytesIO

    from dynamiq.runnables import RunnableConfig

    agent = Agent(
        name="TestAgent",
        id="test_agent",
        llm=test_llm,
        role="You are a helpful assistant.",
        tools=[],
    )

    initial_tool_count = len(agent.tools)
    initial_tool_description = agent.tool_description
    assert "file-read" not in initial_tool_description
    assert "file-search" not in initial_tool_description

    dummy_file = BytesIO(b"test content")
    dummy_file.name = "test.txt"
    dummy_file.seek(0)

    # First run with files - triggers file tool addition
    input_data_with_files = {
        "input": "Analyze this file",
        "files": [dummy_file],
    }

    try:
        agent.run(input_data=input_data_with_files, config=RunnableConfig())
    except Exception:
        # Execution may fail due to API calls, but tool addition happens before that
        pass

    assert len(agent.tools) > initial_tool_count
    updated_tool_description = agent.tool_description
    assert "file-read" in updated_tool_description or "file" in updated_tool_description.lower()

    assert agent.system_prompt_manager._initial_variables["tool_description"] == updated_tool_description

    # Simulates end of run
    agent.reset_run_state()

    after_reset_tool_description = agent.system_prompt_manager._prompt_variables["tool_description"]
    assert after_reset_tool_description == updated_tool_description
    assert "FileReadTool" in after_reset_tool_description or "file" in after_reset_tool_description.lower()

    # Second run without files - should still have correct tool description
    input_data_no_files = {
        "input": "Do something else",
    }

    try:
        agent.run(input_data=input_data_no_files, config=RunnableConfig())
    except Exception:
        pass

    final_tool_description = agent.system_prompt_manager._prompt_variables["tool_description"]
    assert final_tool_description == updated_tool_description
    assert "FileReadTool" in final_tool_description or "file" in final_tool_description.lower()


def test_agent_serialization_excludes_system_prompt_manager(test_llm):
    """Test that system_prompt_manager is excluded from serialization.

    system_prompt_manager is a runtime state container that should not be serialized.
    It should be excluded from to_dict() output to comply with the project rule:
    'Non-serializable field in to_dict(): Runtime objects must be excluded from to_dict()'.
    """
    agent = Agent(
        name="TestAgent",
        id="test_agent",
        llm=test_llm,
        role="You are a helpful assistant.",
        tools=[],
    )

    assert agent.system_prompt_manager is not None
    assert isinstance(agent.system_prompt_manager, AgentPromptManager)

    agent_dict = agent.to_dict()

    assert "system_prompt_manager" not in agent_dict

    assert "system_prompt_manager" in agent.to_dict_exclude_params


def test_instructions_field(test_llm):
    """Test that instructions render in prompt, coexist with other operational instructions, and handle Jinja syntax."""
    python_tool = Python(code="def run(input_data): return input_data")
    agent = Agent(
        name="TestAgent",
        llm=test_llm,
        instructions="Custom user instruction.",
        tools=[python_tool],
        delegation_allowed=True,
    )

    ops_block = agent.system_prompt_manager._prompt_blocks.get("operational_instructions", "")
    assert "Custom user instruction." in ops_block
    assert "delegate_final" in ops_block

    prompt = agent.system_prompt_manager.generate_prompt()
    assert "Custom user instruction." in prompt

    agent2 = Agent(name="TestAgent2", llm=test_llm, instructions="Use {{ user_name }}.", tools=[])
    prompt2 = agent2.system_prompt_manager.generate_prompt()
    assert "{{ user_name }}" in prompt2
