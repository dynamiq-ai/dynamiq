import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager
from dynamiq.nodes.agents.prompts.orchestrators.adaptive import (
    PROMPT_TEMPLATE_ADAPTIVE_FINAL,
    PROMPT_TEMPLATE_ADAPTIVE_PLAN,
    PROMPT_TEMPLATE_ADAPTIVE_REFLECT,
    PROMPT_TEMPLATE_ADAPTIVE_RESPOND,
)
from dynamiq.nodes.agents.prompts.orchestrators.base import PROMPT_TEMPLATE_BASE_HANDLE_INPUT
from dynamiq.nodes.agents.prompts.orchestrators.graph import (
    PROMPT_TEMPLATE_GRAPH_ASSIGN,
    PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT,
    PROMPT_TEMPLATE_GRAPH_PLAN,
)
from dynamiq.nodes.agents.prompts.orchestrators.linear import (
    PROMPT_TEMPLATE_LINEAR_ASSIGN,
    PROMPT_TEMPLATE_LINEAR_FINAL,
    PROMPT_TEMPLATE_LINEAR_PLAN,
)
from dynamiq.nodes.agents.prompts.secondary_instructions import DELEGATION_INSTRUCTIONS
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


def test_agent_initialization_with_model_specific_prompts(test_llm):
    """Test that Agent properly initializes with GPT model-specific prompts (function calling only)."""
    python_tool = Python(code="def run(input_data): return input_data")
    agent = Agent(
        name="TestAgent",
        id="test_agent",
        llm=test_llm,
        role="You are a helpful assistant with special capabilities.",
        tools=[python_tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        verbose=False,
        delegation_allowed=True,
    )

    # Verify prompt manager is initialized
    assert agent.system_prompt_manager is not None
    assert isinstance(agent.system_prompt_manager, AgentPromptManager)

    # Verify model name is set (gpt-5.1 from test_llm fixture)
    assert agent.system_prompt_manager.model_name == "openai/gpt-5.1"

    # Verify role block exists and contains the role text
    assert "role" in agent.system_prompt_manager._prompt_blocks
    assert "helpful assistant" in agent.system_prompt_manager._prompt_blocks["role"].lower()

    # Generate prompt and verify it's not empty
    prompt = agent.system_prompt_manager.generate_prompt()
    assert prompt is not None

    # Verify delegation instructions are in the prompt
    assert DELEGATION_INSTRUCTIONS in prompt


def test_linear_manager_prompt_blocks_setup(test_llm):
    """Test that LinearManager properly sets up orchestrator-specific prompt blocks."""

    # Create LinearManager
    linear_manager = LinearAgentManager(
        name="TestLinearManager",
        id="test_linear_manager",
        llm=test_llm,
    )

    # Verify prompt manager is initialized
    assert linear_manager.system_prompt_manager is not None
    assert isinstance(linear_manager.system_prompt_manager, AgentPromptManager)

    assert linear_manager.system_prompt_manager._prompt_blocks
    assert linear_manager.system_prompt_manager._prompt_blocks["plan"] == PROMPT_TEMPLATE_LINEAR_PLAN
    assert linear_manager.system_prompt_manager._prompt_blocks["assign"] == PROMPT_TEMPLATE_LINEAR_ASSIGN
    assert linear_manager.system_prompt_manager._prompt_blocks["final"] == PROMPT_TEMPLATE_LINEAR_FINAL
    assert linear_manager.system_prompt_manager._prompt_blocks["handle_input"] == PROMPT_TEMPLATE_BASE_HANDLE_INPUT


def test_graph_manager_prompt_blocks_setup(test_llm):
    """Test that GraphManager properly sets up orchestrator-specific prompt blocks."""

    # Create GraphManager
    graph_manager = GraphAgentManager(
        name="TestGraphManager",
        id="test_graph_manager",
        llm=test_llm,
    )

    # Verify prompt manager is initialized
    assert graph_manager.system_prompt_manager is not None
    assert isinstance(graph_manager.system_prompt_manager, AgentPromptManager)

    assert graph_manager.system_prompt_manager._prompt_blocks
    assert graph_manager.system_prompt_manager._prompt_blocks["plan"] == PROMPT_TEMPLATE_GRAPH_PLAN
    assert graph_manager.system_prompt_manager._prompt_blocks["assign"] == PROMPT_TEMPLATE_GRAPH_ASSIGN
    assert graph_manager.system_prompt_manager._prompt_blocks["handle_input"] == PROMPT_TEMPLATE_GRAPH_HANDLE_INPUT


def test_adaptive_manager_prompt_blocks_setup(test_llm):
    """Test that AdaptiveManager properly sets up orchestrator-specific prompt blocks."""

    # Create AdaptiveManager
    adaptive_manager = AdaptiveAgentManager(name="TestAdaptiveManager", id="test_adaptive_manager", llm=test_llm)

    assert adaptive_manager.system_prompt_manager is not None
    assert isinstance(adaptive_manager.system_prompt_manager, AgentPromptManager)

    assert adaptive_manager.system_prompt_manager._prompt_blocks
    assert adaptive_manager.system_prompt_manager._prompt_blocks["plan"] == PROMPT_TEMPLATE_ADAPTIVE_PLAN
    assert adaptive_manager.system_prompt_manager._prompt_blocks["final"] == PROMPT_TEMPLATE_ADAPTIVE_FINAL
    assert adaptive_manager.system_prompt_manager._prompt_blocks["respond"] == PROMPT_TEMPLATE_ADAPTIVE_RESPOND
    assert adaptive_manager.system_prompt_manager._prompt_blocks["reflect"] == PROMPT_TEMPLATE_ADAPTIVE_REFLECT
    assert adaptive_manager.system_prompt_manager._prompt_blocks["handle_input"] == PROMPT_TEMPLATE_BASE_HANDLE_INPUT


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

    # Verify initially no file tools
    initial_tool_count = len(agent.tools)
    initial_tool_description = agent.tool_description
    assert "FileReadTool" not in initial_tool_description
    assert "FileSearchTool" not in initial_tool_description

    # Create a dummy file
    dummy_file = BytesIO(b"test content")
    dummy_file.name = "test.txt"
    dummy_file.seek(0)

    # First run with files - triggers file tool addition
    input_data_with_files = {
        "input": "Analyze this file",
        "files": [dummy_file],
    }

    try:
        # This will add file tools
        agent.run(input_data=input_data_with_files, config=RunnableConfig())
    except Exception:
        # Execution may fail due to API calls, but tool addition happens before that
        pass

    # Verify file tools were added
    assert len(agent.tools) > initial_tool_count
    updated_tool_description = agent.tool_description
    assert "FileReadTool" in updated_tool_description or "file" in updated_tool_description.lower()

    # Verify _initial_variables was updated (the fix)
    assert agent.system_prompt_manager._initial_variables["tool_description"] == updated_tool_description

    # Call reset (simulates end of run)
    agent.reset_run_state()

    # Verify tool_description still includes file tools after reset
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

    # Verify tool description is still correct (includes file tools)
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

    # Verify system_prompt_manager exists
    assert agent.system_prompt_manager is not None
    assert isinstance(agent.system_prompt_manager, AgentPromptManager)

    # Serialize agent to dict
    agent_dict = agent.to_dict()

    # Verify system_prompt_manager is NOT in serialized output
    assert "system_prompt_manager" not in agent_dict

    # Verify it's in the exclude list
    assert "system_prompt_manager" in agent.to_dict_exclude_params
