from dynamiq.nodes.agents.prompts import get_prompt_constant
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager
from dynamiq.nodes.agents.prompts.overrides.gpt import (
    REACT_BLOCK_XML_INSTRUCTIONS_SINGLE as GPT_REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
)
from dynamiq.nodes.agents.prompts.react import XML_DIRECT_OUTPUT_CAPABILITIES
from dynamiq.nodes.types import InferenceMode


def test_init():
    """Test basic initialization."""
    manager = AgentPromptManager(model_name="gpt-5.1", tool_description="Calculator")

    assert manager.model_name == "gpt-5.1"
    assert "tool_description" in manager._prompt_variables
    assert manager._prompt_variables["tool_description"] == "Calculator"


def test_set_block():
    """Test setting prompt blocks."""
    manager = AgentPromptManager()

    manager.set_block("instructions", "Be concise")

    assert manager._prompt_blocks["instructions"] == "Be concise"


def test_update_blocks():
    """Test updating multiple blocks."""
    manager = AgentPromptManager()

    manager.update_blocks({"instructions": "Be helpful", "context": "User is a developer"})

    assert manager._prompt_blocks["instructions"] == "Be helpful"
    assert manager._prompt_blocks["context"] == "User is a developer"


def test_update_variables():
    """Test updating multiple variables."""
    manager = AgentPromptManager()

    manager.update_variables({"var1": "value1", "var2": "value2"})

    assert manager._prompt_variables["var1"] == "value1"
    assert manager._prompt_variables["var2"] == "value2"


def test_reset():
    """Test resetting to initial state."""
    manager = AgentPromptManager(tool_description="Calculator")

    # Modify state - both variables and blocks
    manager.set_variable("temp_var", "temp")
    assert "temp_var" in manager._prompt_variables

    manager.set_block("custom_block", "Custom content")
    assert "custom_block" in manager._prompt_blocks
    assert manager._prompt_blocks["custom_block"] == "Custom content"

    # Modify an initial block
    manager.set_block("instructions", "Modified instructions")
    assert manager._prompt_blocks["instructions"] == "Modified instructions"

    # Reset
    manager.reset()

    # Verify variables are reset
    assert "temp_var" not in manager._prompt_variables
    assert "tool_description" in manager._prompt_variables
    assert manager._prompt_variables["tool_description"] == "Calculator"

    # Verify blocks are reset
    assert "custom_block" not in manager._prompt_blocks
    assert manager._prompt_blocks["instructions"] == ""  # Back to initial empty state

    # Verify initial blocks are still present
    assert "date" in manager._prompt_blocks
    assert "tools" in manager._prompt_blocks
    assert "context" in manager._prompt_blocks


def test_reset_removes_role_block():
    """Test that reset() removes dynamically added role blocks.

    This addresses the concern that if an agent's role attribute is set on one run
    then removed, the old 'role' block would persist across runs without proper reset.
    """
    manager = AgentPromptManager(tool_description="Helper")

    # Simulate first run: agent has a role
    manager.set_block("role", "You are a helpful assistant")
    assert "role" in manager._prompt_blocks
    assert manager._prompt_blocks["role"] == "You are a helpful assistant"

    # Generate prompt with role
    prompt = manager.generate_prompt()
    assert "You are a helpful assistant" in prompt

    # Reset between runs
    manager.reset()

    # Verify role block is removed
    assert "role" not in manager._prompt_blocks

    # Generate prompt again - role should not appear
    prompt = manager.generate_prompt()
    assert "You are a helpful assistant" not in prompt


def test_generate_prompt_basic():
    """Test basic prompt generation."""
    manager = AgentPromptManager(tool_description="Math tools")
    manager.set_block("instructions", "Solve math problems")

    prompt = manager.generate_prompt()

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_date_variable_exists():
    """Test that date variable is automatically set."""
    manager = AgentPromptManager()

    assert "date" in manager._prompt_variables
    assert isinstance(manager._prompt_variables["date"], str)


def test_date_refreshed_on_reset():
    """Test that date is refreshed on reset."""
    manager = AgentPromptManager()
    manager._prompt_variables["block1"] = "Content1"

    manager.reset()

    assert "block1" not in manager._prompt_variables


def test_apply_model_specific_prompts_with_model():
    """Test applying model-specific prompts when model is set."""
    manager = AgentPromptManager(model_name="gpt-5.1")
    manager.setup_for_react_agent(
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=False,
        direct_tool_output_enabled=True,
        has_tools=True,
    )
    prompt = manager.generate_prompt()

    assert prompt is not None
    assert XML_DIRECT_OUTPUT_CAPABILITIES in prompt
    assert GPT_REACT_BLOCK_XML_INSTRUCTIONS_SINGLE == get_prompt_constant(
        "gpt-5.1", "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE", None
    )
