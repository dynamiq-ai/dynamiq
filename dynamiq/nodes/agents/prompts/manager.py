"""Helper functions and managers for model-specific prompts."""

import re
import textwrap
from datetime import datetime
from typing import Any

from jinja2 import Template

from dynamiq.nodes.agents.prompts.react import (
    DELEGATION_INSTRUCTIONS,
    DELEGATION_INSTRUCTIONS_XML,
    HISTORY_SUMMARIZATION_PROMPT,
    REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING,
    REACT_BLOCK_INSTRUCTIONS_MULTI,
    REACT_BLOCK_INSTRUCTIONS_NO_TOOLS,
    REACT_BLOCK_INSTRUCTIONS_SINGLE,
    REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT,
    REACT_BLOCK_OUTPUT_FORMAT,
    REACT_BLOCK_TOOLS,
    REACT_BLOCK_TOOLS_NO_FORMATS,
    REACT_BLOCK_XML_INSTRUCTIONS_MULTI,
    REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS,
    REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
    REACT_MAX_LOOPS_PROMPT,
)
from dynamiq.nodes.agents.prompts.registry import get_prompt_constant
from dynamiq.nodes.agents.prompts.templates import AGENT_PROMPT_TEMPLATE
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger


class AgentPromptManager:
    """
    Manager for holding and managing model-specific prompts for an agent.

    Each agent instance has its own prompt manager that stores all
    prompts needed during the agent's lifetime, including prompt blocks,
    variables, and generation logic.

    Usage:
        agent.prompt_manager.history_prompt
        agent.prompt_manager.max_loops_prompt
        agent.prompt_manager.generate_prompt()
    """

    def __init__(self, model_name: str = None, tool_description: str = ""):
        """
        Initialize the prompt manager.

        Args:
            model_name: The LLM model name for model-specific prompts
            tool_description: Description of available tools
        """
        self.model_name = model_name

        # Runtime prompts (used during agent execution)
        self.history_prompt: str = HISTORY_SUMMARIZATION_PROMPT
        self.max_loops_prompt: str = REACT_MAX_LOOPS_PROMPT

        # Template
        self.agent_template: str = AGENT_PROMPT_TEMPLATE

        # Prompt blocks and variables
        # Store initial blocks for reset
        self._prompt_blocks: dict[str, str] = {
            "date": "{{ date }}",
            "tools": "{{ tool_description }}",
            "instructions": "",
            "context": "{{ context }}",
        }

        # Store initial variables for reset
        self._initial_variables: dict[str, Any] = {
            "tool_description": tool_description,
            "date": datetime.now().strftime("%d %B %Y"),
        }
        self._prompt_variables: dict[str, Any] = self._initial_variables.copy()

    def set_block(self, block_name: str, content: str):
        """Sets or updates a specific prompt block."""
        self._prompt_blocks[block_name] = content

    def update_blocks(self, blocks: dict[str, str]):
        """Updates multiple prompt blocks at once."""
        self._prompt_blocks.update(blocks)

    def set_variable(self, var_name: str, value: Any):
        """Sets or updates a specific prompt variable."""
        self._prompt_variables[var_name] = value

    def update_variables(self, variables: dict[str, Any], merge: bool = True):
        """
        Updates multiple prompt variables at once.

        Args:
            variables: Dictionary of variables to update
            merge: If True, merge with existing variables. If False, replace existing variables.
        """
        if merge:
            self._prompt_variables.update(variables)
        else:
            self._prompt_variables = variables.copy()

    def set_initial_variable(self, var_name: str, value: Any):
        """
        Sets or updates a specific initial variable that persists across resets.

        This method should be used when you need to update the initial state
        that the manager returns to when reset() is called.

        Args:
            var_name: Name of the variable to set
            value: Value to set for the variable
        """
        self._initial_variables[var_name] = value
        self._prompt_variables[var_name] = value

    def reset(self):
        """
        Resets prompt manager to its initial state.

        This should be called between runs to prevent variable accumulation.
        Prompt variables are reset to their initial state.
        The date is refreshed on each reset to ensure it's always current.
        """
        self._prompt_variables = self._initial_variables.copy()
        self._prompt_variables["date"] = datetime.now().strftime("%d %B %Y")

    def generate_prompt(self, block_names: list[str] | None = None, **kwargs) -> str:
        """
        Generates the prompt using specified blocks and variables.

        Args:
            block_names: Optional list of block names to include. If None, includes all blocks.
            **kwargs: Additional variables to use for rendering

        Returns:
            The generated prompt string
        """
        temp_variables = self._prompt_variables.copy()
        temp_variables.update(kwargs)

        formatted_prompt_blocks = {}
        for block, content in self._prompt_blocks.items():
            if block_names is None or block in block_names:
                formatted_content = Template(content).render(**temp_variables)
                if content:
                    formatted_prompt_blocks[block] = formatted_content

        prompt = Template(self.agent_template).render(formatted_prompt_blocks).strip()
        prompt = self._clean_prompt(prompt)
        return textwrap.dedent(prompt)

    def render_block(self, block_name: str, **kwargs) -> str:
        """
        Renders a specific prompt block with variables.

        This is useful for rendering individual blocks like 'plan', 'assign', 'final',
        or 'handle_input' in AgentManager workflows.

        Args:
            block_name: Name of the block to render
            **kwargs: Additional variables to merge with existing prompt variables

        Returns:
            Rendered block content as string
        """
        template_content = self._prompt_blocks.get(block_name)
        if not template_content:
            logger.warning(f"Block '{block_name}' not found in prompt blocks")
            return ""

        # Merge existing variables with provided kwargs
        variables = self._prompt_variables.copy()
        variables.update(kwargs)

        return Template(template_content).render(**variables)

    @staticmethod
    def _clean_prompt(prompt: str) -> str:
        """
        Cleans the generated prompt by removing excess blank lines.

        Args:
            prompt: The prompt to clean

        Returns:
            Cleaned prompt string
        """
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)
        return prompt.strip()

    def setup_for_base_agent(self) -> None:
        """Setup prompts for base Agent class."""
        if not self.model_name:
            return

        # Get model-specific template
        self.agent_template = get_prompt_constant(self.model_name, "AGENT_PROMPT_TEMPLATE", AGENT_PROMPT_TEMPLATE)

        if self.agent_template != AGENT_PROMPT_TEMPLATE:
            logger.info(f"Applied model-specific AGENT_PROMPT_TEMPLATE for model '{self.model_name}'")

    def setup_for_react_agent(
        self,
        inference_mode: InferenceMode,
        parallel_tool_calls_enabled: bool,
        has_tools: bool,
    ) -> None:
        """
        Setup prompts for ReAct-style Agent.

        Updates the prompt blocks with ReAct-specific prompts.
        """
        # Get all prompt blocks for this configuration
        prompt_blocks, agent_template = get_model_specific_prompts(
            model_name=self.model_name,
            inference_mode=inference_mode,
            parallel_tool_calls_enabled=parallel_tool_calls_enabled,
            has_tools=has_tools,
        )

        # Update prompt blocks
        self._prompt_blocks.update(prompt_blocks)

        # Store template
        if agent_template:
            self.agent_template = agent_template

        # Store runtime prompts
        self.history_prompt = get_prompt_constant(
            self.model_name, "HISTORY_SUMMARIZATION_PROMPT", HISTORY_SUMMARIZATION_PROMPT
        )
        self.max_loops_prompt = get_prompt_constant(self.model_name, "REACT_MAX_LOOPS_PROMPT", REACT_MAX_LOOPS_PROMPT)

        # Log only if model-specific prompts were actually applied
        if agent_template and agent_template != AGENT_PROMPT_TEMPLATE:
            logger.debug(f"Applied model-specific prompts and template for model '{self.model_name}'")
        else:
            logger.debug(f"Using default prompts for model '{self.model_name}'")

    def build_delegation_variables(self, delegation_allowed: bool = False) -> dict[str, str]:
        """
        Provide prompt snippets for delegate_final guidance when enabled.

        Args:
            delegation_allowed: Whether delegation is allowed for the agent

        Returns:
            Dictionary containing delegation instructions for both standard and XML formats
        """
        if not delegation_allowed:
            return {"delegation_instructions": "", "delegation_instructions_xml": ""}

        return {
            "delegation_instructions": DELEGATION_INSTRUCTIONS,
            "delegation_instructions_xml": DELEGATION_INSTRUCTIONS_XML,
        }


def get_model_specific_prompts(
    model_name: str,
    inference_mode: InferenceMode,
    parallel_tool_calls_enabled: bool,
    has_tools: bool,
) -> tuple[dict[str, str], str]:
    """
    Get model-specific prompts based on the model name and agent configuration.

    Args:
        model_name: The LLM model name
        inference_mode: The inference mode being used
        parallel_tool_calls_enabled: Whether parallel tool calls are enabled
        has_tools: Whether the agent has tools

    Returns:
        Tuple of (prompt_blocks dict, agent_prompt_template string)
    """
    # Get model-specific agent template
    agent_template = get_prompt_constant(model_name, "AGENT_PROMPT_TEMPLATE", AGENT_PROMPT_TEMPLATE)
    if agent_template != AGENT_PROMPT_TEMPLATE:
        logger.debug(f"Using model-specific AGENT_PROMPT_TEMPLATE for '{model_name}'")

    # Get base instructions based on parallel tool calls setting
    if parallel_tool_calls_enabled:
        instructions_default = get_prompt_constant(
            model_name, "REACT_BLOCK_INSTRUCTIONS_MULTI", REACT_BLOCK_INSTRUCTIONS_MULTI
        )
        instructions_xml = get_prompt_constant(
            model_name, "REACT_BLOCK_XML_INSTRUCTIONS_MULTI", REACT_BLOCK_XML_INSTRUCTIONS_MULTI
        )
        if instructions_default != REACT_BLOCK_INSTRUCTIONS_MULTI:
            logger.debug(f"Using model-specific REACT_BLOCK_INSTRUCTIONS_MULTI for '{model_name}'")
        if instructions_xml != REACT_BLOCK_XML_INSTRUCTIONS_MULTI:
            logger.debug(f"Using model-specific REACT_BLOCK_XML_INSTRUCTIONS_MULTI for '{model_name}'")
    else:
        instructions_default = get_prompt_constant(
            model_name, "REACT_BLOCK_INSTRUCTIONS_SINGLE", REACT_BLOCK_INSTRUCTIONS_SINGLE
        )
        instructions_xml = get_prompt_constant(
            model_name, "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE", REACT_BLOCK_XML_INSTRUCTIONS_SINGLE
        )
        if instructions_default != REACT_BLOCK_INSTRUCTIONS_SINGLE:
            logger.debug(f"Using model-specific REACT_BLOCK_INSTRUCTIONS_SINGLE for '{model_name}'")
        if instructions_xml != REACT_BLOCK_XML_INSTRUCTIONS_SINGLE:
            logger.debug(f"Using model-specific REACT_BLOCK_XML_INSTRUCTIONS_SINGLE for '{model_name}'")

    # Get other model-specific prompts
    react_block_tools = get_prompt_constant(model_name, "REACT_BLOCK_TOOLS", REACT_BLOCK_TOOLS)
    if react_block_tools != REACT_BLOCK_TOOLS:
        logger.debug(f"Using model-specific REACT_BLOCK_TOOLS for '{model_name}'")

    react_block_instructions_no_tools = get_prompt_constant(
        model_name, "REACT_BLOCK_INSTRUCTIONS_NO_TOOLS", REACT_BLOCK_INSTRUCTIONS_NO_TOOLS
    )
    if react_block_instructions_no_tools != REACT_BLOCK_INSTRUCTIONS_NO_TOOLS:
        logger.debug(f"Using model-specific REACT_BLOCK_INSTRUCTIONS_NO_TOOLS for '{model_name}'")

    react_block_output_format = get_prompt_constant(model_name, "REACT_BLOCK_OUTPUT_FORMAT", REACT_BLOCK_OUTPUT_FORMAT)
    if react_block_output_format != REACT_BLOCK_OUTPUT_FORMAT:
        logger.debug(f"Using model-specific REACT_BLOCK_OUTPUT_FORMAT for '{model_name}'")

    # Build initial prompt blocks
    prompt_blocks = {
        "tools": "" if not has_tools else react_block_tools,
        "instructions": react_block_instructions_no_tools if not has_tools else instructions_default,
        "output_format": react_block_output_format,
    }

    # Override based on inference mode
    match inference_mode:
        case InferenceMode.FUNCTION_CALLING:
            prompt_blocks["instructions"] = get_prompt_constant(
                model_name, "REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING", REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING
            )
            if has_tools:
                prompt_blocks["tools"] = get_prompt_constant(
                    model_name, "REACT_BLOCK_TOOLS_NO_FORMATS", REACT_BLOCK_TOOLS_NO_FORMATS
                )

        case InferenceMode.STRUCTURED_OUTPUT:
            prompt_blocks["instructions"] = get_prompt_constant(
                model_name, "REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT", REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT
            )

        case InferenceMode.XML:
            xml_instructions_no_tools = get_prompt_constant(
                model_name, "REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS", REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS
            )
            prompt_blocks["instructions"] = xml_instructions_no_tools if not has_tools else instructions_xml

    return prompt_blocks, agent_template
