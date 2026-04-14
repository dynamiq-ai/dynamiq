"""Helper functions and managers for model-specific prompts."""

import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from jinja2 import Template

from dynamiq.nodes.agents.prompts.react import (
    REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING,
    REACT_BLOCK_INSTRUCTIONS_NO_TOOLS,
    REACT_BLOCK_INSTRUCTIONS_SINGLE,
    REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT,
    REACT_BLOCK_OUTPUT_FORMAT,
    REACT_BLOCK_TOOLS,
    REACT_BLOCK_TOOLS_BRIEF,
    REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS,
    REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
    REACT_MAX_LOOPS_PROMPT,
)
from dynamiq.nodes.agents.prompts.registry import get_prompt_constant
from dynamiq.nodes.agents.prompts.secondary_instructions import (
    CONTEXT_MANAGER_INSTRUCTIONS,
    DELEGATION_INSTRUCTIONS,
    DELEGATION_INSTRUCTIONS_XML,
    REACT_BLOCK_MULTI_TOOL_PLANNING,
    SANDBOX_INSTRUCTIONS_TEMPLATE,
    SUB_AGENT_INSTRUCTIONS,
    TODO_TOOLS_INSTRUCTIONS,
)
from dynamiq.nodes.agents.prompts.templates import AGENT_PROMPT_TEMPLATE
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger


@dataclass
class ReactPromptConfig:
    """Configuration for building a ReAct agent prompt."""

    inference_mode: InferenceMode
    has_tools: bool
    parallel_tool_calls_enabled: bool = False
    delegation_allowed: bool = False
    context_compaction_enabled: bool = False
    todo_management_enabled: bool = False
    sandbox_base_path: str | None = None
    has_sub_agent_tools: bool = False
    role: str | None = None
    instructions: str | None = None


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
                formatted_content = Template(content).render(temp_variables)
                if content:
                    formatted_prompt_blocks[block] = formatted_content

        render_context = {**temp_variables, **formatted_prompt_blocks}
        prompt = Template(self.agent_template).render(render_context).strip()
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

        # Pass dict directly to avoid crash on non-identifier keys
        return Template(template_content).render(variables)

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

    def build_react_prompt(self, config: ReactPromptConfig) -> None:
        """Build the complete prompt for a ReAct-style agent from a single config.

        This is the single entry point for ReAct prompt assembly. It handles:
        model-specific template, instruction/tool blocks by inference mode,
        operational instructions from feature flags, environment block,
        user-provided role and instructions.
        """
        model = self.model_name

        # 1. Model-specific template
        agent_template = get_prompt_constant(model, "AGENT_PROMPT_TEMPLATE", AGENT_PROMPT_TEMPLATE)
        if agent_template:
            self.agent_template = agent_template

        # 2. Instruction and tool blocks based on inference mode
        instructions_default = get_prompt_constant(
            model, "REACT_BLOCK_INSTRUCTIONS_SINGLE", REACT_BLOCK_INSTRUCTIONS_SINGLE
        )
        react_block_tools = get_prompt_constant(model, "REACT_BLOCK_TOOLS", REACT_BLOCK_TOOLS)
        instructions_no_tools = get_prompt_constant(
            model, "REACT_BLOCK_INSTRUCTIONS_NO_TOOLS", REACT_BLOCK_INSTRUCTIONS_NO_TOOLS
        )
        output_format = get_prompt_constant(model, "REACT_BLOCK_OUTPUT_FORMAT", REACT_BLOCK_OUTPUT_FORMAT)

        prompt_blocks: dict[str, str] = {
            "tools": "" if not config.has_tools else react_block_tools,
            "instructions": instructions_no_tools if not config.has_tools else instructions_default,
            "output_format": output_format,
        }

        match config.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                prompt_blocks["instructions"] = get_prompt_constant(
                    model, "REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING", REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING
                )
                if config.has_tools:
                    prompt_blocks["tools"] = get_prompt_constant(
                        model, "REACT_BLOCK_TOOLS_BRIEF", REACT_BLOCK_TOOLS_BRIEF
                    )
            case InferenceMode.STRUCTURED_OUTPUT:
                prompt_blocks["instructions"] = get_prompt_constant(
                    model, "REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT", REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT
                )
            case InferenceMode.XML:
                xml_no_tools = get_prompt_constant(
                    model, "REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS", REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS
                )
                xml_with_tools = get_prompt_constant(
                    model, "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE", REACT_BLOCK_XML_INSTRUCTIONS_SINGLE
                )
                prompt_blocks["instructions"] = xml_no_tools if not config.has_tools else xml_with_tools

        # 3. Operational instructions from feature flags
        ops_parts: list[str] = []
        if config.parallel_tool_calls_enabled:
            ops_parts.append(REACT_BLOCK_MULTI_TOOL_PLANNING)
        if config.delegation_allowed:
            if config.inference_mode == InferenceMode.XML:
                ops_parts.append(DELEGATION_INSTRUCTIONS_XML)
            else:
                ops_parts.append(DELEGATION_INSTRUCTIONS)
        if config.context_compaction_enabled:
            ops_parts.append(CONTEXT_MANAGER_INSTRUCTIONS)
        if config.todo_management_enabled:
            ops_parts.append(TODO_TOOLS_INSTRUCTIONS)
        if config.has_sub_agent_tools:
            ops_parts.append(SUB_AGENT_INSTRUCTIONS)

        # Append user-provided instructions (with raw wrapping)
        if config.instructions:
            user_instructions = config.instructions
            if ("{% raw %}" not in user_instructions) and ("{% endraw %}" not in user_instructions):
                user_instructions = f"{{% raw %}}{user_instructions}{{% endraw %}}"
            ops_parts.append(user_instructions)

        if ops_parts:
            prompt_blocks["operational_instructions"] = "\n\n".join(ops_parts)

        # 4. Environment block
        if config.sandbox_base_path:
            prompt_blocks["environment"] = SANDBOX_INSTRUCTIONS_TEMPLATE.format(
                base_path=config.sandbox_base_path,
            )

        # 5. Role block (with raw wrapping)
        if config.role:
            if ("{% raw %}" in config.role) or ("{% endraw %}" in config.role):
                prompt_blocks["role"] = config.role
            else:
                prompt_blocks["role"] = f"{{% raw %}}{config.role}{{% endraw %}}"

        # 6. Apply all blocks and runtime prompts
        self._prompt_blocks.update(prompt_blocks)
        self.max_loops_prompt = get_prompt_constant(model, "REACT_MAX_LOOPS_PROMPT", REACT_MAX_LOOPS_PROMPT)
