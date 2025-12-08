"""Prompt management system for agents.

This package provides:
- Default prompts and templates in `templates/defaults/`
- Model-specific prompt overrides in `templates/models/`
- Registry for managing model-specific prompts
- AgentPromptManager for centralized prompt setup
"""

from dynamiq.nodes.agents.prompts.base import AgentPromptManager
from dynamiq.nodes.agents.prompts.templates import get_prompt_constant, get_registry, register_model_prompts
