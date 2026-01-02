"""Prompt management system for agents.

This package provides:
- AgentPromptManager for centralized prompt management
- Registry for model-specific prompt overrides
- Default prompts and templates
- Model-specific overrides in `overrides/`
"""

# Import overrides to trigger auto-registration
from dynamiq.nodes.agents.prompts import overrides  # noqa: F401
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager
from dynamiq.nodes.agents.prompts.registry import get_prompt_constant, get_registry, register_model_prompts

__all__ = ["AgentPromptManager", "get_prompt_constant", "get_registry", "register_model_prompts"]
