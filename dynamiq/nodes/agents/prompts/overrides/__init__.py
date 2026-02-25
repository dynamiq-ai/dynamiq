"""Model-specific prompt overrides.

This module automatically registers all model-specific prompts.
"""

from dynamiq.nodes.agents.prompts.registry import register_model_prompts
from dynamiq.utils.logger import logger

# ============================================================================
# Auto-register all model-specific prompts
# ============================================================================

# Model-specific prompt overrides (agent uses function calling only; no XML/SINGLE prompts).
# Register REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING here if a model needs custom FC instructions.
