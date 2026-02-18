"""Model-specific prompt overrides.

This module automatically registers all model-specific prompts.
"""

from dynamiq.nodes.agents.prompts.registry import register_model_prompts
from dynamiq.utils.logger import logger

# ============================================================================
# Auto-register all model-specific prompts
# ============================================================================

# GPT Models (OpenAI)
try:
    from dynamiq.nodes.agents.prompts.overrides.gpt import (
        REACT_BLOCK_INSTRUCTIONS_SINGLE,
        REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
    )

    gpt_prompts = {
        "REACT_BLOCK_INSTRUCTIONS_SINGLE": REACT_BLOCK_INSTRUCTIONS_SINGLE,
        "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE": REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
    }
    gpt_models = (
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.2",
        "gpt-5-mini",
        "gpt-5.2-pro",
    )
    for model_name in gpt_models:
        register_model_prompts(model_name=model_name, prompts=gpt_prompts)

    logger.debug("Registered GPT model prompts")
except ImportError as e:
    logger.debug(f"Could not load GPT prompts: {e}")

# Gemini Models (Google)
try:
    from dynamiq.nodes.agents.prompts.overrides.gemini import REACT_BLOCK_INSTRUCTIONS_SINGLE as GEMINI_REACT_SINGLE
    from dynamiq.nodes.agents.prompts.overrides.gemini import (
        REACT_BLOCK_XML_INSTRUCTIONS_SINGLE as GEMINI_REACT_XML_SINGLE,
    )

    gemini_prompts = {
        "REACT_BLOCK_INSTRUCTIONS_SINGLE": GEMINI_REACT_SINGLE,
        "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE": GEMINI_REACT_XML_SINGLE,
    }
    gemini_models = (
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
    )
    for model_name in gemini_models:
        register_model_prompts(model_name=model_name, prompts=gemini_prompts)

    logger.debug("Registered Gemini model prompts")
except ImportError as e:
    logger.debug(f"Could not load Gemini prompts: {e}")
