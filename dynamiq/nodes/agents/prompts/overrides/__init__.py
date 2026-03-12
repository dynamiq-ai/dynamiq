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

    _GPT_PROMPTS = {
        "REACT_BLOCK_INSTRUCTIONS_SINGLE": REACT_BLOCK_INSTRUCTIONS_SINGLE,
        "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE": REACT_BLOCK_XML_INSTRUCTIONS_SINGLE,
    }

    _GPT_MODELS = [
        # GPT-5 series
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5-codex",
        "gpt-5-chat-latest",
        # GPT-5.1 series
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5.1-chat-latest",
        # GPT-5.2 series
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.2-codex",
        "gpt-5.2-chat-latest",
        # GPT-5.3 series
        "gpt-5.3-codex",
        "gpt-5.3-chat-latest",
        # GPT-5.4 series
        "gpt-5.4",
        "gpt-5.4-pro",
    ]

    for _model in _GPT_MODELS:
        register_model_prompts(model_name=_model, prompts=_GPT_PROMPTS)

    logger.debug("Registered GPT model prompts")
except ImportError as e:
    logger.debug(f"Could not load GPT prompts: {e}")

# Gemini Models (Google)
try:
    from dynamiq.nodes.agents.prompts.overrides.gemini import REACT_BLOCK_INSTRUCTIONS_SINGLE as GEMINI_REACT_SINGLE
    from dynamiq.nodes.agents.prompts.overrides.gemini import (
        REACT_BLOCK_XML_INSTRUCTIONS_SINGLE as GEMINI_REACT_XML_SINGLE,
    )

    register_model_prompts(
        model_name="gemini-3-pro-preview",
        prompts={
            "REACT_BLOCK_INSTRUCTIONS_SINGLE": GEMINI_REACT_SINGLE,
            "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE": GEMINI_REACT_XML_SINGLE,
        },
    )

    logger.debug("Registered Gemini model prompts")
except ImportError as e:
    logger.debug(f"Could not load Gemini prompts: {e}")
