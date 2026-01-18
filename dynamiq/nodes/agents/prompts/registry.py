"""Registry for managing model-specific prompts."""

from typing import Any

from dynamiq.utils.logger import logger


class ModelPromptsRegistry:
    """Registry for managing model-specific prompts."""

    def __init__(self):
        """Initialize the prompts registry."""
        self._registry: dict[str, dict[str, str]] = {}

    def register(self, model_name: str, prompts: dict[str, str]) -> None:
        """
        Register prompts for a specific model.

        Args:
            model_name: The model identifier (e.g., "gpt-5.1", "claude-sonnet-4.5")
            prompts: Dictionary of prompt constant names and their values
        """
        if model_name in self._registry:
            logger.warning(f"Overwriting existing prompts for model '{model_name}'")
        self._registry[model_name] = prompts
        logger.debug(f"Registered {len(prompts)} custom prompts for model '{model_name}'")

    def get(self, model_name: str, prompt_name: str, default: Any = None) -> Any:
        """
        Get a specific prompt for a model.

        Args:
            model_name: The model identifier
            prompt_name: The name of the prompt constant
            default: Default value if prompt not found

        Returns:
            The prompt value or default
        """

        if not model_name:
            return default

        # Try exact match first
        if model_name in self._registry:
            if prompt_name in self._registry[model_name]:
                logger.debug(f"Using model-specific prompt '{prompt_name}' for model '{model_name}'")
                return self._registry[model_name][prompt_name]

        # Try normalized variations
        normalized_names = self._normalize_model_name(model_name)
        for name in normalized_names:
            if name in self._registry:
                if prompt_name in self._registry[name]:
                    logger.debug(
                        f"Using model-specific prompt '{prompt_name}' for model '{name}' (matched from '{model_name}')"
                    )
                    return self._registry[name][prompt_name]

        return default

    def _normalize_model_name(self, model_name: str) -> list[str]:
        """
        Generate possible variations of a model name for matching.

        Args:
            model_name: Original model name

        Returns:
            List of possible name variations
        """
        variations = [model_name]

        # Add lowercase version
        lower = model_name.lower()
        if lower != model_name:
            variations.append(lower)

        # Add version with underscores instead of hyphens
        with_underscores = lower.replace("-", "_")
        if with_underscores != lower:
            variations.append(with_underscores)

        # Add version without provider prefix (e.g., "openai/gpt-4" -> "gpt-4")
        if "/" in model_name:
            without_provider = model_name.split("/", 1)[1]
            variations.append(without_provider)
            variations.append(without_provider.lower())
            variations.append(without_provider.lower().replace("-", "_"))

        return variations


# Global registry instance
_prompts_registry = ModelPromptsRegistry()


def register_model_prompts(model_name: str, prompts: dict[str, str]) -> None:
    """
    Register prompts for a specific model.

    Args:
        model_name: The model identifier
        prompts: Dictionary of prompt constant names and their values
    """
    _prompts_registry.register(model_name, prompts)


def get_prompt_constant(model_name: str, constant_name: str, default_value: Any) -> Any:
    """
    Get a prompt constant for a model or fall back to default.

    Args:
        model_name: The model name to look up
        constant_name: The name of the constant to retrieve
        default_value: The default value if model-specific prompt is not found

    Returns:
        The model-specific prompt or the default value
    """
    return _prompts_registry.get(model_name, constant_name, default_value)


def get_registry() -> ModelPromptsRegistry:
    """Get the global prompts registry."""
    return _prompts_registry
