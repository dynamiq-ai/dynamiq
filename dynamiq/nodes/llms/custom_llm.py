from typing import Any

from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.base import BaseLLM


class CustomLLM(BaseLLM):
    """
    Custom LLM implementation for third-party providers requiring specific formatting.

    This class extends BaseLLM to support providers like OpenRouter, Anthropic, or custom
    endpoints that need special request formatting. It allows adding provider prefixes to
    model names and handles custom headers required by these services.

    Attributes:
        name (str | None): Name of the LLM node. Defaults to "CustomLLM".
        connection (HttpApiKey): Connection to use for the LLM API.
        provider_prefix (str | None): Provider prefix to add to model names (e.g., "openrouter").
                                     When specified, this will automatically prepend
                                     "{provider_prefix}/" to the model name when sending requests.

    """

    name: str | None = "CustomLLM"
    connection: HttpApiKey
    provider_prefix: str | None = None

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Customizes LLM request parameters for third-party providers.

        This method performs two main modifications:
        1. Adds the provider prefix to model name if specified (e.g., "openrouter/anthropic/claude-2")
        2. Handles custom headers from extra parameters and prevents header duplication

        Args:
            params (dict[str, Any]): The original parameters to be sent to the LLM provider.
                                    This includes model, messages, temperature, etc.

        Returns:
            dict[str, Any]: The modified parameters with proper model name formatting
                           and header configuration for the provider.
        """
        params = super().update_completion_params(params)

        extra = getattr(self, "__pydantic_extra__", {}) or {}

        if self.provider_prefix and not params["model"].startswith(self.provider_prefix):
            params["model"] = f"{self.provider_prefix}/{params['model']}"

        if extra and "headers" in extra:
            params["headers"] = {**params.get("headers", {}), **extra["headers"]}
            params.pop("extra_headers", None)

        return params
