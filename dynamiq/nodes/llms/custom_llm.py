from typing import Any

from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.base import BaseLLM


class CustomLLM(BaseLLM):
    """
    Custom LLM implementation for third-party providers requiring specific formatting.

    This class extends BaseLLM to support providers like OpenRouter, Anthropic, or custom
    endpoints that need special request formatting. It allows adding provider prefixes to
    model names.

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

        This method adds the provider prefix to model name if specified
        (e.g., "openrouter/anthropic/claude-2").

        Args:
            params (dict[str, Any]): The original parameters to be sent to the LLM provider.
                                    This includes model, messages, temperature, etc.

        Returns:
            dict[str, Any]: The modified parameters with proper model name formatting.
        """
        params = super().update_completion_params(params)

        if self.provider_prefix and not params["model"].startswith(self.provider_prefix + "/"):
            params["model"] = f"{self.provider_prefix}/{params['model']}"

        return params
