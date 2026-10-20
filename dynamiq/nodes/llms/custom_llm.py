from typing import Any
from urllib.parse import urlsplit

from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.registry import _sync_to_litellm_enabled, model_registry


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

        endpoint = urlsplit(self.connection.url)
        if (
            self.provider_prefix == "anthropic"
            and _sync_to_litellm_enabled()
            and endpoint.hostname in {"api.minimax.io", "api.minimax.cn"}
            and endpoint.path.rstrip("/") == "/anthropic"
            and params["model"] in {"anthropic/MiniMax-M3.1-Flash", "anthropic/MiniMax-M3"}
        ):
            # LiteLLM resolves thinking capabilities under the routed provider, not
            # the native MiniMax registry key. Keep the alias out of ordinary listings.
            import litellm

            info = model_registry.get_model_info(params["model"])
            if info is not None:
                litellm.register_model({params["model"]: {**info, "litellm_provider": "anthropic"}})

        return params
