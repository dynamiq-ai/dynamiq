from typing import Any, ClassVar

from dynamiq.connections import AtlasCloud as AtlasCloudConnection
from dynamiq.nodes.llms.base import BaseLLM


class AtlasCloud(BaseLLM):
    """Atlas Cloud LLM node.

    This class provides an implementation for Large Language Model node that routes
    requests through Atlas Cloud's OpenAI-compatible API to various underlying providers.

    Atlas Cloud model ids are `vendor/model` strings (e.g. `openai/gpt-4.1-mini`,
    `deepseek-ai/deepseek-v3.2`). The node keeps `model` as the Atlas Cloud id and adds
    LiteLLM's `openai/` route prefix only on the request, so LiteLLM strips exactly that
    prefix and sends the id unchanged. Relying on `MODEL_PREFIX` or `custom_llm_provider`
    instead would drop the vendor part of ids that start with `openai/`.

    Attributes:
        connection (AtlasCloudConnection): The connection to use for the Atlas Cloud LLM.
        LITELLM_ROUTE_PREFIX (str): LiteLLM provider route for OpenAI-compatible endpoints.
    """

    connection: AtlasCloudConnection
    LITELLM_ROUTE_PREFIX: ClassVar[str] = "openai/"

    def __init__(self, **kwargs):
        """Initialize the Atlas Cloud LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AtlasCloudConnection()
        super().__init__(**kwargs)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Route the request through LiteLLM's OpenAI-compatible provider.

        Args:
            params (dict[str, Any]): The parameters to be sent to LiteLLM.

        Returns:
            dict[str, Any]: The parameters with the model routed to the OpenAI-compatible provider.
        """
        params = super().update_completion_params(params)
        params["model"] = f"{self.LITELLM_ROUTE_PREFIX}{params['model']}"
        return params
