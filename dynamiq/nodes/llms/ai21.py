from typing import Any

from pydantic import field_validator

from dynamiq.connections import AI21 as AI21Connection
from dynamiq.nodes.llms.base import BaseLLM


class AI21(BaseLLM):
    """AI21 LLM node.

    This class provides an implementation for the AI21 Language Model node.

    Attributes:
        connection (AI21Connection): The connection to use for the AI21 LLM.
        MODEL_PREFIX (str): The prefix for the AI21 model name.
    """
    connection: AI21Connection
    MODEL_PREFIX = "ai21/"

    @field_validator("model", mode="before")
    @classmethod
    def _save_with_ai21_prefix(cls, value: str | None) -> str | None:
        """Save a model given with litellm's "ai21_chat/" prefix as "ai21/<name>", the form earlier releases read."""
        if isinstance(value, str) and value.startswith("ai21_chat/"):
            return cls.MODEL_PREFIX + value.removeprefix("ai21_chat/")
        return value

    def __init__(self, **kwargs):
        """Initialize the AI21 LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AI21Connection()
        super().__init__(**kwargs)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send requests through litellm's "ai21_chat" provider.

        litellm sends an "ai21/" model it does not list, such as "jamba-large", to OpenAI's API with the AI21 key as
        the bearer token. The model stays saved as "ai21/<name>", so earlier releases can still run a saved workflow.
        """
        params = super().update_completion_params(params)
        params["model"] = "ai21_chat/" + params["model"].removeprefix(self.MODEL_PREFIX)
        return params
