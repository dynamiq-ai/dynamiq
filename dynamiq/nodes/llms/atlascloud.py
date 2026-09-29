from dynamiq.connections import AtlasCloud as AtlasCloudConnection
from dynamiq.nodes.llms.base import BaseLLM


class AtlasCloud(BaseLLM):
    """Atlas Cloud LLM node.

    This class provides an implementation for Large Language Model node that routes
    requests through Atlas Cloud's OpenAI-compatible API to various underlying providers.

    Atlas Cloud model ids are `vendor/model` strings (e.g. `openai/gpt-4.1-mini`,
    `deepseek-ai/deepseek-v3.2`). LiteLLM has no Atlas Cloud provider, so requests go
    through its OpenAI-compatible client pointed at the Atlas Cloud endpoint.

    Attributes:
        connection (AtlasCloudConnection): The connection to use for the Atlas Cloud LLM.
        MODEL_PREFIX (str): The LiteLLM prefix for OpenAI-compatible endpoints.
    """

    connection: AtlasCloudConnection
    MODEL_PREFIX = "openai/"

    def __init__(self, **kwargs):
        """Initialize the Atlas Cloud LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AtlasCloudConnection()
        super().__init__(**kwargs)
