from dynamiq.connections import AtlasCloud as AtlasCloudConnection
from dynamiq.nodes.llms.base import BaseLLM


class AtlasCloud(BaseLLM):
    """Atlas Cloud LLM node.

    This class provides an implementation for Large Language Model node that routes
    requests through Atlas Cloud to various underlying providers.

    Atlas Cloud model ids are already fully-qualified `vendor/model` strings (e.g.
    `openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4.6`), so unlike OpenRouter this node
    does not set `MODEL_PREFIX` - the model name is sent to Atlas Cloud unmodified, and
    `AtlasCloudConnection.completion_params` tells LiteLLM how to route it instead.

    Attributes:
        connection (AtlasCloudConnection): The connection to use for the Atlas Cloud LLM.
    """

    connection: AtlasCloudConnection

    def __init__(self, **kwargs):
        """Initialize the Atlas Cloud LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AtlasCloudConnection()
        super().__init__(**kwargs)
