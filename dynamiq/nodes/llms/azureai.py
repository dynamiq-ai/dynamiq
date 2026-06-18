from dynamiq.connections import AzureAI as AzureAIConnection
from dynamiq.nodes.llms._strict import OpenAIStrictToolsMixin
from dynamiq.nodes.llms.base import BaseLLM


class AzureAI(OpenAIStrictToolsMixin, BaseLLM):
    """AzureAI LLM node.

    This class provides an implementation for the AzureAI Language Model node.

    Azure OpenAI shares OpenAI's per-tool ``strict`` structured-outputs support and
    LiteLLM forwards the flag unchanged, so strict tool transformation is applied
    via :class:`OpenAIStrictToolsMixin`.

    Attributes:
        connection (AzureAIConnection | None): The connection to use for the AzureAI LLM.
        MODEL_PREFIX (str): The prefix for the AzureAI model name.
    """
    connection: AzureAIConnection | None = None
    MODEL_PREFIX = "azure/"

    def __init__(self, **kwargs):
        """Initialize the AzureAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AzureAIConnection()
        super().__init__(**kwargs)
