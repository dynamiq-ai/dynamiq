from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.llms._strict import OpenAIStrictToolsMixin
from dynamiq.nodes.llms.base import BaseLLM


class TogetherAI(OpenAIStrictToolsMixin, BaseLLM):
    """TogetherAI LLM node.

    This class provides an implementation for the TogetherAI Language Model node.

    Together's OpenAI-compatible Chat Completions API accepts a per-tool ``strict``
    flag and constrains generated tool arguments to the schema (see Together's
    function-calling best-practices docs). Strict tool transformation is applied
    via :class:`OpenAIStrictToolsMixin`, the same OpenAI structured-outputs shape
    used by Cerebras and DeepSeek; LiteLLM forwards the flag and tightened schema
    through the OpenAI-compatible path unchanged.

    Attributes:
        connection (TogetherAIConnection | None): The connection to use for the TogetherAI LLM.
        MODEL_PREFIX (str): The prefix for the TogetherAI model name.
    """
    connection: TogetherAIConnection | None = None
    MODEL_PREFIX = "together_ai/"

    def __init__(self, **kwargs):
        """Initialize the TogetherAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = TogetherAIConnection()
        super().__init__(**kwargs)
