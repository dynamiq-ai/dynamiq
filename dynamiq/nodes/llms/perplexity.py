from dynamiq.connections import Perplexity as PerplexityConnection
from dynamiq.nodes.llms.base import BaseLLM


class Perplexity(BaseLLM):
    """Perplexity LLM node.

    This class provides an implementation for the Perplexity Language Model node.

    Attributes:
        connection (PerplexityConnection): The connection to use for the Perplexity LLM.
        MODEL_PREFIX (str): The prefix for the Perplexity model name.
    """

    connection: PerplexityConnection
    MODEL_PREFIX = "perplexity/"

    def __init__(self, **kwargs):
        """Initialize the Replicate LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = PerplexityConnection()
        super().__init__(**kwargs)
