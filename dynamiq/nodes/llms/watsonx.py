from dynamiq.connections import WatsonX as WatsonXConnection
from dynamiq.nodes.llms.base import BaseLLM


class WatsonX(BaseLLM):
    """WatsonX LLM node.

    This class provides an implementation for the WatsonX Language Model node.

    Attributes:
        connection (WatsonXConnection | None): The connection to use for the WatsonX LLM.
        MODEL_PREFIX (str): The prefix for the WatsonX model name.
    """
    connection: WatsonXConnection | None = None
    MODEL_PREFIX = "watsonx_text/"

    def __init__(self, **kwargs):
        """Initialize the WatsonX LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = WatsonXConnection()
        super().__init__(**kwargs)
