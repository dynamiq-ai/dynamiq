from dynamiq.connections import Groq as GroqConnection
from dynamiq.nodes.llms.base import BaseLLM


class Groq(BaseLLM):
    """Groq LLM node.

    This class provides an implementation for the Groq Language Model node.

    Attributes:
        connection (GroqConnection | None): The connection to use for the Groq LLM.
        MODEL_PREFIX (str): The prefix for the Groq model name.
    """
    connection: GroqConnection | None = None
    MODEL_PREFIX = "groq/"

    def __init__(self, **kwargs):
        """Initialize the Groq LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GroqConnection()
        super().__init__(**kwargs)
