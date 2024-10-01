from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.base import BaseLLM


class Bedrock(BaseLLM):
    """Bedrock LLM node.

    This class provides an implementation for the Bedrock Language Model node.

    Attributes:
        connection (AWSConnection | None): The connection to use for the Bedrock LLM.
        MODEL_PREFIX (str): The prefix for the Bedrock model name.
    """
    connection: AWSConnection | None = None
    MODEL_PREFIX = "bedrock/"

    def __init__(self, **kwargs):
        """Initialize the Bedrock LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)
