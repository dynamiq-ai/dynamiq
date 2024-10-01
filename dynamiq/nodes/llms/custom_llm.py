from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.base import BaseLLM


class CustomLLM(BaseLLM):
    """Class for creating custom LLM.

    This class provides a foundation for sending various requests using different LLM endpoints.

    Attributes:
        connection (HttpApiKey): The connection to use for the LLM.
    """
    connection: HttpApiKey
