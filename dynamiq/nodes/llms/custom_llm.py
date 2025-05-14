from dynamiq.connections import Http
from dynamiq.nodes.llms.base import BaseLLM


class CustomLLM(BaseLLM):
    """Class for creating custom LLM.

    This class provides a foundation for sending various requests using different LLM endpoints.

    Attributes:
        connection (Http): The connection to use for the LLM.
    """

    connection: Http
