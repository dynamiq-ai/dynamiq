from pydantic import model_validator

from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import GeminiVertexAI
from dynamiq.nodes.llms.base import BaseLLM

VERTEXAI_MODEL_PREFIX = "vertex_ai/"
GEMINI_MODEL_PREFIX = "gemini/"


class Gemini(BaseLLM):
    """Gemini LLM node.

    This class provides an implementation for the Gemini Language Model node.

    Attributes:
        connection (GeminiConnection | GeminiVertexAI): The connection to use for the Gemini LLM.
    """
    connection: GeminiConnection | GeminiVertexAI

    @model_validator(mode="after")
    def check_model(self) -> "Gemini":
        """Validate and set the model prefix based on the connection type.

        Returns:
            self: The updated instance.
        """
        connection = self.connection
        value = self.model
        if not value.startswith(VERTEXAI_MODEL_PREFIX) and isinstance(connection, GeminiVertexAI):
            self.model = f"{VERTEXAI_MODEL_PREFIX}{value}"
        elif not value.startswith(GEMINI_MODEL_PREFIX) and isinstance(connection, GeminiConnection):
            self.model = f"{GEMINI_MODEL_PREFIX}{value}"
        return self

    def __init__(self, **kwargs):
        """Initialize the Gemini LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)
