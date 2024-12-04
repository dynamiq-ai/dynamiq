from typing import Union

from litellm import CustomStreamWrapper, ModelResponse
from pydantic import Field

from dynamiq.connections import Perplexity as PerplexityConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.runnables import RunnableConfig


class Perplexity(BaseLLM):
    """Perplexity LLM node.

    This class provides an implementation for the Perplexity Language Model node.

    Attributes:
        connection (PerplexityConnection): The connection to use for the Perplexity LLM.
        MODEL_PREFIX (str): The prefix for the Perplexity model name.
        return_citations (bool): Whether to return citations in the response.
    """

    connection: PerplexityConnection
    return_citations: bool = Field(default=False, description="Whether to return citations in the response")

    MODEL_PREFIX = "perplexity/"

    def __init__(self, **kwargs):
        """Initialize the Perplexity LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = PerplexityConnection()

        if "connection" in kwargs:
            kwargs["connection"].conn_params["return_citations"] = kwargs.get("return_citations", False)

        super().__init__(**kwargs)

    def _handle_completion_response(
        self,
        response: Union["ModelResponse", "CustomStreamWrapper"],
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict:
        """Handle completion response with citations.

        Args:
            response (ModelResponse | CustomStreamWrapper): The response from the LLM.
            config (RunnableConfig, optional): The configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content, tool calls, and citations.
        """
        result = super()._handle_completion_response(response, config, **kwargs)

        if hasattr(response, "citations"):
            result["citations"] = response.citations

        return result
