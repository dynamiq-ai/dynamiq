from typing import Any

from dynamiq.connections import HttpApiKey
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM, BaseLLMInputSchema
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig


class OpenAI(BaseLLM):
    """OpenAI LLM node.

    This class provides an implementation for the OpenAI Language Model node.

    Attributes:
        connection (OpenAIConnection | None): The connection to use for the OpenAI LLM.
    """
    connection: OpenAIConnection | None = None

    def __init__(self, **kwargs):
        """Initialize the OpenAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    def is_o_family(self) -> bool:
        """Determine if the model belongs to the o_family (e.g. o1 or o3).

        Returns:
            bool: True if the model is an o_family model, otherwise False.
        """
        return "o1" in self.model or "o3" in self.model

    def execute(
        self,
        input_data: BaseLLMInputSchema,
        config: RunnableConfig = None,
        prompt: Prompt | None = None,
        schema: dict | None = None,
        inference_mode: InferenceMode | None = None,
        **kwargs,
    ):
        """Execute the LLM node.

        This method processes the input data, formats the prompt, and generates a response using
        the configured LLM.

        Args:
            input_data (BaseLLMInputSchema): The input data for the LLM.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            prompt (Prompt, optional): The prompt to use for this execution. Defaults to None.
            schema (Dict[str, Any], optional): schema_ for structured output or function calling.
                Overrides instance schema_ if provided.
            inference_mode (InferenceMode, optional): Mode of inference.
                Overrides instance inference_mode if provided.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content and tool calls.
        """
        config = ensure_config(config)
        prompt = prompt or self.prompt or Prompt(messages=[], tools=None)
        messages = prompt.format_messages(**dict(input_data))
        base_tools = prompt.format_tools(**dict(input_data))
        self.run_on_node_execute_run(callbacks=config.callbacks, prompt_messages=messages, **kwargs)

        params = self.connection.conn_params
        if self.client and not isinstance(self.connection, HttpApiKey):
            params = {"client": self.client}

        current_inference_mode = inference_mode or self.inference_mode
        current_schema = schema or self.schema_
        response_format, tools = self._get_response_format_and_tools(
            inference_mode=current_inference_mode, schema=current_schema
        )
        tools = tools or base_tools

        common_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.streaming.enabled,
            "tools": tools,
            "tool_choice": self.tool_choice,
            "stop": self.stop,
            "top_p": self.top_p,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": response_format,
            "drop_params": True,
            **params,
        }

        if self.is_o_family():
            common_params["max_completion_tokens"] = self.max_tokens
        else:
            common_params["temperature"] = self.temperature
            common_params["max_tokens"] = self.max_tokens

        response = self._completion(**common_params)

        handle_completion = (
            self._handle_streaming_completion_response if self.streaming.enabled else self._handle_completion_response
        )

        return handle_completion(
            response=response, messages=messages, config=config, input_data=dict(input_data), **kwargs
        )
