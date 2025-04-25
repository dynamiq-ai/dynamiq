import json
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from dynamiq.connections import BaseConnection, HttpApiKey
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types.llm_tool import Tool

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper, ModelResponse


class BaseLLMUsageData(BaseModel):
    """Model for LLM usage data.

    Attributes:
        prompt_tokens (int): Number of prompt tokens.
        prompt_tokens_cost_usd (float | None): Cost of prompt tokens in USD.
        completion_tokens (int): Number of completion tokens.
        completion_tokens_cost_usd (float | None): Cost of completion tokens in USD.
        total_tokens (int): Total number of tokens.
        total_tokens_cost_usd (float | None): Total cost of tokens in USD.
    """
    prompt_tokens: int
    prompt_tokens_cost_usd: float | None
    completion_tokens: int
    completion_tokens_cost_usd: float | None
    total_tokens: int
    total_tokens_cost_usd: float | None


class BaseLLMInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_input_fields(self, context):
        prompt = context.context.get("prompt") or context.context.get("instance_prompt")
        if prompt:
            required_parameters = prompt.get_required_parameters()
            provided_parameters = set(self.model_dump().keys())

            if not required_parameters.issubset(provided_parameters):
                raise ValueError(
                    f"Error: Invalid parameters were provided. Expected: {required_parameters}. "
                    f"Got: {provided_parameters}"
                )
            return self

        raise ValueError("Error: Unable to run llm. Prompt was not provided.")


class BaseLLM(ConnectionNode):
    """Base class for all LLM nodes.

    Attributes:
        MODEL_PREFIX (ClassVar[str | None]): Optional model prefix.
        name (str | None): Name of the LLM node. Defaults to "LLM".
        model (str): Model to use for the LLM.
        prompt (Prompt | None): Prompt to use for the LLM.
        connection (BaseConnection): Connection to use for the LLM.
        group (Literal[NodeGroup.LLMS]): Group for the node. Defaults to NodeGroup.LLMS.
        temperature (float): Temperature for the LLM. Defaults to 0.1.
        max_tokens (int): Maximum number of tokens for the LLM. Defaults to 1000.
        stop (list[str]): List of tokens to stop at for the LLM.
        error_handling (ErrorHandling): Error handling config. Defaults to ErrorHandling(timeout_seconds=600).
        top_p (float | None): Value to consider tokens with top_p probability.
        seed (int | None): Seed for generating the same result for repeated requests.
        presence_penalty (float | None): Penalize new tokens based on their existence in the text.
        frequency_penalty (float | None): Penalize new tokens based on their frequency in the text.
        tool_choice (str | None): Value to control which function is called by the model.
        thinking_enabled (bool): Enables advanced reasoning if set to True.
        budget_tokens (int): Maximum number of tokens allocated for thinking.
        response_format (dict[str, Any]): JSON schema that specifies the structure of the llm's output
        tools list[Tool]: List of tools that llm can call.
    """

    MODEL_PREFIX: ClassVar[str | None] = None
    name: str | None = "LLM"
    model: str
    prompt: Prompt | None = None
    connection: BaseConnection
    group: Literal[NodeGroup.LLMS] = NodeGroup.LLMS
    temperature: float = 0.1
    max_tokens: int = 1000
    stop: list[str] | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    top_p: float | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tool_choice: str | None = None
    thinking_enabled: bool | None = None
    budget_tokens: int = 1024
    response_format: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    _completion: Callable = PrivateAttr()
    _stream_chunk_builder: Callable = PrivateAttr()
    input_schema: ClassVar[type[BaseLLMInputSchema]] = BaseLLMInputSchema

    @field_validator("model")
    @classmethod
    def set_model(cls, value: str | None) -> str:
        """Set the model with the appropriate prefix.

        Args:
            value (str | None): The model value.

        Returns:
            str: The model value with the prefix.
        """
        if cls.MODEL_PREFIX is not None and not value.startswith(cls.MODEL_PREFIX):
            value = f"{cls.MODEL_PREFIX}{value}"
        return value

    def __init__(self, **kwargs):
        """Initialize the BaseLLM instance.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        # Save a bit of loading time as litellm is slow
        from litellm import completion, stream_chunk_builder

        # Avoid the same imports multiple times and for future usage in execute
        self._completion = completion
        self._stream_chunk_builder = stream_chunk_builder

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        return {"instance_prompt": self.prompt}

    def get_messages(
        self,
        prompt,
        input_data,
    ) -> list[dict]:
        """
        Format and filter message parameters based on provider requirements.
        Override this in provider-specific subclasses.
        """
        messages = prompt.format_messages(**dict(input_data))
        return messages

    @classmethod
    def get_usage_data(
        cls,
        model: str,
        completion: "ModelResponse",
    ) -> BaseLLMUsageData:
        """Get usage data for the LLM.

        This method generates usage data for the LLM based on the provided messages.

        Args:
            model (str): The model to use for generating the usage data.
            completion (ModelResponse): The completion response from the LLM.

        Returns:
            BaseLLMUsageData: A model containing the usage data for the LLM.
        """
        from litellm import cost_per_token

        usage = completion.model_extra["usage"]
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        try:
            prompt_tokens_cost_usd, completion_tokens_cost_usd = cost_per_token(
                model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            total_tokens_cost_usd = prompt_tokens_cost_usd + completion_tokens_cost_usd
        except Exception:
            prompt_tokens_cost_usd, completion_tokens_cost_usd, total_tokens_cost_usd = None, None, None

        return BaseLLMUsageData(
            prompt_tokens=prompt_tokens,
            prompt_tokens_cost_usd=prompt_tokens_cost_usd,
            completion_tokens=completion_tokens,
            completion_tokens_cost_usd=completion_tokens_cost_usd,
            total_tokens=total_tokens,
            total_tokens_cost_usd=total_tokens_cost_usd,
        )

    def _handle_completion_response(
        self,
        response: Union["ModelResponse", "CustomStreamWrapper"],
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict:
        """Handle completion response.

        Args:
            response (ModelResponse | CustomStreamWrapper): The response from the LLM.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content and tool calls if present.
        """
        content = response.choices[0].message.content
        result = {"content": content}
        if tool_calls := response.choices[0].message.tool_calls:
            tool_calls_parsed = {}
            for tc in tool_calls:
                call = tc.model_dump()
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                tool_calls_parsed[call["function"]["name"]] = call
            result["tool_calls"] = tool_calls_parsed

        usage_data = self.get_usage_data(model=self.model, completion=response).model_dump()
        self.run_on_node_execute_run(callbacks=config.callbacks, usage_data=usage_data, **kwargs)

        return result

    def _handle_streaming_completion_response(
        self,
        response: Union["ModelResponse", "CustomStreamWrapper"],
        messages: list[dict],
        config: RunnableConfig = None,
        **kwargs,
    ):
        """Handle streaming completion response.

        Args:
            response (ModelResponse | CustomStreamWrapper): The response from the LLM.
            messages (list[dict]): The messages used for the LLM.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content and tool calls.
        """
        chunks = []
        for chunk in response:
            chunks.append(chunk)

            self.run_on_node_execute_stream(
                config.callbacks,
                chunk.model_dump(),
                **kwargs,
            )

        full_response = self._stream_chunk_builder(chunks=chunks, messages=messages)
        return self._handle_completion_response(response=full_response, config=config, **kwargs)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Updates or modifies the parameters for the completion method.

        This method can be overridden by subclasses to customize the parameters
        passed to the completion method. By default, it enables usage information
        in streaming mode if streaming is enabled and include_usage is set.
        Args:
            params (dict[str, Any]): The parameters to be updated.

        Returns:
            dict[str, Any]: The updated parameters.
        """
        if self.streaming and self.streaming.enabled and self.streaming.include_usage and params.get("stream", False):
            params.setdefault("stream_options", {})
            params["stream_options"]["include_usage"] = True
        return params

    def execute(
        self,
        input_data: BaseLLMInputSchema,
        config: RunnableConfig = None,
        prompt: Prompt | None = None,
        tools: list[Tool] | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Execute the LLM node.

        This method processes the input data, formats the prompt, and generates a response using
        the configured LLM.

        Args:
            input_data (BaseLLMInputSchema): The input data for the LLM.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            prompt (Prompt, optional): The prompt to use for this execution. Defaults to None.
            tools list[Tool]: List of tools that llm can call.
            response_format (dict[str, Any]): JSON schema that specifies the structure of the llm's output
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content and tool calls.
        """
        config = ensure_config(config)
        prompt = prompt or self.prompt or Prompt(messages=[], tools=None, response_format=None)
        messages = self.get_messages(prompt, input_data)
        base_tools = prompt.format_tools(**dict(input_data))
        self.run_on_node_execute_run(callbacks=config.callbacks, prompt_messages=messages, **kwargs)

        params = self.connection.conn_params.copy()
        if self.client and not isinstance(self.connection, HttpApiKey):
            params.update({"client": self.client})
        if self.thinking_enabled:
            params.update({"thinking": {"type": "enabled", "budget_tokens": self.budget_tokens}})

        response_format = self.response_format or response_format or prompt.response_format
        tools = self.tools or tools or base_tools

        common_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.streaming.enabled,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": tools,
            "tool_choice": self.tool_choice,
            "stop": self.stop if self.stop else None,
            "top_p": self.top_p,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": response_format,
            "drop_params": True,
            **params,
        }

        common_params = self.update_completion_params(common_params)

        response = self._completion(**common_params)

        handle_completion = (
            self._handle_streaming_completion_response if self.streaming.enabled else self._handle_completion_response
        )

        return handle_completion(
            response=response, messages=messages, config=config, input_data=dict(input_data), **kwargs
        )
