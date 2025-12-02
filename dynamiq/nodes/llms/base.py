import copy
import json
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Union

from litellm import get_max_tokens, supports_vision
from litellm.exceptions import (
    APIConnectionError,
    BudgetExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.utils import supports_pdf_input
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from dynamiq.callbacks.streaming import BaseStreamingCallbackHandler
from dynamiq.connections import BaseConnection, HttpApiKey
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.llm_tool import Tool
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from litellm import CustomStreamWrapper, ModelResponse


LLM_RATE_LIMIT_ERROR_INDICATORS = (
    "rate limit",
    "429",
    "quota exceeded",
    "too many requests",
    "ratelimit",
    "throttl",
    "capacity",
    "resource_exhausted",
)

LLM_CONNECTION_ERROR_INDICATORS = (
    "connection",
    "timeout",
    "timed out",
    "unreachable",
    "service unavailable",
    "503",
    "502",
    "504",
    "gateway",
    "network",
    "dns",
    "refused",
    "reset",
    "closed",
    "internal server error",
    "500",
)


class FallbackTrigger(str, Enum):
    ANY = "any"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"


class FallbackConfig(BaseModel):
    """Configuration for LLM fallback behavior.

    Attributes:
        llm: The fallback LLM to use when the primary LLM fails. Required when enabled=True.
        enabled: Whether fallback is enabled. Defaults to False.
        triggers: List of trigger conditions that will activate the fallback.
            Use FallbackTrigger.ANY to trigger on any error.

    Examples:
        # Single trigger
        FallbackConfig(llm=my_llm, enabled=True, triggers=[FallbackTrigger.RATE_LIMIT])

        # Multiple triggers
        FallbackConfig(llm=my_llm, enabled=True, triggers=[FallbackTrigger.RATE_LIMIT, FallbackTrigger.CONNECTION])
    """

    llm: "BaseLLM | None" = None
    enabled: bool = False
    triggers: list[FallbackTrigger] = Field(default_factory=lambda: [FallbackTrigger.ANY])

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_llm_required_when_enabled(self) -> "FallbackConfig":
        """Validate that llm is provided when fallback is enabled."""
        if self.enabled and self.llm is None:
            raise ValueError("FallbackConfig requires 'llm' when 'enabled' is True")
        return self


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
        temperature (float | None): Temperature for the LLM.
        max_tokens (int | None): Maximum number of tokens for the LLM.
        stop (list[str]): List of tokens to stop at for the LLM.
        error_handling (ErrorHandling): Error handling config. Defaults to ErrorHandling(timeout_seconds=600).
        top_p (float | None): Value to consider tokens with top_p probability.
        seed (int | None): Seed for generating the same result for repeated requests.
        presence_penalty (float | None): Penalize new tokens based on their existence in the text.
        frequency_penalty (float | None): Penalize new tokens based on their frequency in the text.
        tool_choice (str | None): Value to control which function is called by the model.
        thinking_enabled (bool): Enables advanced reasoning if set to True.
        budget_tokens (int): Maximum number of tokens allocated for thinking.
        response_format (dict[str, Any]): JSON schema that specifies the structure of the llm's output.
        tools (list[Tool]): List of tools that llm can call.
        fallback (FallbackConfig): Configuration for fallback behavior.
    """

    MODEL_PREFIX: ClassVar[str | None] = None
    name: str | None = "LLM"
    model: str
    prompt: Prompt | None = None
    connection: BaseConnection
    group: Literal[NodeGroup.LLMS] = NodeGroup.LLMS
    temperature: float | None = None
    max_tokens: int | None = None
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
    tools: list[Tool | dict] | None = None
    input_schema: ClassVar[type[BaseLLMInputSchema]] = BaseLLMInputSchema
    inference_mode: InferenceMode = Field(
        default=InferenceMode.DEFAULT,
        deprecated="Please use `tools` and `response_format` parameters "
        "for selecting between function calling and structured output.",
    )
    schema_: dict[str, Any] | type[BaseModel] | None = Field(
        None,
        description="Schema for structured output or function calling.",
        alias="schema",
        deprecated="Please use `tools` and `response_format` parameters "
        "for function calling and structured output respectively.",
    )
    fallback: FallbackConfig | None = Field(
        default=None,
        description="Configuration for fallback behavior including the fallback LLM.",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    _completion: Callable = PrivateAttr()
    _stream_chunk_builder: Callable = PrivateAttr()
    _is_fallback_run: bool = PrivateAttr(default=False)
    _json_schema_fields: ClassVar[list[str]] = ["model", "temperature", "max_tokens", "prompt"]

    @classmethod
    def _generate_json_schema(cls, models: list[str], **kwargs) -> dict[str, Any]:
        """
        Generates full json schema of BaseLLM Node.

        This schema is designed for compatibility with the WorkflowYamlParser,
        containing enough partial information to instantiate an BaseLLM.
        Parameters name to be included in the schema are either defined in the _json_schema_fields class variable or
        passed via the fields parameter.

        It generates a schema using provided models.

        Args:
            models (list[str]): List of available models.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Generated json schema.
        """
        schema = super()._generate_json_schema(**kwargs)
        schema["properties"]["model"]["enum"] = models
        return schema

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

    def init_components(self, connection_manager=None):
        """Initialize components including fallback LLM if configured.

        Args:
            connection_manager: The connection manager for initializing connections.
        """
        super().init_components(connection_manager)
        if self.fallback and self.fallback.llm and self.fallback.llm.is_postponed_component_init:
            self.fallback.llm.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self) -> dict:
        """Exclude fallback configuration during serialization."""
        return super().to_dict_exclude_params | {"fallback": True}

    def to_dict(self, **kwargs) -> dict:
        """Convert to dictionary representation."""
        data = super().to_dict(**kwargs)
        if self.fallback:
            data["fallback"] = self.fallback.model_dump(exclude={"llm": True})
            data["fallback"]["llm"] = self.fallback.llm.to_dict(**kwargs) if self.fallback.llm else None
        if self._is_fallback_run:
            data["is_fallback"] = True
        return data

    def reset_run_state(self):
        """Reset the run state of the LLM."""
        self._is_fallback_run = False

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        return {"instance_prompt": self.prompt}

    def get_token_limit(self) -> int:
        """Returns token limits of a llm.

        Returns:
            int: Number of tokens.
        """
        return get_max_tokens(self.model)

    @property
    def is_vision_supported(self) -> bool:
        """Check if the LLM supports vision/image processing."""
        return supports_vision(self.model)

    @property
    def is_pdf_input_supported(self) -> bool:
        """Check if the LLM supports PDF input."""
        return supports_pdf_input(self.model)

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

    def _get_response_format_and_tools(
        self,
        prompt: Prompt | None = None,
        tools: list[Tool | dict] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Get response format and tools
        Args:
            input_data (BaseLLMInputSchema): The input data for the LLM.
            prompt (Prompt | None): The prompt to use.
            tools (list[Tool] | None): The tools to use.
            response_format (dict[str, Any] | None): The response format to use.
        Returns:
            tuple[dict[str, Any] | None, dict[str, Any] | None]: Response format and tools.
        Raises:
            ValueError: If schema is None when using STRUCTURED_OUTPUT or FUNCTION_CALLING modes.
        """
        response_format = response_format or self.response_format or prompt.response_format
        tools = tools or self.tools or prompt.tools

        # Suppress DeprecationWarning if deprecated parameters are not set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            use_inference_mode = (not response_format or not tools) and self.inference_mode != InferenceMode.DEFAULT

        if use_inference_mode:
            schema = self.schema_
            match self.inference_mode:
                case InferenceMode.STRUCTURED_OUTPUT:
                    if schema is None:
                        raise ValueError("Schema must be provided when using STRUCTURED_OUTPUT inference mode")
                    response_format = response_format or schema
                case InferenceMode.FUNCTION_CALLING:
                    if schema is None:
                        raise ValueError("Schema must be provided when using FUNCTION_CALLING inference mode")
                    tools = tools or schema

        if tools:
            tools = [tool.model_dump() if isinstance(tool, Tool) else tool for tool in tools]

        return response_format, tools

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
        tools: list[Tool | dict] | None = None,
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
            tools (list[Tool|dict]): List of tools that llm can call.
            response_format (dict[str, Any]): JSON schema that specifies the structure of the llm's output
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the generated content and tool calls.
        """
        config = ensure_config(config)
        self.reset_run_state()
        prompt = prompt or self.prompt or Prompt(messages=[], tools=None, response_format=None)
        messages = self.get_messages(prompt, input_data)
        self.run_on_node_execute_run(callbacks=config.callbacks, prompt_messages=messages, **kwargs)

        extra = copy.deepcopy(self.__pydantic_extra__)
        params = self.connection.conn_params.copy()
        if self.client and not isinstance(self.connection, HttpApiKey):
            params.update({"client": self.client})
        if self.thinking_enabled:
            params.update({"thinking": {"type": "enabled", "budget_tokens": self.budget_tokens}})
        if extra:
            params.update(extra)

        response_format, tools = self._get_response_format_and_tools(
            prompt=prompt,
            tools=tools,
            response_format=response_format,
        )
        # Check if a streaming callback is available in the config and enable streaming only if it is
        # This is to avoid unnecessary streaming to reduce CPU usage
        is_streaming_callback_available = any(
            isinstance(callback, BaseStreamingCallbackHandler) for callback in config.callbacks
        )
        common_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": self.streaming.enabled and is_streaming_callback_available,
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
            self._handle_streaming_completion_response
            if self.streaming.enabled and is_streaming_callback_available
            else self._handle_completion_response
        )

        return handle_completion(
            response=response, messages=messages, config=config, input_data=dict(input_data), **kwargs
        )

    def _is_rate_limit_error(self, exception_type: type[Exception], error_str: str) -> bool:
        """Check if the error is a rate limit error.

        Args:
            exception_type: The type of exception.
            error_str: Lowercase error message string.

        Returns:
            bool: True if it's a rate limit error.
        """
        if issubclass(exception_type, (RateLimitError, BudgetExceededError)):
            return True
        return any(indicator in error_str for indicator in LLM_RATE_LIMIT_ERROR_INDICATORS)

    def _is_connection_error(self, exception_type: type[Exception], error_str: str) -> bool:
        """Check if the error is a connection error.

        Args:
            exception_type: The type of exception.
            error_str: Lowercase error message string.

        Returns:
            bool: True if it's a connection error.
        """
        if issubclass(exception_type, (APIConnectionError, Timeout, ServiceUnavailableError, InternalServerError)):
            return True
        if issubclass(exception_type, (ConnectionError, TimeoutError, OSError)):
            return True
        return any(indicator in error_str for indicator in LLM_CONNECTION_ERROR_INDICATORS)

    def _should_trigger_fallback(self, exception_type: type[Exception], exception_message: str | None = None) -> bool:
        """Determine if exception should trigger fallback to secondary LLM.

        Args:
            exception_type: The type of exception that caused the primary LLM to fail.
            exception_message: The exception message string for string-based detection.

        Returns:
            bool: True if fallback should be triggered, False otherwise.
        """
        if not self.fallback or not self.fallback.enabled or not self.fallback.llm:
            return False

        triggers = set(self.fallback.triggers)
        if FallbackTrigger.ANY in triggers:
            return True

        error_str = (exception_message or "").lower()

        if FallbackTrigger.RATE_LIMIT in triggers and self._is_rate_limit_error(exception_type, error_str):
            return True
        if FallbackTrigger.CONNECTION in triggers and self._is_connection_error(exception_type, error_str):
            return True

        return False

    def run_sync(
        self,
        input_data: dict,
        config: RunnableConfig = None,
        depends_result: dict = None,
        **kwargs,
    ) -> RunnableResult:
        """Run the LLM with fallback support.

        If the primary LLM fails and a fallback is configured, the primary failure
        is traced first, then the fallback LLM is executed separately.

        The fallback receives the same transformed input that the primary received,
        and the primary's output_transformer is applied to the fallback's output.

        Args:
            input_data: Input data for the LLM.
            config: Configuration for the run.
            depends_result: Results of dependent nodes.
            **kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Result of the LLM execution.
        """
        result = super().run_sync(input_data=input_data, config=config, depends_result=depends_result, **kwargs)

        if result.status != RunnableStatus.FAILURE:
            return result

        if not self.fallback or not self.fallback.llm:
            return result

        if not result.error:
            return result

        if not self._should_trigger_fallback(result.error.type, result.error.message):
            return result

        fallback_llm = self.fallback.llm
        fallback_llm._is_fallback_run = True
        logger.warning(
            f"LLM {self.name} - {self.id}: Primary LLM ({self.model}) failed. "
            f"Error: {result.error.type.__name__}: {result.error.message}. "
            f"Attempting fallback to {fallback_llm.name} - {fallback_llm.id}"
        )

        # Use the primary's already transformed input for fallback
        # This ensures fallback works with the same prepared input as primary
        fallback_kwargs = {k: v for k, v in kwargs.items() if k != "run_depends"}
        fallback_kwargs["parent_run_id"] = kwargs.get("parent_run_id")

        fallback_input = result.input.model_dump() if hasattr(result.input, "model_dump") else result.input
        fallback_result = fallback_llm.run_sync(
            input_data=fallback_input,
            config=config,
            depends_result=None,  # Input is already transformed, no need to merge depends
            **fallback_kwargs,
        )

        if fallback_result.status == RunnableStatus.SUCCESS:
            logger.info(f"LLM {self.name} - {self.id}: Fallback LLM ({fallback_llm.model}) succeeded")
            # Apply primary node's output_transformer to fallback result
            transformed_output = self.transform_output(fallback_result.output, config=config, **kwargs)
            return RunnableResult(
                status=RunnableStatus.SUCCESS,
                input=result.input,
                output=transformed_output,
            )

        logger.error(f"LLM {self.name} - {self.id}: Fallback LLM ({fallback_llm.model}) failed.")
        return result
