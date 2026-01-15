import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping

from litellm import get_supported_openai_params, supports_function_calling
from pydantic import Field, model_validator

from dynamiq.callbacks import AgentStreamingParserCallback, StreamingQueueCallbackHandler
from dynamiq.nodes.agents.base import Agent as BaseAgent
from dynamiq.nodes.agents.components import parser, schema_generator
from dynamiq.nodes.agents.components.history_manager import HistoryManagerMixin
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    AgentUnknownToolException,
    JSONParsingError,
    MaxLoopsExceededException,
    ParsingError,
    RecoverableAgentException,
    TagNotFoundError,
    XMLParsingError,
)
from dynamiq.nodes.agents.utils import SummarizationConfig, ToolCacheEntry, XMLParser
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools import ContextManagerTool
from dynamiq.nodes.tools.context_manager import _apply_context_manager_tool_effect
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.prompts import Message, MessageRole, VisionMessage
from dynamiq.runnables import RunnableConfig
from dynamiq.types.llm_tool import Tool
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger

final_answer_function_schema = {
    "type": "function",
    "strict": True,
    "function": {
        "name": "provide_final_answer",
        "description": "Function should be called when if you can answer the initial request"
        " or if there is not request at all.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about why you can answer original question.",
                },
                "answer": {"type": "string", "description": "Answer on initial request."},
            },
            "required": ["thought", "answer"],
        },
    },
}

TYPE_MAPPING = {
    int: "integer",
    float: "float",
    bool: "boolean",
    str: "string",
    dict: "object",
}

UNKNOWN_TOOL_NAME = "unknown_tool"


class Agent(HistoryManagerMixin, BaseAgent):
    """Unified Agent that uses a ReAct-style strategy for processing tasks by interacting with tools in a loop."""

    name: str = "Agent"
    max_loops: int = Field(default=15, ge=2)
    inference_mode: InferenceMode = Field(default=InferenceMode.DEFAULT)
    behaviour_on_max_loops: Behavior = Field(
        default=Behavior.RAISE,
        description="Define behavior when max loops are exceeded. Options are 'raise' or 'return'.",
    )
    parallel_tool_calls_enabled: bool = Field(
        default=False,
        description="Enable multi-tool execution in a single step. "
        "When True, the agent can call multiple tools in parallel.",
    )
    direct_tool_output_enabled: bool = Field(
        default=False,
        description="Enable direct tool output capability. "
        "When True, the agent can return raw tool outputs directly without summarization.",
    )

    format_schema: list = Field(default_factory=list)
    summarization_config: SummarizationConfig = Field(default_factory=SummarizationConfig)

    _tools: list[Tool] = []
    _response_format: dict[str, Any] | None = None

    def get_clone_attr_initializers(self) -> dict[str, Callable[[Node], Any]]:
        """
        Define attribute initializers for cloning.

        Ensures that cloned agents get fresh instances of:
        - _tool_cache: Independent tool execution cache

        Returns:
            Dictionary mapping attribute names to initializer functions
        """
        base = super().get_clone_attr_initializers()
        base.update(
            {
                "_tool_cache": lambda _: {},
            }
        )
        return base

    def log_reasoning(self, thought: str, action: str, action_input: str, loop_num: int) -> None:
        """
        Logs reasoning step of agent.

        Args:
            thought (str): Reasoning about next step.
            action (str): Chosen action.
            action_input (str): Input to the tool chosen by action.
            loop_num (int): Number of reasoning loop.
        """
        logger.info(
            "\n------------------------------------------\n"
            f"Agent {self.name}: Loop {loop_num}:\n"
            f"Thought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {action_input}"
            "\n------------------------------------------"
        )

    def log_final_output(self, thought: str, final_output: str, loop_num: int) -> None:
        """
        Logs final output of the agent.

        Args:
            final_output (str): Final output of agent.
            loop_num (int): Number of reasoning loop
        """
        logger.info(
            "\n------------------------------------------\n"
            f"Agent {self.name}: Loop {loop_num}\n"
            f"Thought: {thought}\n"
            f"Final answer: {final_output}"
            "\n------------------------------------------\n"
        )

    def _should_delegate_final(
        self,
        tool: Node | None,
        action_input: Any,
    ) -> bool:
        """Only Agent tools with per-call delegate_final flag can delegate."""
        if not self.delegation_allowed:
            return False

        if not isinstance(tool, Agent):
            return False

        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except json.JSONDecodeError:
                return False

        if isinstance(action_input, Mapping):
            return bool(action_input.get("delegate_final"))

        return False

    @model_validator(mode="after")
    def validate_inference_mode(self):
        """Validate whether specified model can be inferenced in provided mode."""
        match self.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                if not supports_function_calling(model=self.llm.model):
                    raise ValueError(f"Model {self.llm.model} does not support function calling")

            case InferenceMode.STRUCTURED_OUTPUT:
                params = get_supported_openai_params(model=self.llm.model)
                if "response_format" not in params:
                    raise ValueError(f"Model {self.llm.model} does not support structured output")

        return self

    @model_validator(mode="after")
    def _ensure_context_manager_tool(self):
        """Automatically add ContextManagerTool when summarization is enabled."""
        try:
            if self.summarization_config.enabled:
                has_context_tool = any(isinstance(t, ContextManagerTool) for t in self.tools)
                if not has_context_tool:
                    # Add with a stable name for addressing from the agent
                    self.tools.append(ContextManagerTool(llm=self.llm, name="context-manager"))
        except Exception as e:
            logger.error(f"Failed to ensure ContextManagerTool: {e}")
        return self

    def _append_recovery_instruction(
        self,
        *,
        error_label: str,
        error_detail: str,
        llm_generated_output: str | None,
        extra_guidance: str | None = None,
    ) -> None:
        """Append a correction instruction to prompt for recoverable agent errors."""

        error_context = llm_generated_output if llm_generated_output else "No response generated"

        self._prompt.messages.append(
            Message(role=MessageRole.ASSISTANT, content=f"Previous response:\n{error_context}", static=True)
        )

        guidance_suffix = f" {extra_guidance.strip()}" if extra_guidance else ""

        correction_message = (
            "Correction Instruction: The previous response could not be parsed due to the "
            f"following error: '{error_label}: {error_detail}'. Please regenerate the response "
            "strictly following the required format, ensuring all tags or labeled sections are "
            "present and correctly structured, and that any JSON content is valid." + guidance_suffix
        )

        self._prompt.messages.append(
            Message(role=MessageRole.USER, content=correction_message, static=True)
        )

    def stream_reasoning(self, content: dict[str, Any], config: RunnableConfig, **kwargs) -> None:
        """
        Streams intermediate reasoning of the Agent.

        Args:
            content (dict[str, Any]): Content that will be sent.
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.
        """
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            self.stream_content(
                content=content,
                source=self.name,
                step="reasoning",
                config=config,
                **kwargs,
            )

    def _append_assistant_message(self, llm_result: Any, llm_generated_output: str) -> None:
        """
        Appends the assistant's message to conversation history based on inference mode.

        Args:
            llm_result: The full LLM result object (needed for function calling mode).
            llm_generated_output: The generated text output from the LLM.
        """
        if self.inference_mode == InferenceMode.FUNCTION_CALLING:
            # For function calling, construct a message that includes the tool call
            if "tool_calls" in dict(llm_result.output):
                try:
                    tool_call = list(llm_result.output["tool_calls"].values())[0]
                    function_name = tool_call["function"]["name"]
                    function_args = json.dumps(tool_call["function"]["arguments"])
                    message_content = f"Function call: {function_name}({function_args})"
                    self._prompt.messages.append(
                        Message(role=MessageRole.ASSISTANT, content=message_content, static=True)
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract tool call from LLM result: {e}. Using raw output instead.")
                    self._prompt.messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=llm_generated_output or "Cannot extract tool call from LLM result.",
                            static=True,
                        )
                    )
        elif llm_generated_output:
            # For other modes, use the generated text output
            self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=llm_generated_output, static=True))

    def _handle_default_mode(
        self, llm_generated_output: str, loop_num: int
    ) -> tuple[str | None, str | None, dict | list | None]:
        """Handle DEFAULT inference mode parsing."""
        if not llm_generated_output or not llm_generated_output.strip():
            self._append_recovery_instruction(
                error_label="EmptyResponse",
                error_detail="The model returned an empty reply while using the Thought/Action format.",
                llm_generated_output=llm_generated_output,
                extra_guidance=(
                    "Re-evaluate the latest observation and respond with 'Thought:' followed by either "
                    "an 'Action:' plus JSON 'Action Input:' or a final 'Answer:' section."
                ),
            )
            return None, None, None

        if "Answer:" in llm_generated_output:
            thought, final_answer = parser.extract_default_final_answer(llm_generated_output)
            self.log_final_output(thought, final_answer, loop_num)
            return thought, "final_answer", final_answer

        thought, action, action_input = parser.parse_default_action(
            llm_generated_output, self.parallel_tool_calls_enabled
        )
        self.log_reasoning(thought, action, action_input, loop_num)
        return thought, action, action_input

    def _handle_function_calling_mode(
        self, llm_result: Any, loop_num: int
    ) -> tuple[str | None, str | None, dict | list | None] | tuple[str, str, str]:
        """Handle FUNCTION_CALLING inference mode parsing.

        Returns:
            tuple: (thought, action, action_input) for normal actions
                   (thought, "final_answer", final_answer) for final answers
        """
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: using function calling inference mode")

        if "tool_calls" not in dict(llm_result.output):
            logger.error("Error: No function called.")
            raise ActionParsingException("Error: No function called, you need to call the correct function.")

        action = list(llm_result.output["tool_calls"].values())[0]["function"]["name"].strip()
        llm_generated_output_json = list(llm_result.output["tool_calls"].values())[0]["function"]["arguments"]

        thought = llm_generated_output_json["thought"]
        if action == "provide_final_answer":
            final_answer = llm_generated_output_json["answer"]
            self.log_final_output(thought, final_answer, loop_num)
            return thought, "final_answer", final_answer

        action_input = llm_generated_output_json["action_input"]

        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input)
            except json.JSONDecodeError as e:
                raise ActionParsingException(f"Error parsing action_input string. {e}", recoverable=True)

        self.log_reasoning(thought, action, action_input, loop_num)
        return thought, action, action_input

    def _handle_structured_output_mode(
        self, llm_generated_output: str | dict, loop_num: int
    ) -> tuple[str | None, str | None, dict | list | None] | tuple[str, str, str]:
        """Handle STRUCTURED_OUTPUT inference mode parsing.

        Returns:
            tuple: (thought, action, action_input) for normal actions
                   (thought, "final_answer", final_answer) for final answers
        """
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: using structured output inference mode")

        try:
            if isinstance(llm_generated_output, str):
                llm_generated_output_json = json.loads(llm_generated_output)
            else:
                llm_generated_output_json = llm_generated_output
        except json.JSONDecodeError as e:
            raise ActionParsingException(f"Error parsing action. {e}", recoverable=True)

        if "action" not in llm_generated_output_json or "thought" not in llm_generated_output_json:
            raise ActionParsingException("No action or thought provided.", recoverable=True)

        thought = llm_generated_output_json["thought"]
        action = llm_generated_output_json["action"]
        action_input = llm_generated_output_json["action_input"]

        if action == "finish":
            self.log_final_output(thought, action_input, loop_num)
            return thought, "final_answer", action_input

        try:
            if isinstance(action_input, str):
                action_input = json.loads(action_input)
        except json.JSONDecodeError as e:
            raise ActionParsingException(f"Error parsing action_input string. {e}", recoverable=True)

        self.log_reasoning(thought, action, action_input, loop_num)
        return thought, action, action_input

    def _handle_xml_mode(
        self, llm_generated_output: str, loop_num: int, config: RunnableConfig, **kwargs
    ) -> tuple[str | None, str | None, dict | list | None]:
        """Handle XML inference mode parsing."""
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: using XML inference mode")

        if not llm_generated_output or not llm_generated_output.strip():
            self._append_recovery_instruction(
                error_label="EmptyResponse",
                error_detail="The model returned an empty reply while XML format was required.",
                llm_generated_output=llm_generated_output,
                extra_guidance=(
                    "Respond with <thought>...</thought> and "
                    "either <action>/<action_input> or <answer> tags, "
                    "making sure to address the latest observation."
                ),
            )
            return None, None, None

        if self.parallel_tool_calls_enabled:
            try:
                parsed_result = XMLParser.parse_unified_xml_format(llm_generated_output)

                thought = parsed_result.get("thought", "")

                if parsed_result.get("is_final", False):
                    final_answer = parsed_result.get("answer", "")
                    self.log_final_output(thought, final_answer, loop_num)
                    return thought, "final_answer", final_answer

                tools_data = parsed_result.get("tools", [])
                action = tools_data

                if len(tools_data) > 1:
                    for tool_payload in tools_data:
                        if isinstance(tool_payload.get("input"), dict) and tool_payload["input"].get("delegate_final"):
                            raise ActionParsingException(
                                "delegate_final is only supported for single agent tool calls.",
                                recoverable=True,
                            )

                if len(tools_data) == 1:
                    self.log_reasoning(
                        thought,
                        tools_data[0].get("name", "unknown_tool"),
                        tools_data[0].get("input", {}),
                        loop_num,
                    )
                else:
                    self.log_reasoning(thought, "multiple_tools", str(tools_data), loop_num)

                tools_data_for_streaming = [
                    {
                        "name": tool.get("name", ""),
                        "type": self.tool_by_names.get(tool.get("name", "")).type,
                    }
                    for tool in tools_data
                    if tool.get("name", "") and self.tool_by_names.get(tool.get("name", ""))
                ]

                self.stream_reasoning(
                    {
                        "thought": thought,
                        "tools": tools_data_for_streaming,
                        "loop_num": loop_num,
                    },
                    config,
                    **kwargs,
                )
                return thought, action, tools_data

            except (XMLParsingError, TagNotFoundError, JSONParsingError) as e:
                self._append_recovery_instruction(
                    error_label=type(e).__name__,
                    error_detail=str(e),
                    llm_generated_output=llm_generated_output,
                    extra_guidance=(
                        "Return <thought> with the resolved plan and list tool calls inside <tools>, "
                        "or mark the run as final with <answer>."
                    ),
                )
                return None, None, None
        else:
            try:
                parsed_data = XMLParser.parse(
                    llm_generated_output, required_tags=["thought", "answer"], optional_tags=["output"]
                )
                thought = parsed_data.get("thought")
                final_answer = parsed_data.get("answer")
                self.log_final_output(thought, final_answer, loop_num)
                return thought, "final_answer", final_answer

            except TagNotFoundError:
                logger.debug("XMLParser: Not a final answer structure, trying action structure.")
                try:
                    parsed_data = XMLParser.parse(
                        llm_generated_output,
                        required_tags=["thought", "action", "action_input"],
                        optional_tags=["output"],
                        json_fields=["action_input"],
                    )
                    thought = parsed_data.get("thought")
                    action = parsed_data.get("action")
                    action_input = parsed_data.get("action_input")
                    self.log_reasoning(thought, action, action_input, loop_num)
                    return thought, action, action_input
                except ParsingError as e:
                    logger.error(f"XMLParser: Empty or invalid XML response for action parsing: {e}")
                    raise ActionParsingException(
                        "The previous response was empty or invalid. "
                        "Provide <thought> with either <action>/<action_input> or <answer>.",
                        recoverable=True,
                    )

            except ParsingError as e:
                logger.error(f"XMLParser: Empty or invalid XML response: {e}")
                raise ActionParsingException(
                    "The previous response was empty or invalid. " "Please provide the required XML tags.",
                    recoverable=True,
                )

    def _setup_prompt_and_stop_sequences(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
    ) -> int:
        """Setup the prompt with system message, history, and configure stop sequences.

        Args:
            input_message: The user's input message
            history_messages: Optional conversation history

        Returns:
            int: The summary offset (position where history starts)
        """
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.generate_prompt(
                tools_name=self.tool_names,
                input_formats=schema_generator.generate_input_formats(self.tools, self.sanitize_tool_name),
                **self.system_prompt_manager.build_delegation_variables(self.delegation_allowed),
            ),
            static=True,
        )

        if history_messages:
            self._prompt.messages = [system_message, *history_messages, input_message]
        else:
            self._prompt.messages = [system_message, input_message]

        summary_offset = self._history_offset = len(self._prompt.messages)

        # Configure stop sequences based on inference mode
        stop_sequences = []
        if self.inference_mode == InferenceMode.DEFAULT:
            stop_sequences.extend(["Observation: ", "\nObservation:"])
        elif self.inference_mode == InferenceMode.XML:
            stop_sequences.extend(
                [
                    "\nObservation:",
                    "Observation:",
                    "</output>\n<",
                    "</output><",
                ]
            )
        self.llm.stop = stop_sequences

        return summary_offset

    def _setup_streaming_callback(
        self, config: RunnableConfig, loop_num: int, **kwargs
    ) -> tuple[AgentStreamingParserCallback | None, RunnableConfig, bool]:
        """Setup streaming callback and modify LLM config if agent streaming is enabled.

        Args:
            config: The runnable configuration
            loop_num: Current loop iteration number
            **kwargs: Additional parameters

        Returns:
            tuple: (streaming_callback, modified_config, original_streaming_enabled)
        """
        streaming_callback = None
        original_streaming_enabled = self.llm.streaming.enabled

        if self.streaming.enabled:
            streaming_callback = AgentStreamingParserCallback(
                agent=self,
                config=config,
                loop_num=loop_num,
                **kwargs,
            )

            if not original_streaming_enabled:
                self.llm.streaming.enabled = True

            llm_config = config.model_copy(deep=False)
            llm_config.callbacks = [
                callback for callback in llm_config.callbacks if not isinstance(callback, StreamingQueueCallbackHandler)
            ]
            llm_config.callbacks.append(streaming_callback)
        else:
            llm_config = config

        return streaming_callback, llm_config, original_streaming_enabled

    def _execute_single_tool(
        self, action: str, action_input: Any, thought: str, loop_num: int, config: RunnableConfig, **kwargs
    ) -> tuple[Any, dict, Any] | tuple[str, Any, Any]:
        """Execute a single tool with caching support."""
        tool = self.tool_by_names.get(self.sanitize_tool_name(action))
        if not tool:
            raise AgentUnknownToolException(
                f"Unknown tool: {action}. Use only available tools and provide only the tool's name in the "
                "action field. Do not include any additional reasoning. "
                "Please correct the action field or state that you cannot answer the question."
            )

        self.stream_reasoning(
            {
                "thought": thought,
                "action": action,
                "tool": {"name": tool.name, "type": tool.type},
                "action_input": action_input,
                "loop_num": loop_num,
            },
            config,
            **kwargs,
        )

        tool_cache_entry = ToolCacheEntry(action=action, action_input=action_input)
        tool_result = self._tool_cache.get(tool_cache_entry, None)
        delegate_final = self._should_delegate_final(tool, action_input)

        if not tool_result:
            tool_result, tool_files = self._run_tool(
                tool, action_input, config, delegate_final=delegate_final, **kwargs
            )
        else:
            logger.info(f"Agent {self.name} - {self.id}: Cached output of {action} found.")
            tool_files = {}

        if delegate_final:
            self.log_final_output(thought, tool_result, loop_num)
            if self.streaming.enabled:
                self.stream_content(
                    content=tool_result,
                    source=tool.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return "DELEGATED", tool_result, tool

        return tool_result, tool_files, tool

    def _add_observation_and_stream(
        self,
        tool_result: Any,
        tool_files: dict,
        tool: Any,
        action: Any,
        action_input: Any,
        config: RunnableConfig,
        **kwargs,
    ) -> None:
        """Add observation to prompt and stream tool result if enabled."""
        observation = f"\nObservation: {tool_result}\n"
        self._prompt.messages.append(Message(role=MessageRole.USER, content=observation, static=True))

        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            if tool is not None:
                source_name = tool_name = tool.name
            elif isinstance(action, list):
                tool_names = [
                    tool_data["name"] if isinstance(tool_data, dict) and "name" in tool_data else UNKNOWN_TOOL_NAME
                    for tool_data in action
                ]

                if len(tool_names) == 1:
                    source_name = tool_name = tool_names[0]
                else:
                    unique_tools = list(set(tool_names))
                    if len(unique_tools) == 1:
                        source_name = tool_name = f"{unique_tools[0]} (parallel)"
                    else:
                        source_name = tool_name = " + ".join(unique_tools)
            else:
                source_name = tool_name = str(action)

            self.stream_content(
                content={
                    "name": tool_name,
                    "input": action_input,
                    "result": tool_result,
                    "files": tool_files,
                },
                source=source_name,
                step="tool",
                config=config,
                **kwargs,
            )

    def _execute_tools_and_update_prompt(
        self,
        action: str | None,
        action_input: Any,
        thought: str | None,
        loop_num: int,
        config: RunnableConfig,
        **kwargs,
    ) -> str | None:
        """Execute tools based on action and update prompt with observations.

        Args:
            action: The action/tool name to execute
            action_input: Input parameters for the tool
            thought: The agent's reasoning
            loop_num: Current loop iteration number
            config: Runnable configuration
            **kwargs: Additional parameters

        Returns:
            str | None: Final answer if delegation occurred, None to continue loop
        """
        if action and self.tools:
            tool_result = None
            tool_files: Any = []
            tool = None

            try:
                # Handle XML parallel mode
                if self.inference_mode == InferenceMode.XML and self.parallel_tool_calls_enabled:
                    tools_data = action_input if isinstance(action_input, list) else [action_input]
                    execution_output = self._execute_tools(tools_data, config, **kwargs)
                    tool_result, tool_files = self._separate_tool_result_and_files(execution_output)
                else:
                    result = self._execute_single_tool(action, action_input, thought, loop_num, config, **kwargs)
                    if isinstance(result, tuple) and len(result) == 3 and result[0] == "DELEGATED":
                        return result[1]
                    tool_result, tool_files, tool = result

            except RecoverableAgentException as e:
                tool_result = f"{type(e).__name__}: {e}"

            # Apply context manager effects if needed
            if isinstance(tool, ContextManagerTool):
                _apply_context_manager_tool_effect(self._prompt, tool_result, self._history_offset)

            # Add observation and stream result
            self._add_observation_and_stream(tool_result, tool_files, tool, action, action_input, config, **kwargs)

        else:
            # No action or no tools available
            self.stream_reasoning(
                {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "loop_num": loop_num,
                },
                config,
                **kwargs,
            )

        return None

    def _run_agent(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """
        Executes the ReAct strategy by iterating through thought, action, and observation cycles.
        Args:
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.
        Returns:
            str: Final answer provided by the agent.
        Raises:
            RuntimeError: If the maximum number of loops is reached without finding a final answer.
            Exception: If an error occurs during execution.
        """
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: Running ReAct strategy")

        summary_offset = self._setup_prompt_and_stop_sequences(input_message, history_messages)

        for loop_num in range(1, self.max_loops + 1):
            try:
                streaming_callback, llm_config, original_streaming_enabled = self._setup_streaming_callback(
                    config, loop_num, **kwargs
                )

                try:
                    llm_result = self._run_llm(
                        messages=self._prompt.messages,
                        tools=self._tools,
                        response_format=self._response_format,
                        config=llm_config,
                        **kwargs,
                    )
                finally:
                    if not original_streaming_enabled:
                        try:
                            self.llm.streaming.enabled = original_streaming_enabled
                        except Exception:
                            logger.error("Failed to restore llm.streaming.enabled state")

                action, action_input = None, None
                llm_generated_output = ""

                if streaming_callback and streaming_callback.accumulated_content:
                    llm_generated_output = streaming_callback.accumulated_content
                else:
                    llm_generated_output = llm_result.output.get("content", "")

                llm_reasoning = (
                    llm_generated_output[:200]
                    if llm_generated_output
                    else str(llm_result.output.get("tool_calls", ""))[:200]
                )
                logger.info(f"Agent {self.name} - {self.id}: Loop {loop_num}, " f"reasoning:\n{llm_reasoning}...")

                # Append assistant message to conversation history BEFORE parsing
                # This ensures the LLM can see its own output during error recovery
                self._append_assistant_message(llm_result, llm_generated_output)

                # Parse LLM output based on inference mode
                match self.inference_mode:
                    case InferenceMode.DEFAULT:
                        result = self._handle_default_mode(llm_generated_output, loop_num)
                    case InferenceMode.FUNCTION_CALLING:
                        result = self._handle_function_calling_mode(llm_result, loop_num)
                    case InferenceMode.STRUCTURED_OUTPUT:
                        result = self._handle_structured_output_mode(llm_generated_output, loop_num)
                    case InferenceMode.XML:
                        result = self._handle_xml_mode(llm_generated_output, loop_num, config, **kwargs)
                # Handle final answer
                if result[1] == "final_answer":
                    return result[2]

                # Handle recovery (for modes that support it)
                # Check if both thought and action are None, which indicates (None, None, None) recovery
                if result[0] is None and result[1] is None:
                    continue

                thought, action, action_input = result

                final_answer = self._execute_tools_and_update_prompt(
                    action, action_input, thought, loop_num, config, **kwargs
                )

                if final_answer is not None:
                    return final_answer

            except ActionParsingException as e:
                extra_guidance = None
                if self.inference_mode == InferenceMode.XML:
                    extra_guidance = (
                        "Ensure the reply contains <thought> along "
                        "with either <action>/<action_input> or a final "
                        "<answer> tag."
                    )
                elif self.inference_mode == InferenceMode.DEFAULT:
                    extra_guidance = (
                        "Provide 'Thought:' and either 'Action:' "
                        "with a JSON 'Action Input:' or a final 'Answer:' section."
                    )

                self._append_recovery_instruction(
                    error_label=type(e).__name__,
                    error_detail=str(e),
                    llm_generated_output=llm_generated_output,
                    extra_guidance=extra_guidance,
                )
                continue

            summary_offset = self._try_summarize_history(
                input_message=input_message, summary_offset=summary_offset, config=config, **kwargs
            )

        if self.behaviour_on_max_loops == Behavior.RAISE:
            error_message = (
                f"Agent {self.name} (ID: {self.id}) "
                f"has reached the maximum loop limit of {self.max_loops} "
                f"without finding a final answer. "
                f"Last response: {self._prompt.messages[-1].content}\n"
                f"Consider increasing the maximum number of loops or "
                f"reviewing the task complexity to ensure completion."
            )
            raise MaxLoopsExceededException(message=error_message)
        else:
            max_loop_final_answer = self._handle_max_loops_exceeded(input_message, config, **kwargs)
            if self.streaming.enabled:
                self.stream_content(
                    content=max_loop_final_answer,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return max_loop_final_answer

    def _try_summarize_history(
        self,
        input_message: Message | VisionMessage,
        summary_offset: int,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> int:
        """
        Check if summarization is needed and perform it if token limit is exceeded.

        Args:
            input_message: User request message
            summary_offset: Current summary offset
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent

        Returns:
            int: Updated summary offset after summarization (or unchanged if not needed)
        """
        if not self.summarization_config.enabled:
            return summary_offset

        if self.is_token_limit_exceeded():
            return self.summarize_history(
                input_message=input_message,
                summary_offset=summary_offset,
                config=config,
                **kwargs,
            )

        return summary_offset

    def _handle_max_loops_exceeded(
        self, input_message: Message | VisionMessage, config: RunnableConfig | None = None, **kwargs
    ) -> str:
        """
        Handle the case where max loops are exceeded by crafting a thoughtful response.
        Uses XMLParser to extract the final answer from the LLM's last attempt.

        Args:
            input_message (Message | VisionMessage): Initial user message.
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.

        Returns:
            str: Final answer provided by the agent.
        """
        # Use model-specific max loops prompt from prompt manager
        max_loops_prompt = self.system_prompt_manager.max_loops_prompt

        system_message = Message(content=max_loops_prompt, role=MessageRole.SYSTEM, static=True)
        conversation_history = Message(
            content=self.aggregate_history(self._prompt.messages), role=MessageRole.USER, static=True
        )
        llm_final_attempt_result = self._run_llm(
            [system_message, input_message, conversation_history], config=config, **kwargs
        )
        llm_final_attempt = llm_final_attempt_result.output["content"]
        self._run_depends = [NodeDependency(node=self.llm).to_dict()]

        try:
            final_answer = XMLParser.extract_first_tag_lxml(llm_final_attempt, ["answer"])
            if final_answer is None:
                logger.warning("Max loops handler: lxml failed to extract <answer>, falling back to regex.")
                final_answer = XMLParser.extract_first_tag_regex(llm_final_attempt, ["answer"])

            if final_answer is None:
                logger.error(
                    "Max loops handler: Failed to extract <answer> tag even with fallbacks. Returning raw output."
                )
                final_answer = llm_final_attempt

        except Exception as e:
            logger.error(f"Max loops handler: Error during final answer extraction: {e}. Returning raw output.")
            final_answer = llm_final_attempt

        return f"{final_answer}"

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks required for the ReAct strategy."""
        super()._init_prompt_blocks()
        # Delegation guidance is rendered via prompt variables managed by AgentPromptManager
        self.system_prompt_manager.update_variables(
            self.system_prompt_manager.build_delegation_variables(self.delegation_allowed)
        )

        # Handle function calling schema generation first
        if self.inference_mode == InferenceMode.FUNCTION_CALLING:
            self._tools = schema_generator.generate_function_calling_schemas(
                self.tools, self.delegation_allowed, self.sanitize_tool_name, self.llm
            )
        elif self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
            self._response_format = schema_generator.generate_structured_output_schemas(
                self.tools, self.sanitize_tool_name, self.delegation_allowed
            )

        # Setup ReAct-specific prompts via prompt manager
        self.system_prompt_manager.setup_for_react_agent(
            inference_mode=self.inference_mode,
            parallel_tool_calls_enabled=self.parallel_tool_calls_enabled,
            has_tools=bool(self.tools),
        )

        # Only auto-wrap the entire role in a raw block if the user did not
        # provide explicit raw/endraw markers. This allows roles to mix
        # literal sections (via raw) with Jinja variables like {{ input }}
        # without creating nested raw blocks.
        if self.role:
            if ("{% raw %}" in self.role) or ("{% endraw %}" in self.role):
                self.system_prompt_manager.set_block("role", self.role)
            else:
                self.system_prompt_manager.set_block("role", f"{{% raw %}}{self.role}{{% endraw %}}")

    @staticmethod
    def _build_unique_file_key(files_map: dict[str, Any], base: str) -> str:
        key = base or "file"
        if key not in files_map:
            return key
        suffix = 1
        while f"{key}_{suffix}" in files_map:
            suffix += 1
        return f"{key}_{suffix}"

    def _merge_tool_files(self, aggregated: dict[str, Any], tool_name: str, files: Any) -> None:
        if not files:
            return

        sanitized_name = self.sanitize_tool_name(tool_name) or "tool"

        if isinstance(files, dict):
            for key, value in files.items():
                base_key = key or sanitized_name
                unique_key = self._build_unique_file_key(aggregated, base_key)
                aggregated[unique_key] = value
        elif isinstance(files, (list, tuple)):
            for idx, file_obj in enumerate(files):
                base_key = getattr(file_obj, "name", None) or f"{sanitized_name}_{idx}"
                unique_key = self._build_unique_file_key(aggregated, base_key)
                aggregated[unique_key] = file_obj
        else:
            unique_key = self._build_unique_file_key(aggregated, sanitized_name)
            aggregated[unique_key] = files

    @staticmethod
    def _separate_tool_result_and_files(execution_result: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(execution_result, dict):
            content = execution_result.get("content", "")
            files = execution_result.get("files", {})
            if isinstance(files, dict):
                return content, files
            if isinstance(files, (list, tuple)):
                return content, {str(index): file for index, file in enumerate(files)}
            if files:
                return content, {"result": files}
            return content, {}
        return execution_result, {}

    def _run_single_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        config: RunnableConfig,
        update_run_depends: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        tool = self.tool_by_names.get(self.sanitize_tool_name(tool_name))
        if not tool:
            return {
                "tool_name": tool_name,
                "success": False,
                "tool_input": tool_input,
                "result": f"Unknown tool: {tool_name}. Please use only available tools.",
                "files": {},
                "dependency": None,
            }

        delegate_final = self._should_delegate_final(tool, tool_input)
        if delegate_final and not update_run_depends:
            return {
                "tool_name": tool.name,
                "success": False,
                "tool_input": tool_input,
                "result": "delegate_final is only supported for single agent tool calls.",
                "files": {},
                "dependency": None,
            }

        try:
            tool_result, tool_files, dependency = self._run_tool(
                tool,
                tool_input,
                config,
                update_run_depends=update_run_depends,
                collect_dependency=True,
                delegate_final=delegate_final,
                **kwargs,
            )
            return {
                "tool_name": tool.name,
                "success": True,
                "tool_input": tool_input,
                "result": tool_result,
                "files": tool_files,
                "dependency": dependency,
            }
        except RecoverableAgentException as e:
            error_message = f"{type(e).__name__}: {e}"
            logger.error(error_message)
            return {
                "tool_name": tool.name,
                "success": False,
                "tool_input": tool_input,
                "result": error_message,
                "files": {},
                "dependency": None,
            }

    def _stream_tool_result(self, result: dict[str, Any], config: RunnableConfig, **kwargs) -> None:
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            try:
                self.stream_content(
                    content={
                        "name": result.get("tool_name"),
                        "input": result.get("tool_input"),
                        "result": result.get("result"),
                        "files": result.get("files"),
                    },
                    source=str(result.get("tool_name")),
                    step="tool",
                    config=config,
                    **kwargs,
                )
            except Exception as stream_err:
                logger.error(f"Streaming error for tool {result.get('tool_name')}: {stream_err}")

    def _execute_tools(
        self, tools_data: list[dict[str, Any]], config: RunnableConfig, **kwargs
    ) -> str | dict[str, Any]:
        """
        Execute one or more tools and gather their results.

        Args:
            tools_data (list): List of dictionaries containing name and input for each tool
            config (RunnableConfig): Configuration for the runnable
            **kwargs: Additional arguments for tool execution

        Returns:
            str | dict[str, Any]: Combined observation string with all tool results and optional files
        """
        all_results: list[dict[str, Any]] = []

        if not tools_data:
            return ""

        prepared_tools: list[dict[str, Any]] = []

        for idx, td in enumerate(tools_data):
            tool_name = td.get("name")
            tool_input = td.get("input")
            if tool_name is None or tool_input is None:
                error_message = "Invalid tool payload: missing 'name' or 'input'"
                logger.error(error_message)
                all_results.append(
                    {
                        "order": idx,
                        "tool_name": tool_name or UNKNOWN_TOOL_NAME,
                        "success": False,
                        "tool_input": tool_input,
                        "result": error_message,
                        "files": {},
                        "dependency": None,
                    }
                )
                continue
            prepared_tools.append({"order": idx, "name": tool_name, "input": tool_input})

        if prepared_tools:
            if len(prepared_tools) == 1:
                tool_payload = prepared_tools[0]
                res = self._run_single_tool(
                    tool_payload["name"],
                    tool_payload["input"],
                    config,
                    update_run_depends=True,
                    **kwargs,
                )
                res["order"] = tool_payload["order"]
                all_results.append(res)
                self._stream_tool_result(res, config, **kwargs)
            else:
                max_workers = len(prepared_tools)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {}
                    for tool_payload in prepared_tools:
                        future = executor.submit(
                            self._run_single_tool,
                            tool_payload["name"],
                            tool_payload["input"],
                            config,
                            False,
                            **kwargs,
                        )
                        future_map[future] = tool_payload

                    for future in as_completed(future_map.keys()):
                        tool_payload = future_map[future]
                        tool_name = tool_payload["name"]
                        tool_input = tool_payload["input"]
                        try:
                            res = future.result()
                        except Exception as e:
                            error_message = f"Error executing tool {tool_name}: {str(e)}"
                            logger.error(error_message)
                            res = {
                                "tool_name": tool_name,
                                "success": False,
                                "tool_input": tool_input,
                                "result": error_message,
                                "files": {},
                                "dependency": None,
                            }
                        res["order"] = tool_payload["order"]
                        all_results.append(res)
                        self._stream_tool_result(res, config, **kwargs)

        observation_parts: list[str] = []
        aggregated_files: dict[str, Any] = {}

        ordered_results = sorted(all_results, key=lambda r: r.get("order", 0))

        for result in ordered_results:
            tool_name = result.get("tool_name", UNKNOWN_TOOL_NAME)
            result_content = result.get("result", "")
            success_status = "SUCCESS" if result.get("success") else "ERROR"
            observation_parts.append(f"--- {tool_name} has resulted in {success_status} ---\n{result_content}")

            self._merge_tool_files(aggregated_files, tool_name, result.get("files"))

        dependencies = [result.get("dependency") for result in ordered_results if result.get("dependency")]
        if dependencies:
            self._run_depends = dependencies

        combined_observation = "\n\n".join(observation_parts)

        if aggregated_files:
            return {"content": combined_observation, "files": aggregated_files}
        return combined_observation
