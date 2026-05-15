import json
from concurrent.futures import as_completed
from typing import Any, Callable, Mapping

from litellm import get_supported_openai_params, supports_function_calling, supports_response_schema
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from dynamiq.callbacks import AgentStreamingParserCallback, StreamingQueueCallbackHandler
from dynamiq.executors.context import ContextAwareThreadPoolExecutor
from dynamiq.nodes.agents.base import Agent as BaseAgent
from dynamiq.nodes.agents.components import parser, schema_generator
from dynamiq.nodes.agents.components.history_manager import HistoryManagerMixin
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    JSONParsingError,
    MaxLoopsExceededException,
    OutputFileNotFoundError,
    ParsingError,
    RecoverableAgentException,
    TagNotFoundError,
)
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager, ReactPromptConfig
from dynamiq.nodes.agents.utils import SummarizationConfig, ToolCacheEntry, XMLParser, extract_message_text
from dynamiq.nodes.node import Node, NodeDependency
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME, ParallelToolCallsInputSchema, ToolCallItem
from dynamiq.nodes.tools.todo_tools import TodoItem, TodoWriteTool
from dynamiq.nodes.types import ActionType, Behavior, InferenceMode
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.cancellation import check_cancellation
from dynamiq.types.llm_tool import Tool
from dynamiq.types.streaming import (
    AgentReasoningEventMessageData,
    AgentToolData,
    AgentToolInputErrorEventMessageData,
    AgentToolResultEventMessageData,
    StreamingMode,
)
from dynamiq.utils import generate_uuid, serialize_files_in_value
from dynamiq.utils.json_parser import parse_llm_json_output
from dynamiq.utils.logger import logger


class ToolCallArguments(BaseModel):
    thought: str = ""
    action_input: dict | str

    @field_validator("thought", mode="before")
    @classmethod
    def _coerce_thought(cls, v: Any) -> str:
        return v or ""


class FinalAnswerArguments(BaseModel):
    thought: str = ""
    answer: str | dict | list
    output_files: str = ""

    @field_validator("thought", mode="before")
    @classmethod
    def _coerce_thought(cls, v: Any) -> str:
        return v or ""

    @field_validator("output_files", mode="before")
    @classmethod
    def _coerce_output_files(cls, v: Any) -> str:
        return v or ""


class FunctionCall(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v, strict=False)
            except json.JSONDecodeError as e:
                raise ValueError(f"Tool call arguments are not valid JSON: {e}")
        return v or {}

    def parse_as_tool_call(self) -> ToolCallArguments:
        try:
            return ToolCallArguments.model_validate(self.arguments)
        except Exception:
            raise ActionParsingException(
                "Your tool call is missing required fields. "
                "Every tool call must include 'thought' (your reasoning) "
                "and 'action_input' (the tool parameters as an object).",
                recoverable=True,
            )

    def parse_as_final_answer(self) -> FinalAnswerArguments:
        try:
            return FinalAnswerArguments.model_validate(self.arguments)
        except Exception:
            raise ActionParsingException(
                "Your final answer call is missing required fields. "
                "The 'provide_final_answer' function must include 'thought' (your reasoning) "
                "and 'answer' (your final response to the user).",
                recoverable=True,
            )


class ToolCall(BaseModel):
    id: str | None = None
    function: FunctionCall


class AgentState(BaseModel):
    """
    Encapsulates the dynamic state of an agent during execution.

    Tracks loop progress and todos. Provides its own serialization
    to string for injection into observations.
    """

    current_loop: int = 0
    max_loops: int = 0
    todos: list[TodoItem] = Field(default_factory=list)

    def reset(self, max_loops: int = 0) -> None:
        """Reset state for a new execution."""
        self.current_loop = 0
        self.max_loops = max_loops
        self.todos = []

    def update_loop(self, current: int) -> None:
        """Update current loop number."""
        self.current_loop = current

    def update_todos(self, todos: list[dict | TodoItem]) -> None:
        """Update todo list from dicts or TodoItem objects."""
        self.todos = [t if isinstance(t, TodoItem) else TodoItem(**t) for t in todos]

    def to_prompt_string(self) -> str:
        """
        Serialize state to a string for observation injection.

        Returns:
            str: Formatted state string, or empty string if no state to show.
        """
        sections = []

        if self.current_loop > 0:
            sections.append(f"Progress: Loop {self.current_loop}/{self.max_loops}")

        if self.todos:
            todo_lines = [t.to_display_string() for t in self.todos]
            sections.append("Todos:\n" + "\n".join(todo_lines))

        return "\n".join(sections) if sections else ""


UNKNOWN_TOOL_NAME = "unknown_tool"


class Agent(HistoryManagerMixin, BaseAgent):
    """Unified Agent that uses a ReAct-style strategy for processing tasks by interacting with tools in a loop."""

    name: str = "agent"
    max_loops: int = Field(default=15, ge=2)
    inference_mode: InferenceMode = Field(default=InferenceMode.DEFAULT)
    behaviour_on_max_loops: Behavior = Field(
        default=Behavior.RAISE,
        description="Define behavior when max loops are exceeded. Options are 'raise' or 'return'.",
    )
    direct_tool_output_enabled: bool = Field(
        default=False,
        description="Enable direct tool output capability. "
        "When True, the agent can return raw tool outputs directly without summarization.",
    )

    format_schema: list = Field(default_factory=list)
    summarization_config: SummarizationConfig = Field(default_factory=SummarizationConfig)
    state: AgentState = Field(default_factory=AgentState, exclude=True)
    response_format: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional JSON schema (dict) for structured final output. Also accepts a "
            "pydantic BaseModel subclass as input convenience — it is converted to its "
            "JSON schema dict immediately and stored as dict. When set, the agent's "
            "final answer is parsed from JSON into a dict. Default None keeps the "
            "existing string-return behaviour."
        ),
    )

    _tools: list[Tool] = []
    _response_format: dict[str, Any] | None = None
    _requested_output_files: list[str] = []
    _streaming_tool_run_id: str | None = None
    _streaming_tool_run_ids: list[str] = PrivateAttr(default_factory=list)

    @field_validator("response_format", mode="before")
    @classmethod
    def _normalize_response_format(cls, v):
        """Normalize any accepted input to a raw JSON schema dict.

        Routes all non-None inputs through ``unwrap_response_format`` so the
        stored value has a single shape regardless of how it was constructed:
        BaseModel class, litellm-wrapped dict, or raw schema dict all land as
        the raw schema dict.
        """
        if v is None:
            return v

        from dynamiq.nodes.agents.components import schema_generator

        return schema_generator.unwrap_response_format(v)

    def get_clone_attr_initializers(self) -> dict[str, Callable[[Node], Any]]:
        """
        Define attribute initializers for cloning.

        Ensures that cloned agents get fresh instances of:
        - _tool_cache: Independent tool execution cache
        - state: Independent AgentState to avoid race conditions in parallel execution

        Returns:
            Dictionary mapping attribute names to initializer functions
        """
        base = super().get_clone_attr_initializers()
        base.update(
            {
                "_tool_cache": lambda _: {},
                "state": lambda _: AgentState(),
            }
        )
        return base

    def reset_run_state(self):
        """Resets the agent's run state including AgentState.

        When resuming from a checkpoint, AgentState is not wiped — it will be
        restored from the checkpoint data inside _run_agent().
        """
        super().reset_run_state()
        if not self.is_resumed:
            self.state.reset()
        self._streaming_tool_run_id = None
        self._streaming_tool_run_ids = []
        self._pending_fc_tool_call_ids: list[str] = []

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

    def _emit_tool_input_error(
        self, error: Exception, loop_num: int, config: "RunnableConfig | None" = None, **kwargs
    ) -> None:
        """Emit a streaming event to signal that a tool input parse failed.

        Consumers that received partial tool_input chunks can use the
        tool_run_id in the event to discard them.
        """
        self._stream_agent_event(
            AgentToolInputErrorEventMessageData(
                tool_run_id=self._streaming_tool_run_id or "",
                name=self.name,
                error=str(error),
                loop_num=loop_num,
            ),
            "tool_input_error",
            config,
            **kwargs,
        )
        self._streaming_tool_run_id = None

    def _should_delegate_final(
        self,
        tool: Node | None,
        action_input: Any,
    ) -> bool:
        """Only Agent tools with per-call delegate_final flag can delegate."""
        if not self.delegation_allowed:
            return False

        if not isinstance(tool, SubAgentTool):
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
    def _ensure_context_manager_tool(self):
        """Automatically add ContextManagerTool when summarization is enabled."""
        try:
            if self.summarization_config.enabled:
                has_context_tool = any(isinstance(t, ContextManagerTool) for t in self.tools)
                if not has_context_tool:
                    context_tool = ContextManagerTool(
                        llm=self.llm,
                        name="context-manager",
                        token_budget_ratio=self.summarization_config.token_budget_ratio,
                    )
                    self.tools.append(context_tool)
                    self._excluded_tool_ids.add(context_tool.id)
        except Exception as e:
            logger.error(f"Failed to ensure ContextManagerTool: {e}")
        return self

    @model_validator(mode="after")
    def _ensure_todo_tools(self):
        """Automatically add TodoWriteTool when todo is enabled in file_store config."""
        try:
            if self.file_store.enabled and self.file_store.todo_enabled:
                has_todo_write = any(isinstance(t, TodoWriteTool) for t in self.tools)

                if not has_todo_write:
                    todo_tool = TodoWriteTool(
                        name="todo-write",
                        file_store=self.file_store.backend,
                    )
                    self.tools.append(todo_tool)
                    self._excluded_tool_ids.add(todo_tool.id)
                    logger.info("Agent: Added TodoWriteTool")
        except Exception as e:
            logger.error(f"Failed to ensure TodoWriteTool: {e}")
        return self

    @model_validator(mode="after")
    def validate_inference_mode(self):
        """Validate (and where safe, auto-correct) the inference mode for the chosen model.

        Runs after auto-tool validators so self.tools reflects the final set
        (including any ContextManagerTool / TodoWriteTool appended above).
        """
        model = self.llm.model
        match self.inference_mode:
            case InferenceMode.FUNCTION_CALLING:
                if not supports_function_calling(model=model):
                    raise ValueError(
                        f"Model {model} does not support function calling. "
                        f"Try inference_mode=InferenceMode.XML or InferenceMode.DEFAULT."
                    )

            case InferenceMode.STRUCTURED_OUTPUT:
                if self.tools and "bedrock/" in model:
                    fallback = InferenceMode.XML
                    logger.warning(
                        "Agent: STRUCTURED_OUTPUT + tools is not safe on Bedrock model '%s' "
                        "(LiteLLM emulates response_format via a forced json_tool_call that "
                        "providers reject when other tools are present). Auto-downgrading to %s.",
                        model,
                        fallback.value,
                    )
                    self.inference_mode = fallback
                    return self

                params = get_supported_openai_params(model=model) or []
                schema_ok = True
                try:
                    schema_ok = bool(supports_response_schema(model=model))
                except Exception:
                    schema_ok = True
                if "response_format" not in params or not schema_ok:
                    logger.warning(
                        "Agent: LiteLLM reports model '%s' may not fully support json_schema "
                        "structured output. Proceeding anyway; if it fails at runtime, switch "
                        "to inference_mode=InferenceMode.FUNCTION_CALLING or InferenceMode.XML.",
                        model,
                    )

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

    def _stream_agent_event(
        self,
        content: AgentReasoningEventMessageData | AgentToolResultEventMessageData | AgentToolInputErrorEventMessageData,
        step: str,
        config: RunnableConfig,
        **kwargs,
    ) -> None:
        """Stream agent event if streaming is enabled.

        Args:
            content: The event data (reasoning or tool result).
            step: Event type ("reasoning" or "tool").
            config: Configuration for the runnable.
            **kwargs: Additional keyword arguments.
        """
        if not (self.streaming.enabled and self.streaming.mode == StreamingMode.ALL):
            return

        source = content.tool.name if step == "reasoning" else content.name
        try:
            stream_content = content.to_dict() if hasattr(content, "to_dict") else content.model_dump()
        except Exception as e:
            logger.error(f"Failed to serialize stream agent event via to_dict, applying fallback: {e}")
            stream_content = serialize_files_in_value(content.model_dump())
        self.stream_content(
            content=stream_content,
            source=source,
            step=step,
            config=config,
            **kwargs,
        )

    def _stream_batch_reasoning_event(
        self,
        prepared_tools: list[dict[str, Any]],
        thought: str | None,
        loop_num: int,
        config: RunnableConfig,
        **kwargs,
    ) -> str:
        """Stream a single batch reasoning event for parallel tool calls.

        Builds per-tool reasoning objects (assigning a ``tool_run_id`` to each
        prepared tool payload) and emits one ``run_parallel`` reasoning event
        whose ``action_input`` is the list of per-tool reasoning dicts.

        Args:
            prepared_tools: Mutable list of tool payloads; each dict gets a
                ``tool_run_id`` key added in-place.
            thought: The agent's reasoning text.
            loop_num: Current loop iteration number.
            config: Configuration for the runnable.
            **kwargs: Additional keyword arguments forwarded to streaming.

        Returns:
            str: The batch tool_run_id used for the run_parallel event.
        """
        per_tool_reasoning = []
        for i, tp in enumerate(prepared_tools):
            tid = self._streaming_tool_run_ids[i] if i < len(self._streaming_tool_run_ids) else generate_uuid()
            tp["tool_run_id"] = tid
            resolved = self.tool_by_names.get(self.sanitize_tool_name(tp["name"]))
            tool_data = AgentToolData(
                name=resolved.name if resolved else tp["name"],
                type=resolved.type if resolved else "unknown",
                action_type=resolved.action_type.value if resolved and resolved.action_type else None,
            )
            per_tool_reasoning.append(
                AgentReasoningEventMessageData(
                    tool_run_id=tid,
                    thought=tp.get("thought", ""),
                    action=tp["name"],
                    tool=tool_data,
                    action_input=tp["input"],
                    loop_num=loop_num,
                ).model_dump()
            )

        batch_tool_run_id = (
            generate_uuid() if self._streaming_tool_run_ids else self._streaming_tool_run_id or generate_uuid()
        )
        self._streaming_tool_run_id = None
        self._streaming_tool_run_ids = []
        batch_tool_data = AgentToolData(
            name=PARALLEL_TOOL_NAME,
            type="tool",
            action_type=ActionType.PARALLEL_EXECUTION.value,
        )
        self._stream_agent_event(
            AgentReasoningEventMessageData(
                tool_run_id=batch_tool_run_id,
                thought=thought or "",
                action=PARALLEL_TOOL_NAME,
                tool=batch_tool_data,
                action_input=per_tool_reasoning,
                loop_num=loop_num,
            ),
            "reasoning",
            config,
            **kwargs,
        )
        return batch_tool_run_id

    def _stream_batch_tool_result_event(
        self,
        batch_tool_run_id: str,
        prepared_tools: list[dict[str, Any]],
        all_results: list[dict[str, Any]],
        loop_num: int,
        config: RunnableConfig,
        **kwargs,
    ) -> None:
        """Stream a completion event after all parallel tools have finished.

        Emits a single ``run_parallel`` tool-result event summarizing which
        tools were executed and their status, without including the actual results.

        Args:
            batch_tool_run_id: The tool_run_id from the batch reasoning event.
            prepared_tools: The tool payloads (with ``tool_run_id`` assigned).
            all_results: Collected results from all tool executions.
            loop_num: Current loop iteration number.
            config: Configuration for the runnable.
            **kwargs: Additional keyword arguments forwarded to streaming.
        """
        results_by_run_id = {r.get("tool_run_id"): r for r in all_results if r.get("tool_run_id")}
        per_tool_summary = []
        for tp in prepared_tools:
            tool_name = tp["name"]
            result_entry = results_by_run_id.get(tp.get("tool_run_id"), {})
            resolved = self.tool_by_names.get(self.sanitize_tool_name(tool_name))
            tool_data = AgentToolData(
                name=resolved.name if resolved else tool_name,
                type=resolved.type if resolved else "unknown",
                action_type=resolved.action_type.value if resolved and resolved.action_type else None,
            )
            per_tool_summary.append(
                AgentToolResultEventMessageData(
                    tool_run_id=tp.get("tool_run_id", ""),
                    name=tool_name,
                    tool=tool_data,
                    input=tp["input"],
                    result=None,
                    loop_num=loop_num,
                    status=RunnableStatus.SUCCESS if result_entry.get("success") else RunnableStatus.FAILURE,
                ).model_dump()
            )

        batch_tool_data = AgentToolData(
            name=PARALLEL_TOOL_NAME,
            type="tool",
            action_type=ActionType.PARALLEL_EXECUTION.value,
        )

        overall_status = (
            RunnableStatus.SUCCESS
            if all(s.get("status") == RunnableStatus.SUCCESS for s in per_tool_summary)
            else RunnableStatus.FAILURE
        )

        self._stream_agent_event(
            AgentToolResultEventMessageData(
                tool_run_id=batch_tool_run_id,
                name=PARALLEL_TOOL_NAME,
                tool=batch_tool_data,
                input=[{"name": tp["name"], "input": tp["input"]} for tp in prepared_tools],
                result=per_tool_summary,
                loop_num=loop_num,
                status=overall_status,
            ),
            "tool",
            config,
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
            self._pending_fc_tool_call_ids = []
            if "tool_calls" in dict[Any, Any](llm_result.output):
                try:
                    tool_calls = llm_result.output["tool_calls"]
                    payload: list[dict] = []
                    pending_ids: list[str] = []
                    fa_stub_ids: list[str] = []
                    for tc in tool_calls:
                        function_name = tc["function"]["name"].strip()
                        tc_id = tc.get("id") or generate_uuid()
                        raw_args = tc["function"]["arguments"]
                        arguments_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
                        entry = {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": arguments_str,
                            },
                        }
                        if function_name == "provide_final_answer":
                            payload.append(entry)
                            fa_stub_ids.append(tc_id)
                        elif self.parallel_tool_calls_enabled or not pending_ids:
                            # Only the first non-final-answer call is kept when parallel
                            # execution is disabled
                            payload.append(entry)
                            pending_ids.append(tc_id)
                    if payload:
                        called_names = ", ".join(p["function"]["name"] for p in payload)
                        self._prompt.messages.append(
                            Message(
                                role=MessageRole.ASSISTANT,
                                content=f"Calling: {called_names}",
                                tool_calls=payload,
                                static=True,
                            )
                        )
                        for fa_id in fa_stub_ids:
                            self._prompt.messages.append(
                                Message(
                                    role=MessageRole.TOOL,
                                    content="Acknowledged.",
                                    tool_call_id=fa_id,
                                    name="provide_final_answer",
                                    static=True,
                                )
                            )
                        self._pending_fc_tool_call_ids = pending_ids
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
            self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=llm_generated_output, static=True))

    def _rollback_orphan_fc_payload(self) -> None:
        """Remove the assistant(tool_calls=[…]) and any inline FA stub replies
        appended by _append_assistant_message when validation later fails.

        Keeping unpaired tool_calls in history would violate OpenAI's FC protocol
        and 400 the next request. Called from the recoverable-error branch.
        """
        if not (self.inference_mode == InferenceMode.FUNCTION_CALLING and self._pending_fc_tool_call_ids):
            return
        while self._prompt.messages and self._prompt.messages[-1].role == MessageRole.TOOL:
            self._prompt.messages.pop()
        if (
            self._prompt.messages
            and self._prompt.messages[-1].role == MessageRole.ASSISTANT
            and self._prompt.messages[-1].tool_calls
        ):
            self._prompt.messages.pop()
        self._pending_fc_tool_call_ids = []

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
            thought, final_answer, output_files_raw = parser.extract_default_final_answer(llm_generated_output)
            self._requested_output_files = self._parse_output_files_csv(output_files_raw)
            self.log_final_output(thought, final_answer, loop_num)
            return thought, "final_answer", final_answer

        thought, action, action_input = parser.parse_default_action(llm_generated_output)
        self.log_reasoning(thought, action, action_input, loop_num)
        return thought, action, action_input

    def _handle_function_calling_mode(
        self, llm_result: Any, loop_num: int
    ) -> tuple[str | None, str | None, dict | list | None] | tuple[str, str, str]:
        """Handle FUNCTION_CALLING inference mode parsing.

        When the LLM returns multiple tool_calls (native parallel tool calling),
        all calls are collected and routed through the existing parallel execution
        path using PARALLEL_TOOL_NAME, preserving identical streaming events.

        Returns:
            tuple: (thought, action, action_input) for normal actions
                   (thought, "final_answer", final_answer) for final answers
        """
        if self.verbose:
            logger.info(f"Agent {self.name} - {self.id}: using function calling inference mode")

        if not llm_result.output.get("tool_calls"):
            logger.error("Error: No function called.")
            raise ActionParsingException(
                "You must always respond by calling a function. "
                "Call a tool function to continue, or call 'provide_final_answer' to finish.",
                recoverable=True,
            )

        try:
            tool_calls = [ToolCall.model_validate(tc) for tc in llm_result.output["tool_calls"]]
        except Exception as e:
            raise ActionParsingException(f"Error parsing tool calls: {e}", recoverable=True)

        if not tool_calls:
            raise ActionParsingException(
                "You must always respond by calling a function. "
                "Call a tool function to continue, or call 'provide_final_answer' to finish.",
                recoverable=True,
            )

        first_call = tool_calls[0]
        action = first_call.function.name.strip()

        if action == "provide_final_answer":
            final_args = first_call.function.parse_as_final_answer()
            thought = final_args.thought
            self._requested_output_files = self._parse_output_files_csv(final_args.output_files)
            self.log_final_output(thought, final_args.answer, loop_num)
            return thought, "final_answer", final_args.answer

        actual_tool_calls = [tc for tc in tool_calls if tc.function.name.strip() != "provide_final_answer"]

        if len(actual_tool_calls) > 1 and not self.parallel_tool_calls_enabled:
            logger.warning(
                f"Agent {self.name} - {self.id}: LLM returned {len(actual_tool_calls)} tool calls "
                f"but parallel_tool_calls_enabled is False. Only the first tool call will be executed, "
                f"remaining {len(actual_tool_calls) - 1} call(s) will be dropped."
            )

        if len(actual_tool_calls) > 1 and self.parallel_tool_calls_enabled:
            tool_items = []
            for tc in actual_tool_calls:
                args = tc.function.parse_as_tool_call()
                tc_input = args.action_input
                if isinstance(tc_input, str):
                    try:
                        tc_input = json.loads(tc_input, strict=False)
                    except json.JSONDecodeError as e:
                        raise ActionParsingException(f"Error parsing action_input string. {e}", recoverable=True)
                if not isinstance(tc_input, dict):
                    tc_input = {"input": tc_input}
                tool_items.append(
                    ToolCallItem(
                        name=tc.function.name.strip(),
                        input=tc_input,
                        thought=args.thought,
                    )
                )
            thought = "\n".join(item.thought for item in tool_items if item.thought)
            validated = ParallelToolCallsInputSchema(tools=tool_items)
            action_input = validated.model_dump()
            self.log_reasoning(thought, PARALLEL_TOOL_NAME, action_input["tools"], loop_num)
            return thought, PARALLEL_TOOL_NAME, action_input

        single_call = actual_tool_calls[0]
        action = single_call.function.name.strip()
        args = single_call.function.parse_as_tool_call()
        thought = args.thought
        action_input = args.action_input
        if isinstance(action_input, str):
            try:
                action_input = json.loads(action_input, strict=False)
            except json.JSONDecodeError as e:
                raise ActionParsingException(f"Error parsing action_input string. {e}", recoverable=True)
        if not isinstance(action_input, dict):
            action_input = {"input": action_input}

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
                try:
                    llm_generated_output_json = json.loads(llm_generated_output, strict=False)
                except json.JSONDecodeError:
                    # Handle known LLM bug where multiple JSON objects are returned.
                    # Use raw_decode to parse only the first valid JSON object.
                    decoder = json.JSONDecoder(strict=False)
                    llm_generated_output_json, _ = decoder.raw_decode(llm_generated_output.strip())
            else:
                llm_generated_output_json = llm_generated_output
        except (json.JSONDecodeError, ValueError) as e:
            raise ActionParsingException(f"Error parsing action. {e}", recoverable=True)

        if "action" not in llm_generated_output_json or "thought" not in llm_generated_output_json:
            raise ActionParsingException("No action or thought provided.", recoverable=True)

        thought = llm_generated_output_json["thought"]
        action = llm_generated_output_json["action"]
        action_input = llm_generated_output_json["action_input"]

        if action == "finish":
            self._requested_output_files = self._parse_output_files_csv(
                llm_generated_output_json.get("output_files") or ""
            )
            self.log_final_output(thought, action_input, loop_num)
            return thought, "final_answer", action_input

        try:
            if isinstance(action_input, str):
                action_input = action_input.replace("\\'", "'")
                action_input = json.loads(action_input, strict=False)
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

        try:
            parsed_data = XMLParser.parse(
                llm_generated_output,
                required_tags=["thought", "answer"],
                optional_tags=["output", "output_files"],
            )
            thought = parsed_data.get("thought")
            final_answer = parsed_data.get("answer")

            self._requested_output_files = self._parse_output_files_csv(
                parsed_data.get("output_files") or ""
            )

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
            except JSONParsingError as e:
                logger.error(f"XMLParser: Invalid JSON in action_input: {e}")
                raise ActionParsingException(
                    "The <action_input> value must be valid JSON. Put the whole JSON on one line; "
                    'use \\n for newlines inside strings and \\" for quotes. '
                    "Provide <thought> with <action> and <action_input> again.",
                    recoverable=True,
                )
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

    def _setup_stop_sequences(self) -> None:
        """Configure LLM stop sequences based on the current inference mode."""
        stop_sequences = []
        if self.inference_mode == InferenceMode.DEFAULT:
            stop_sequences.extend(["Observation: ", "\nObservation:"])
        elif self.inference_mode == InferenceMode.XML:
            stop_sequences.extend(["\nObservation:", "Observation:", "</output>\n<", "</output><"])
        self.llm.stop = stop_sequences

    def _setup_prompt_and_stop_sequences(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
    ) -> None:
        """Setup the prompt with system message, history, and configure stop sequences.

        Args:
            input_message: The user's input message
            history_messages: Optional conversation history
        """
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=self.generate_prompt(
                tools_name=self.tool_names,
                input_formats=schema_generator.generate_input_formats(self.tools, self.sanitize_tool_name),
            ),
            static=True,
        )

        if history_messages:
            self._prompt.messages = [system_message, *history_messages, input_message]
        else:
            self._prompt.messages = [system_message, input_message]

        self._history_offset = 1
        self._pinned_input = input_message
        self._setup_stop_sequences()

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
        self,
        action: str,
        action_input: Any,
        thought: str,
        loop_num: int,
        config: RunnableConfig,
        update_run_depends: bool = True,
        collect_dependency: bool = False,
        is_parallel: bool = False,
        tool_run_id: str | None = None,
        **kwargs,
    ) -> tuple[Any, list, bool, bool, dict | None]:
        """Execute a single tool with caching support.

        Args:
            update_run_depends: Whether to update self._run_depends. Set to False for parallel execution.
            collect_dependency: Whether to collect and return the dependency dict.
            is_parallel: Whether this tool is being executed in parallel with other tools.
                When True, the tool will be cloned for thread-safe execution.
            tool_run_id: Pre-generated tool run ID. When provided, reuses it instead of
                generating a new one, so tool result events match the batch reasoning event.

        Returns:
            tuple: (tool_result, tool_files, is_delegated, success, dependency)
        """
        tool = self.tool_by_names.get(self.sanitize_tool_name(action))

        if not tool:
            error_message = (
                f"Unknown tool: {action}. Use only available tools and provide only the tool's name in the "
                "action field. Do not include any additional reasoning. "
                "Please correct the action field or state that you cannot answer the question."
            )
            return error_message, [], False, False, None

        tool_run_id = tool_run_id or self._streaming_tool_run_id or generate_uuid()
        self._streaming_tool_run_id = None
        self._streaming_tool_run_ids = []
        tool_data = AgentToolData(
            name=tool.name,
            type=tool.type,
            action_type=tool.action_type.value if tool.action_type else None,
        )

        self._stream_agent_event(
            AgentReasoningEventMessageData(
                tool_run_id=tool_run_id,
                thought=thought or "",
                action=action,
                tool=tool_data,
                action_input=action_input,
                loop_num=loop_num,
            ),
            "reasoning",
            config,
            **kwargs,
        )
        try:
            check_cancellation(config)
            if isinstance(tool, ContextManagerTool):
                tool_result = None
                to_summarize, to_preserve = self._split_history()
                if not to_summarize:
                    logger.info(f"Agent {self.name} - {self.id}: Nothing to summarize, skipping context compaction.")
                    skip_message = (
                        "Nothing was summarized because the conversation history is small enough to fit in context."
                    )
                    self._stream_agent_event(
                        AgentToolResultEventMessageData(
                            tool_run_id=tool_run_id,
                            name=tool.name,
                            tool=tool_data,
                            input=action_input,
                            result=skip_message,
                            files=[],
                            loop_num=loop_num,
                            output={},
                            status=RunnableStatus.SKIP,
                        ),
                        "tool",
                        config,
                        **kwargs,
                    )
                    return skip_message, [], False, True, None
                tool_input = {**(action_input if isinstance(action_input, dict) else {}), "messages": to_summarize}
            else:
                tool_cache_entry = ToolCacheEntry(action=action, action_input=action_input)
                tool_result = self._tool_cache.get(tool_cache_entry, None)
                tool_input = action_input

            delegate_final = self._should_delegate_final(tool, action_input)

            dependency: dict | None = None
            if not tool_result:
                tool_kwargs = kwargs.copy()

                run_tool_result = self._run_tool(
                    tool,
                    tool_input,
                    config,
                    delegate_final=delegate_final,
                    update_run_depends=update_run_depends,
                    collect_dependency=collect_dependency,
                    is_parallel=is_parallel,
                    tool_run_id=tool_run_id,
                    **tool_kwargs,
                )
                if collect_dependency:
                    tool_result, tool_files, tool_output_meta, dependency = run_tool_result
                else:
                    tool_result, tool_files, tool_output_meta = run_tool_result

            else:
                logger.info(f"Agent {self.name} - {self.id}: Cached output of {action} found.")
                tool_result, tool_output_meta = tool_result
                tool_files = []

            if delegate_final:
                self.log_final_output(thought, tool_result, loop_num)
                # Stream tool result (with files) before streaming final answer
                self._stream_agent_event(
                    AgentToolResultEventMessageData(
                        tool_run_id=tool_run_id,
                        name=tool.name,
                        tool=tool_data,
                        input=action_input,
                        result=tool_result,
                        files=tool_files,
                        loop_num=loop_num,
                        output=tool_output_meta,
                    ),
                    "tool",
                    config,
                    **kwargs,
                )
                if self.streaming.enabled:
                    self.stream_content(
                        content=tool_result,
                        source=tool.name,
                        step="answer",
                        config=config,
                        **kwargs,
                    )
                return tool_result, tool_files, True, True, dependency

            if isinstance(tool, ContextManagerTool):
                self._compact_history(
                    summary=tool_output_meta.get("summary", tool_result),
                    pinned_content=extract_message_text(self._pinned_input) if self._pinned_input else None,
                    preserved=to_preserve,
                )

            # Stream the result
            self._stream_agent_event(
                AgentToolResultEventMessageData(
                    tool_run_id=tool_run_id,
                    name=tool.name,
                    tool=tool_data,
                    input=action_input,
                    result=tool_result,
                    files=tool_files,
                    loop_num=loop_num,
                    output=tool_output_meta,
                ),
                "tool",
                config,
                **kwargs,
            )

            return tool_result, tool_files, False, True, dependency

        except RecoverableAgentException as e:
            # Stream error result with the same tool_run_id used for reasoning
            error_message = f"{type(e).__name__}: {e}"
            self._stream_agent_event(
                AgentToolResultEventMessageData(
                    tool_run_id=tool_run_id,
                    name=tool.name,
                    tool=tool_data,
                    input=action_input,
                    result=error_message,
                    files=[],
                    loop_num=loop_num,
                    output={},
                    status=RunnableStatus.FAILURE,
                ),
                "tool",
                config,
                **kwargs,
            )
            return error_message, [], False, False, None

    def _add_observation(self, tool_result: Any) -> None:
        """Add observation to prompt.

        Args:
            tool_result: The result from the tool execution.
        """
        observation = f"\nObservation: {tool_result}\n"
        self._prompt.messages.append(Message(role=MessageRole.USER, content=observation, static=True))

    def _emit_tool_observations(
        self,
        tool_result: Any,
        ordered_results: list[dict[str, Any]] | None = None,
        tool_name: str | None = None,
    ) -> None:
        """Append tool observations to prompt history.

        In FUNCTION_CALLING mode with stashed tool_call_ids, emits one
        ``role: "tool"`` message per id (per-tool result for parallel runs,
        the combined result for single-tool runs) — required by OpenAI's
        function-calling protocol. Falls back to the legacy ``role: "user"``
        ``Observation: ...`` message in all other cases.
        """
        pending_ids = getattr(self, "_pending_fc_tool_call_ids", None) or []
        if self.inference_mode == InferenceMode.FUNCTION_CALLING and pending_ids:
            if ordered_results and len(ordered_results) == len(pending_ids):
                for tc_id, result in zip(pending_ids, ordered_results):
                    self._prompt.messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=str(result.get("result", "")),
                            tool_call_id=tc_id,
                            name=result.get("tool_name"),
                            static=True,
                        )
                    )
            else:
                self._prompt.messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=str(tool_result),
                        tool_call_id=pending_ids[0],
                        name=tool_name,
                        static=True,
                    )
                )
            self._pending_fc_tool_call_ids = []
            return
        self._add_observation(tool_result)

    def _validate_parallel_tool_input(self, action_input: Any) -> list[dict[str, Any]] | None:
        """Validate and parse parallel tool input schema.

        If validation fails, logs error and adds observation for agent recovery.

        Args:
            action_input: Raw input from LLM for the parallel tool.

        Returns:
            list: Validated tools list, or None if validation failed
        """
        try:
            validated = ParallelToolCallsInputSchema.model_validate(action_input).model_dump()
            return validated["tools"]
        except Exception as e:
            error_message = f"Invalid parallel tool input: {e}"
            logger.error(error_message)
            self._add_observation(error_message)
            return None

    def _check_subagent_limits(self, tools_data: list[dict[str, Any]], action: str) -> str | None:
        """Check if any SubAgentTool in the batch would exceed its max_calls.

        Returns an error message if limit exceeded, None if all OK.
        """
        subagent_calls: dict[str, int] = {}
        is_parallel = self.sanitize_tool_name(action) == PARALLEL_TOOL_NAME

        for td in tools_data:
            name = td.get("name", action) if (is_parallel and isinstance(td, dict)) else action
            sanitized = self.sanitize_tool_name(name)
            tool = self.tool_by_names.get(sanitized)
            if tool and isinstance(tool, SubAgentTool) and tool.max_calls is not None:
                subagent_calls[sanitized] = subagent_calls.get(sanitized, 0) + 1

        exceeded = []
        for name, batch_count in subagent_calls.items():
            tool = self.tool_by_names[name]
            remaining = tool.max_calls - tool._call_count
            if batch_count > remaining:
                exceeded.append(
                    f"'{tool.name}' needs {batch_count} call(s) but only {remaining} remaining "
                    f"(limit: {tool.max_calls}, used: {tool._call_count})"
                )

        if exceeded:
            return (
                f"Sub-agent invocation limit exceeded. {'; '.join(exceeded)}. "
                f"No sub-agent calls in this batch were executed. "
                f"Use other tools or provide a final answer using information already gathered."
            )

        return None

    def _should_skip_parallel_mode(
        self, action: str | None, action_input: Any
    ) -> tuple[bool, str | None, Any, list[str]]:
        """Check if parallel mode should be skipped for ContextManagerTool.

        When ContextManagerTool is detected in a parallel tool list, we filter
        to only execute that tool to ensure safe context modification.

        Args:
            action: The action to execute
            action_input: The action input (list for parallel mode)

        Returns:
            tuple: (skip_parallel, action_name, action_input, skipped_tools)
                - skip_parallel: True if parallel mode should be skipped
                - action_name: The tool name to execute
                - action_input: The tool input to use
                - skipped_tools: List of tool names that were skipped
        """
        # Get ContextManagerTool names
        context_manager_names = {
            self.sanitize_tool_name(t.name) for t in self.tools if isinstance(t, ContextManagerTool)
        }

        # Check if ContextManagerTool is in a parallel tool list
        if isinstance(action_input, list):
            for tool_data in action_input:
                if isinstance(tool_data, dict):
                    tool_name = self.sanitize_tool_name(tool_data.get("name", ""))
                    if tool_name in context_manager_names:
                        # Collect names of other tools that will be skipped
                        skipped_tools = [
                            td.get("name", "unknown")
                            for td in action_input
                            if isinstance(td, dict)
                            and self.sanitize_tool_name(td.get("name", "")) not in context_manager_names
                        ]
                        logger.info(
                            f"Agent {self.name} - {self.id}: ContextManagerTool detected in parallel call. "
                            f"Filtering to execute only ContextManagerTool. Skipped tools: {skipped_tools}"
                        )
                        return True, tool_data.get("name", ""), tool_data.get("input", {}), skipped_tools
        elif isinstance(action, str) and self.sanitize_tool_name(action) in context_manager_names:
            # Single tool mode - ContextManagerTool detected, no tools skipped
            return True, action, action_input, []

        return False, action, action_input, []

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
        check_cancellation(config)
        if action and self.tools:
            tool_result = None
            skipped_tools: list[str] = []

            if self.sanitize_tool_name(action) == PARALLEL_TOOL_NAME:
                action_input = self._validate_parallel_tool_input(action_input)
                if action_input is None:
                    return None

            # Check if ContextManagerTool is in the action - if so, skip parallel mode
            skip_parallel, action, action_input, skipped_tools = self._should_skip_parallel_mode(action, action_input)

            # Handle XML parallel mode (only for multiple tools, not for ContextManagerTool)
            tools_data = action_input if isinstance(action_input, list) else [action_input]

            # Check subagent invocation limits before executing
            subagent_error = self._check_subagent_limits(tools_data, action)
            if subagent_error:
                self._add_observation(subagent_error)
                return None

            ordered_results: list[dict[str, Any]] = []
            if (
                self.sanitize_tool_name(action) == PARALLEL_TOOL_NAME
                and self.parallel_tool_calls_enabled
                and not skip_parallel
            ):
                tool_result, _, ordered_results = self._execute_tools(
                    tools_data, thought, loop_num, config, **kwargs
                )
            else:
                tool_result, _, is_delegated, _, _ = self._execute_single_tool(
                    action, action_input, thought, loop_num, config, **kwargs
                )
                if is_delegated:
                    return tool_result

            if skipped_tools:
                skipped_notice = (
                    f"\n\n[Note: The following tools were NOT executed because context-manager "
                    f"must run alone: {', '.join(skipped_tools)}. Please call them separately after "
                    f"context compression completes.]"
                )
                tool_result = f"{tool_result}{skipped_notice}" if tool_result else skipped_notice

            self._emit_tool_observations(tool_result, ordered_results, tool_name=action)

        # else: No action or no tools available - no reasoning to stream

        return None

    def _inject_state_into_messages(self, messages: list[Message | VisionMessage]) -> list[Message | VisionMessage]:
        """
        Create a copy of messages with state injected into the last user message.

        Original messages are not modified. Handles both Message and VisionMessage types.
        """
        state_info = self.state.to_prompt_string()
        if not state_info or not messages:
            return messages

        last_msg = messages[-1]
        if last_msg.role != MessageRole.USER:
            return messages

        state_suffix = f"\n\n[State: {state_info}]"

        if isinstance(last_msg, VisionMessage):
            new_content = list(last_msg.content) + [VisionMessageTextContent(text=state_suffix)]
            return messages[:-1] + [
                VisionMessage(
                    role=last_msg.role,
                    content=new_content,
                    static=last_msg.static,
                )
            ]

        return messages[:-1] + [
            Message(
                role=last_msg.role,
                content=f"{last_msg.content}{state_suffix}",
                metadata=last_msg.metadata,
                static=last_msg.static,
            )
        ]

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

        self.state.max_loops = self.max_loops

        completed = self.get_start_iteration()
        start_loop = completed + 1 if completed > 0 else 1
        self._requested_output_files = []

        if start_loop > 1:
            logger.info(
                f"Agent {self.name} - {self.id}: resuming from loop {start_loop}, "
                f"skipping {completed} completed loops"
            )
            if not self._prompt.messages:
                self._setup_prompt_and_stop_sequences(input_message, history_messages)
            else:
                self._pinned_input = input_message
            self._setup_stop_sequences()
            self.state.max_loops = self.max_loops
            self._refresh_agent_state(start_loop)
        else:
            self._refresh_agent_state(1)
            self._setup_prompt_and_stop_sequences(input_message, history_messages)
        self.reset_resumed_flag()

        for loop_num in range(start_loop, self.max_loops + 1):
            check_cancellation(config)

            if loop_num > start_loop:
                self._refresh_agent_state(loop_num)

            try:
                streaming_callback, llm_config, original_streaming_enabled = self._setup_streaming_callback(
                    config, loop_num, **kwargs
                )

                # Append state to the last user message before LLM call
                messages = self._inject_state_into_messages(self._prompt.messages)

                try:
                    native_parallel = (
                        self.parallel_tool_calls_enabled and self.inference_mode == InferenceMode.FUNCTION_CALLING
                    )
                    llm_result = self._run_llm(
                        messages=messages,
                        tools=self._tools,
                        response_format=self._response_format,
                        config=llm_config,
                        parallel_tool_calls=True if native_parallel else None,
                        **kwargs,
                    )
                finally:
                    if not original_streaming_enabled:
                        try:
                            self.llm.streaming.enabled = original_streaming_enabled
                        except Exception:
                            logger.error("Failed to restore llm.streaming.enabled state")

                check_cancellation(config)

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
                logger.info(f"Agent {self.name} - {self.id}: Loop {loop_num}, reasoning:\n{llm_reasoning}...")

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
                    self._resolve_requested_output_files(strict=True)
                    final_answer = result[2]
                    if self.response_format is not None:
                        final_answer = self._coerce_to_response_format(final_answer)
                    return final_answer

                # Handle recovery (for modes that support it)
                # Check if action is None, which indicates (None, None, None) recovery
                if result[1] is None:
                    continue

                thought, action, action_input = result
                check_cancellation(config)

                final_answer = self._execute_tools_and_update_prompt(
                    action, action_input, thought, loop_num, config, **kwargs
                )
                check_cancellation(config)

                self._completed_loops = loop_num

                if config and config.checkpoint and config.checkpoint.context:
                    config.checkpoint.context.save_mid_run(self.id)

                if final_answer is not None:
                    if self.response_format is not None:
                        final_answer = self._coerce_to_response_format(final_answer)
                    return final_answer

            except OutputFileNotFoundError as e:
                self._requested_output_files = []
                self._append_recovery_instruction(
                    error_label=type(e).__name__,
                    error_detail=str(e),
                    llm_generated_output=llm_generated_output,
                    extra_guidance=(
                        "The response format is correct, but some files could not be found. "
                        "Please create the missing files or correct the file paths, "
                        "then provide your final answer again."
                    ),
                )
                continue

            except ParsingError as e:
                self._append_recovery_instruction(
                    error_label=type(e).__name__,
                    error_detail=str(e),
                    llm_generated_output=llm_generated_output,
                    extra_guidance=(
                        "Your final answer must be valid JSON that conforms exactly to the declared "
                        "response_format schema. Return only the JSON document — no prose, Markdown, "
                        "or code fences — and provide the final answer again."
                    ),
                )
                continue

            except ActionParsingException as e:
                self._emit_tool_input_error(e, loop_num, config, **kwargs)
                self._rollback_orphan_fc_payload()

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
                elif self.inference_mode == InferenceMode.FUNCTION_CALLING:
                    extra_guidance = (
                        "You MUST respond by calling a function — never plain text. "
                        "Call a tool with 'thought' and 'action_input', "
                        "or call 'provide_final_answer' with 'thought' and 'answer' to finish."
                    )

                self._append_recovery_instruction(
                    error_label=type(e).__name__,
                    error_detail=str(e),
                    llm_generated_output=llm_generated_output,
                    extra_guidance=extra_guidance,
                )
                continue

            # Inject automatic summarization if token limit exceeded (like Context Manager Tool)
            self._try_summarize_history(config=config, **kwargs)

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
            self._resolve_requested_output_files(strict=False)
            if self.streaming.enabled:
                self.stream_content(
                    content=max_loop_final_answer,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            if self.response_format is not None:
                try:
                    max_loop_final_answer = self._coerce_to_response_format(max_loop_final_answer)
                except ParsingError as e:
                    raise ParsingError(f"Max-loops fallback answer could not be coerced to response_format: {e}") from e
            return max_loop_final_answer

    def _try_summarize_history(
        self,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> None:
        """
        Check if summarization is needed and inject it automatically if token limit is exceeded.

        Works like an automatic Context Manager Tool invocation.

        Args:
            config: Configuration for the agent run
            **kwargs: Additional parameters for running the agent
        """
        if not self.summarization_config.enabled:
            return

        if self.is_token_limit_exceeded():
            logger.info(
                f"Agent {self.name} - {self.id}: Token limit exceeded. Automatically invoking Context Manager Tool."
            )

            context_tool = next((t for t in self.tools if isinstance(t, ContextManagerTool)), None)

            if context_tool is None:
                logger.error(f"Agent {self.name} - {self.id}: Context Manager Tool not found.")
                return

            action = self.sanitize_tool_name(context_tool.name)

            self._execute_tools_and_update_prompt(
                action=action,
                action_input={},
                thought=None,
                loop_num=0,  # Use 0 for automatic invocation
                config=config,
                **kwargs,
            )

    @staticmethod
    def _parse_output_files_csv(raw: str) -> list[str]:
        """Parse a comma-separated string of file paths into a list.

        Strips whitespace from each entry and drops empty entries.
        """
        if not raw or not raw.strip():
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _resolve_requested_output_files(self, *, strict: bool = True) -> None:
        """Resolve ``_requested_output_files`` against the file backend.

        Each requested path is checked as-is first, then by basename.
        When *strict* is ``True`` (inside the normal loop), missing files
        raise :class:`OutputFileNotFoundError` so the agent can retry.
        When *strict* is ``False`` (max-loops path), missing files are
        silently dropped because there are no retries left.
        """
        if not self._requested_output_files:
            return

        file_backend = self.sandbox_backend or self.file_store_backend
        if not file_backend:
            return

        resolved: list[str] = []
        file_not_found: list[str] = []
        for f in self._requested_output_files:
            basename = f.rsplit("/", 1)[-1]
            if file_backend.exists(f):
                resolved.append(f)
            elif f != basename and file_backend.exists(basename):
                resolved.append(basename)
            else:
                file_not_found.append(f)

        if file_not_found and strict:
            raise OutputFileNotFoundError(f"File not found: {file_not_found}.", recoverable=True)

        if file_not_found and not strict:
            logger.warning(
                f"Agent {self.name} - {self.id}: " f"max-loops output_files not found (skipped): {file_not_found}"
            )

        self._requested_output_files = resolved

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
        max_loops_prompt = self.system_prompt_manager.render_max_loops_prompt()

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

            raw_output_files = XMLParser.extract_first_tag_lxml(llm_final_attempt, ["output_files"])
            if raw_output_files is None:
                raw_output_files = XMLParser.extract_first_tag_regex(llm_final_attempt, ["output_files"])
            self._requested_output_files = self._parse_output_files_csv(raw_output_files or "")

        except Exception as e:
            logger.error(f"Max loops handler: Error during final answer extraction: {e}. Returning raw output.")
            final_answer = llm_final_attempt
            self._requested_output_files = []

        return f"{final_answer}"

    def _refresh_agent_state(self, loop_num: int) -> None:
        """
        Refresh the agent state with current values.

        Args:
            loop_num: Current loop iteration number.
        """
        self.state.update_loop(loop_num)
        todo_backend = None
        if self.sandbox_backend:
            todo_backend = self.sandbox_backend
        elif self.file_store.enabled and self.file_store.todo_enabled:
            todo_backend = self.file_store.backend

        if todo_backend:
            try:
                from dynamiq.nodes.tools.todo_tools import TODOS_FILE_PATH

                if todo_backend.exists(TODOS_FILE_PATH):
                    content = todo_backend.retrieve(TODOS_FILE_PATH)
                    data = json.loads(content.decode("utf-8"))
                    self.state.update_todos(data.get("todos", []))
            except Exception as e:
                logger.debug("Failed to load todo state (none or invalid): %s", e)

    def _init_prompt_blocks(self):
        """Initialize the prompt blocks required for the ReAct strategy."""
        # Generate inference-mode schemas
        if self.inference_mode == InferenceMode.FUNCTION_CALLING:
            self._tools = schema_generator.generate_function_calling_schemas(
                self.tools,
                self.delegation_allowed,
                self.sanitize_tool_name,
                response_format=self.response_format,
            )
        elif self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
            self._response_format = schema_generator.generate_structured_output_schemas(
                self.tools, self.sanitize_tool_name, self.delegation_allowed
            )

        # Build the entire prompt in one call
        model_name = getattr(self.llm, "model", None)
        response_format_schema = (
            schema_generator.unwrap_response_format(self.response_format) if self.response_format is not None else None
        )
        self.system_prompt_manager = AgentPromptManager(
            model_name=model_name,
            tool_description=self.tool_description,
            response_format_schema=response_format_schema,
        )

        self.system_prompt_manager.build_react_prompt(
            ReactPromptConfig(
                inference_mode=self.inference_mode,
                has_tools=bool(self.tools) or (self.skills.enabled and self.skills.source is not None),
                parallel_tool_calls_enabled=self.parallel_tool_calls_enabled,
                delegation_allowed=self.delegation_allowed,
                context_compaction_enabled=self.summarization_config.enabled,
                todo_management_enabled=(self.file_store.enabled and self.file_store.todo_enabled)
                or bool(self.sandbox_backend),
                sandbox_base_path=self.sandbox_backend.base_path if self.sandbox_backend else None,
                has_sub_agent_tools=any(isinstance(t, SubAgentTool) for t in self.tools),
                role=self.role,
                instructions=self.instructions,
            )
        )

    def _coerce_to_response_format(self, final_answer: Any) -> Any:
        """Parse the raw final answer into a dict matching ``response_format``.

        ``response_format`` is always a JSON schema dict (BaseModel input is
        converted to dict at validation time). FUNCTION_CALLING mode may pass an
        already-parsed dict; other modes pass a JSON string that needs parsing.
        """
        if self.response_format is None:
            return final_answer

        if isinstance(final_answer, str):
            try:
                final_answer = parse_llm_json_output(final_answer)
            except ValueError as e:
                raise ParsingError(f"Final answer is not valid JSON for response_format: {e}") from e

        if isinstance(final_answer, (dict, list)):
            return final_answer

        raise ParsingError(
            f"Cannot coerce final answer of type {type(final_answer).__name__} to the requested response_format."
        )

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

    def _is_tool_parallel_eligible(self, tool_name: str) -> bool:
        """Check if a tool is eligible for parallel execution based on its is_parallel_execution_allowed flag."""
        tool = self.tool_by_names.get(self.sanitize_tool_name(tool_name))
        return tool.is_parallel_execution_allowed if tool else False

    def _execute_tools(
        self,
        tools_data: list[dict[str, Any]],
        thought: str | None,
        loop_num: int,
        config: RunnableConfig,
        **kwargs,
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """
        Execute one or more tools and gather their results.

        Tools are split into two groups based on their ``is_parallel_execution_allowed`` flag:
        parallel-eligible tools run concurrently first, then sequential-only
        tools run one-by-one.

        Args:
            tools_data (list): List of dictionaries containing name and input for each tool
            thought: The agent's reasoning
            loop_num: Current loop iteration number
            config (RunnableConfig): Configuration for the runnable
            **kwargs: Additional arguments for tool execution

        Returns:
            tuple: (combined_observation, aggregated_files, ordered_results)
                where ordered_results is a list of per-tool result dicts in
                input order. Each entry contains tool_name, success, result,
                and files keys.
        """
        all_results: list[dict[str, Any]] = []

        if not tools_data:
            return "", {}, []

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
                        "result": error_message,
                        "files": [],
                    }
                )
                continue
            prepared_tools.append(
                {"order": idx, "name": tool_name, "input": tool_input, "thought": td.get("thought", "")}
            )

        def _execute_single_tool_to_result(tool_payload: dict[str, Any], **extra) -> dict[str, Any]:
            """Execute a single tool and wrap the result as a dict."""
            tool_result, tool_files, _, success, dependency = self._execute_single_tool(
                tool_payload["name"],
                tool_payload["input"],
                tool_payload.get("thought") or thought or "",
                loop_num,
                config,
                collect_dependency=True,
                **extra,
                **kwargs,
            )
            return {
                "order": tool_payload["order"],
                "tool_name": tool_payload["name"],
                "tool_run_id": tool_payload.get("tool_run_id"),
                "success": success,
                "result": tool_result,
                "files": tool_files,
                "dependency": dependency,
            }

        sequential_group: list[dict[str, Any]] = []

        if prepared_tools:

            if len(prepared_tools) == 1:
                all_results.append(_execute_single_tool_to_result(prepared_tools[0], update_run_depends=True))
            else:
                batch_tool_run_id = self._stream_batch_reasoning_event(
                    prepared_tools, thought, loop_num, config, **kwargs
                )

                parallel_group = [tp for tp in prepared_tools if self._is_tool_parallel_eligible(tp["name"])]
                sequential_group = [tp for tp in prepared_tools if not self._is_tool_parallel_eligible(tp["name"])]

                if sequential_group:
                    seq_names = [tp["name"] for tp in sequential_group]
                    logger.info(
                        f"Agent {self.name} - {self.id}: tools excluded from parallel execution "
                        f"(is_parallel_execution_allowed=False): {seq_names}"
                    )

                # Phase 1: run parallel-eligible tools concurrently
                if len(parallel_group) > 1:
                    max_workers = len(parallel_group)
                    with ContextAwareThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_map = {}
                        for tool_payload in parallel_group:
                            check_cancellation(config)
                            future = executor.submit(
                                _execute_single_tool_to_result,
                                tool_payload,
                                is_parallel=True,
                                update_run_depends=False,
                                tool_run_id=tool_payload["tool_run_id"],
                            )
                            future_map[future] = tool_payload

                        for future in as_completed(future_map.keys()):
                            check_cancellation(config)
                            all_results.append(future.result())
                elif len(parallel_group) == 1:
                    all_results.append(
                        _execute_single_tool_to_result(
                            parallel_group[0],
                            update_run_depends=False,
                            tool_run_id=parallel_group[0]["tool_run_id"],
                        )
                    )

                # Update run_depends from parallel results before sequential phase
                # so the first sequential tool depends on all parallel tools.
                if parallel_group:
                    parallel_deps = [r.get("dependency") for r in all_results if r.get("dependency")]
                    if parallel_deps:
                        self._run_depends = parallel_deps

                # Phase 2: run sequential-only tools one-by-one
                for tool_payload in sequential_group:
                    check_cancellation(config)
                    all_results.append(
                        _execute_single_tool_to_result(
                            tool_payload,
                            update_run_depends=True,
                            tool_run_id=tool_payload["tool_run_id"],
                        )
                    )

                self._stream_batch_tool_result_event(
                    batch_tool_run_id, prepared_tools, all_results, loop_num, config, **kwargs
                )

        observation_parts: list[str] = []
        aggregated_files: dict[str, Any] = {}

        ordered_results = sorted(all_results, key=lambda r: r.get("order", 0))

        for result in ordered_results:
            tool_name = result.get("tool_name", UNKNOWN_TOOL_NAME)
            result_content = result.get("result", "")
            success_status = "SUCCESS" if result.get("success") else "ERROR"
            observation_parts.append(f"--- {tool_name} has resulted in {success_status} ---\n{result_content}")

            self._merge_tool_files(aggregated_files, tool_name, result.get("files"))

        # For parallel-only batches, set run_depends from collected results.
        # For batches with sequential tools, run_depends is already correct
        # (chained by update_run_depends=True in the sequential loop).
        if not sequential_group:
            dependencies = [result.get("dependency") for result in ordered_results if result.get("dependency")]
            if dependencies:
                self._run_depends = dependencies

        combined_observation = "\n\n".join(observation_parts)

        return combined_observation, aggregated_files, ordered_results
