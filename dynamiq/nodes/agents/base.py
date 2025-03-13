import io
import re
import textwrap
from datetime import datetime
from enum import Enum
from typing import Any, Callable, ClassVar

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.memory import Memory, MemoryRetrievalStrategy
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.exceptions import AgentUnknownToolException, InvalidActionException, ToolExecutionException
from dynamiq.nodes.agents.utils import create_message_from_input
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, MessageRole, Prompt, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import deep_merge

AGENT_PROMPT_TEMPLATE = """
You are AI powered assistant.
{%- if date %}
- Always up-to-date with the latest technologies and best practices.
- Current date: {{date}}
{%- endif %}

{%- if instructions %}
# PRIMARY INSTRUCTIONS
{{instructions}}
{%- endif %}

{%- if tools %}
# AVAILABLE TOOLS
{{tools}}
{%- endif %}

{%- if files %}
# USER UPLOADS
Files provided by user: {{files}}
{%- endif %}

{%- if output_format %}
# RESPONSE FORMAT
{{output_format}}
{%- endif %}

{%- if context %}
# AGENT PERSONA & STYLE
(This section defines how the assistant presents information - its personality, tone, and style.
These style instructions enhance but should never override or contradict the PRIMARY INSTRUCTIONS above.)
{{context}}
{%- endif %}
"""


class StreamChunkChoiceDelta(BaseModel):
    """Delta model for content chunks."""
    content: str | dict
    source: str
    step: str


class StreamChunkChoice(BaseModel):
    """Stream chunk choice model."""

    delta: StreamChunkChoiceDelta


class StreamChunk(BaseModel):
    """Model for streaming chunks with choices containing delta updates."""

    choices: list[StreamChunkChoice]


class AgentStatus(str, Enum):
    """Represents the status of an agent's execution."""

    SUCCESS = "success"
    FAIL = "fail"


class AgentIntermediateStepModelObservation(BaseModel):
    initial: str | dict | None = None
    tool_using: str | dict | None = None
    tool_input: str | dict | None = None
    tool_output: Any = None
    updated: str | dict | None = None


class AgentIntermediateStep(BaseModel):
    input_data: str | dict
    model_observation: AgentIntermediateStepModelObservation
    final_answer: str | dict | None = None


class ToolParams(BaseModel):
    global_params: dict[str, Any] = Field(default_factory=dict, alias="global")
    by_name_params: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="by_name")
    by_id_params: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="by_id")


class AgentInputSchema(BaseModel):
    input: str = Field(default="", description="Text input for the agent.")
    images: list[str | bytes | io.BytesIO] = Field(
        default=None, description="Image inputs (URLs, bytes, or file objects)."
    )
    files: list[io.BytesIO | bytes] = Field(default=None, description="Parameter to provide files to the agent.")

    user_id: str = Field(default=None, description="Parameter to provide user ID.")
    session_id: str = Field(default=None, description="Parameter to provide session ID.")
    metadata: dict | list = Field(default={}, description="Parameter to provide metadata.")

    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    tool_params: ToolParams | None = Field(
        default_factory=ToolParams,
        description=(
            "Structured parameters for tools. Use 'global_params' for all tools, "
            "'by_name' for tool names, or 'by_id' for tool IDs. "
            "Values are dictionaries merged with tool inputs."
        ),
        is_accessible_to_agent=False,
    )

    @field_validator("tool_params", mode="before")
    @classmethod
    def handle_empty_tool_params(cls, v):
        if v == "" or v is None:
            return ToolParams()
        return v

    @model_validator(mode="after")
    def validate_input_fields(self, context):
        ctx_msg = context.context.get("role") or ""
        messages = [
            context.context.get("input_message"),
            Message(role=MessageRole.USER, content=ctx_msg),
        ]
        required_parameters = Prompt(messages=messages).get_required_parameters()

        parameters = self.model_dump()
        provided_parameters = set(parameters.keys())

        if not required_parameters.issubset(provided_parameters):
            raise ValueError(
                f"Error: Invalid parameters were provided. Expected: {required_parameters}. "
                f"Got: {provided_parameters}"
            )

        none_elements = []
        for key, value in parameters.items():
            if key in required_parameters and value is None:
                none_elements.append(key)

        if none_elements:
            raise ValueError(f"Error: None was provided for parameters {none_elements}.")

        return self


class Agent(Node):
    """Base class for an AI Agent that interacts with a Language Model and tools."""

    AGENT_PROMPT_TEMPLATE: ClassVar[str] = AGENT_PROMPT_TEMPLATE

    llm: Node = Field(..., description="LLM used by the agent.")
    group: NodeGroup = NodeGroup.AGENTS
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    tools: list[Node] = []
    files: list[io.BytesIO | bytes] | None = None
    images: list[str | bytes | io.BytesIO] = None
    name: str = "Agent"
    max_loops: int = 1
    memory: Memory | None = Field(None, description="Memory node for the agent.")
    memory_retrieval_strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.BOTH
    verbose: bool = Field(False, description="Whether to print verbose logs.")

    input_message: Message | VisionMessage = Message(role=MessageRole.USER, content="{{input}}")
    role: str | None = ""
    _prompt_blocks: dict[str, str] = PrivateAttr(default_factory=dict)
    _prompt_variables: dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[AgentInputSchema]] = AgentInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._intermediate_steps: dict[int, dict] = {}
        self._run_depends: list[dict] = []
        self._prompt = Prompt(messages=[])
        self._init_prompt_blocks()

    @model_validator(mode="after")
    def validate_input_fields(self):
        if self.input_message:
            self.input_message.role = MessageRole.USER

        return self

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        return {"input_message": self.input_message, "role": self.role}

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "llm": True,
            "tools": True,
            "memory": True,
            "files": True,
            "images": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        data["tools"] = [tool.to_dict(**kwargs) for tool in self.tools]
        data["memory"] = self.memory.to_dict(**kwargs) if self.memory else None
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
        if self.images:
            data["images"] = [{"name": getattr(f, "name", f"image_{i}")} for i, f in enumerate(self.images)]
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize components for the manager and agents.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

        for tool in self.tools:
            if tool.is_postponed_component_init:
                tool.init_components(connection_manager)
            tool.is_optimized_for_agents = True

    def sanitize_tool_name(self, s: str):
        """Sanitize tool name to follow [^a-zA-Z0-9_-]."""
        s = s.replace(" ", "-")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", s)
        return sanitized

    def _init_prompt_blocks(self):
        """Initializes default prompt blocks and variables."""
        self._prompt_blocks = {
            "date": "{date}",
            "tools": "{tool_description}",
            "files": "{file_description}",
            "instructions": "",
        }
        self._prompt_variables = {
            "tool_description": self.tool_description,
            "file_description": self.file_description,
            "date": datetime.now().strftime("%d %B %Y"),
        }

    def set_block(self, block_name: str, content: str):
        """Adds or updates a prompt block."""
        self._prompt_blocks[block_name] = content

    def set_prompt_variable(self, variable_name: str, value: Any):
        """Sets or updates a prompt variable."""
        self._prompt_variables[variable_name] = value

    def _retrieve_chat_history(self, messages: list[Message]) -> str:
        """Converts a list of messages to a formatted string."""
        return "\n".join([f"**{msg.role.value}:** {msg.content}" for msg in messages])

    def _prepare_metadata(self, input_data: dict) -> dict:
        """
        Prepare metadata from input data.

        Args:
            input_data (dict): Input data containing user information

        Returns:
            dict: Processed metadata
        """
        EXCLUDED_KEYS = {"user_id", "session_id", "input", "metadata", "files", "tool_params"}

        custom_metadata = input_data.get("metadata", {}).copy()
        custom_metadata.update({k: v for k, v in input_data.items() if k not in EXCLUDED_KEYS})

        if "files" in custom_metadata:
            del custom_metadata["files"]
        if "tool_params" in custom_metadata:
            del custom_metadata["tool_params"]

        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")

        if user_id:
            custom_metadata["user_id"] = user_id
        if session_id:
            custom_metadata["session_id"] = session_id

        return custom_metadata

    def execute(
        self,
        input_data: AgentInputSchema,
        input_message: Message | VisionMessage | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Executes the agent with the given input data.
        """
        logger.info(f"Agent {self.name} - {self.id}: started with input {dict(input_data)}")
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        custom_metadata = self._prepare_metadata(dict(input_data))

        input_message = create_message_from_input(dict(input_data))

        input_message = input_message or self.input_message
        input_message = input_message.format_message(**dict(input_data))

        if self.memory:
            history_messages = self._retrieve_memory(dict(input_data))
            if len(history_messages) > 0:
                history_messages.insert(
                    0,
                    Message(
                        role=MessageRole.SYSTEM,
                        content="Below is the previous conversation history. "
                        "Use this context to inform your response.",
                    ),
                )
            if isinstance(input_message, Message):
                memory_content = input_message.content
            else:
                text_parts = [
                    content.text for content in input_message.content if isinstance(content, VisionMessageTextContent)
                ]
                memory_content = " ".join(text_parts) if text_parts else "Image input"
            self.memory.add(role=MessageRole.USER, content=memory_content, metadata=custom_metadata)
        else:
            history_messages = None

        if self.role:
            self._prompt_blocks["context"] = Template(self.role).render(**dict(input_data))

        files = input_data.files
        if files:
            self.files = files
            self._prompt_variables["file_description"] = self.file_description

        if input_data.tool_params:
            kwargs["tool_params"] = input_data.tool_params

        self._prompt_variables.update(dict(input_data))
        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        kwargs.pop("run_depends", None)

        result = self._run_agent(input_message, history_messages, config=config, **kwargs)

        if self.memory:
            self.memory.add(role=MessageRole.ASSISTANT, content=result, metadata=custom_metadata)

        execution_result = {
            "content": result,
            "intermediate_steps": self._intermediate_steps,
        }
        logger.info(f"Node {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

        return execution_result

    @staticmethod
    def _make_filters(user_id: str | None, session_id: str | None) -> dict | None:
        """Build a filter dictionary based on user_id and session_id."""
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if session_id:
            filters["session_id"] = session_id

        return filters if filters else None

    def _retrieve_memory(self, input_data):
        """
        Retrieves memory based on the selected strategy:
        - RELEVANT: retrieves relevant memory based on the user input
        - ALL: retrieves all messages in the memory
        - BOTH: retrieves both relevant memory and all messages
        """
        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")

        user_filters = self._make_filters(user_id, session_id)

        if session_id:
            messages = self.memory.search(query=None, filters=user_filters)

        else:
            user_query = input_data.get("input", "")

            if self.memory_retrieval_strategy == MemoryRetrievalStrategy.RELEVANT:
                messages = self.memory.search(query=user_query, filters=user_filters)

            elif self.memory_retrieval_strategy == MemoryRetrievalStrategy.ALL:
                messages = self.memory.get_all()

            elif self.memory_retrieval_strategy == MemoryRetrievalStrategy.BOTH:
                relevant_history_messages = self.memory.search(query=user_query, filters=user_filters)
                messages_all = self.memory.get_all()
                seen = set()
                messages = []
                for msg in relevant_history_messages + messages_all:
                    if msg.content not in seen:
                        messages.append(msg)
                        seen.add(msg.content)

        return messages

    def _run_llm(self, messages: list[Message | VisionMessage], config: RunnableConfig | None = None, **kwargs) -> str:
        """Runs the LLM with a given prompt and handles streaming or full responses."""
        try:
            llm_result = self.llm.run(
                input_data={},
                config=config,
                prompt=Prompt(messages=messages),
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict()]
            if llm_result.status != RunnableStatus.SUCCESS:
                error_message = f"LLM '{self.llm.name}' failed: {llm_result.output.get('content')}"
                raise ValueError({error_message})

            return llm_result

        except Exception as e:
            raise e

    def stream_content(
        self,
        content: str | dict,
        source: str,
        step: str,
        config: RunnableConfig | None = None,
        by_tokens: bool | None = None,
        **kwargs,
    ) -> str | dict:
        """
        Streams data.

        Args:
            content (str | dict): Data that will be streamed.
            source (str): Source of the content.
            step (str): Description of the step.
            by_tokens (Optional[bool]): Determines whether to stream content by tokens or not.
                If None it is determined based on StreamingConfig. Defaults to None.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            str | dict: Streamed data.
        """
        if (by_tokens is None and self.streaming.by_tokens) or by_tokens:
            return self.stream_by_tokens(content=content, source=source, step=step, config=config, **kwargs)
        return self.stream_response(content=content, source=source, step=step, config=config, **kwargs)

    def stream_by_tokens(self, content: str, source: str, step: str, config: RunnableConfig | None = None, **kwargs):
        """Streams the input content to the callbacks."""
        tokens = content.split(" ")
        final_response = []
        for token in tokens:
            final_response.append(token)
            token_with_prefix = " " + token
            token_for_stream = StreamChunk(
                choices=[
                    StreamChunkChoice(delta=StreamChunkChoiceDelta(content=token_with_prefix, source=source, step=step))
                ]
            )
            self.run_on_node_execute_stream(
                callbacks=config.callbacks,
                chunk=token_for_stream.model_dump(),
                **kwargs,
            )
        return " ".join(final_response)

    def stream_response(
        self, content: str | dict, source: str, step: str, config: RunnableConfig | None = None, **kwargs
    ):
        response_for_stream = StreamChunk(
            choices=[StreamChunkChoice(delta=StreamChunkChoiceDelta(content=content, source=source, step=step))]
        )

        self.run_on_node_execute_stream(
            callbacks=config.callbacks,
            chunk=response_for_stream.model_dump(),
            **kwargs,
        )
        return content

    def _run_agent(
        self,
        input_message: Message | VisionMessage,
        history_messages: list[Message] | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str:
        """Runs the agent with the generated prompt and handles exceptions."""
        formatted_prompt = self.generate_prompt()
        system_message = Message(role=MessageRole.SYSTEM, content=formatted_prompt)
        if history_messages:
            self._prompt.messages = [system_message, *history_messages, input_message]
        else:
            self._prompt.messages = [system_message, input_message]

        try:
            llm_result = self._run_llm(self._prompt.messages, config=config, **kwargs).output["content"]
            self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=llm_result))

            if self.streaming.enabled:
                return self.stream_content(
                    content=llm_result,
                    source=self.name,
                    step="answer",
                    config=config,
                    **kwargs,
                )
            return llm_result

        except Exception as e:
            raise e

    def _extract_final_answer(self, output: str) -> str:
        """Extracts the final answer from the output string."""
        match = re.search(r"Answer:\s*(.*)", output, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _get_tool(self, action: str) -> Node:
        """Retrieves the tool corresponding to the given action."""
        tool = self.tool_by_names.get(self.sanitize_tool_name(action))
        if not tool:
            raise AgentUnknownToolException(
                f"Unknown tool: {action}."
                "Use only available tools and provide only the tool's name in the action field. "
                "Do not include any additional reasoning. "
                "Please correct the action field or state that you cannot answer the question."
            )
        return tool

    def _apply_parameters(self, merged_input: dict, params: dict, source: str, debug_info: list = None):
        """Apply parameters from the specified source to the merged input."""
        if debug_info is None:
            debug_info = []
        for key, value in params.items():
            if key in merged_input and isinstance(value, dict) and isinstance(merged_input[key], dict):
                merged_nested = merged_input[key].copy()
                merged_input[key] = deep_merge(value, merged_nested)
                debug_info.append(f"  - From {source}: Merged nested {key}")
            else:
                merged_input[key] = value
                debug_info.append(f"  - From {source}: Set {key}={value}")

    def _run_tool(self, tool: Node, tool_input: dict, config, **kwargs) -> Any:
        """Runs a specific tool with the given input."""
        if self.files:
            if tool.is_files_allowed is True:
                tool_input["files"] = self.files

        merged_input = tool_input.copy() if isinstance(tool_input, dict) else {"input": tool_input}
        raw_tool_params = kwargs.get("tool_params", ToolParams())
        tool_params = (
            ToolParams.model_validate(raw_tool_params) if isinstance(raw_tool_params, dict) else raw_tool_params
        )

        if tool_params:
            debug_info = []
            if self.verbose:
                debug_info.append(f"Tool parameter merging for {tool.name} (ID: {tool.id}):")
                debug_info.append(f"Starting with input: {merged_input}")

            # 1. Apply global parameters (lowest priority)
            global_params = tool_params.global_params
            if global_params:
                self._apply_parameters(merged_input, global_params, "global", debug_info)

            # 2. Apply parameters by tool name (medium priority)
            name_params = tool_params.by_name_params.get(tool.name, {}) or tool_params.by_name_params.get(
                self.sanitize_tool_name(tool.name), {}
            )
            if name_params:
                self._apply_parameters(merged_input, name_params, f"name:{tool.name}", debug_info)

            # 3. Apply parameters by tool ID (highest priority)
            id_params = tool_params.by_id_params.get(tool.id, {})
            if id_params:
                self._apply_parameters(merged_input, id_params, f"id:{tool.id}", debug_info)

            if self.verbose and debug_info:
                logger.debug("\n".join(debug_info))

        tool_result = tool.run(
            input_data=merged_input,
            config=config,
            run_depends=self._run_depends,
            **(kwargs | {"recoverable_error": True}),
        )
        self._run_depends = [NodeDependency(node=tool).to_dict()]
        if tool_result.status != RunnableStatus.SUCCESS:
            error_message = f"Tool '{tool.name}' failed: {tool_result.output}"
            if tool_result.output["recoverable"]:
                raise ToolExecutionException({error_message})
            else:
                raise ValueError({error_message})
        return tool_result.output["content"]

    @property
    def tool_description(self) -> str:
        """Returns a description of the tools available to the agent."""
        return (
            "\n".join(
                [
                    f"{tool.name}:\n <{tool.name}_description>\n{tool.description.strip()}\n<\\{tool.name}_description>"
                    for tool in self.tools
                ]
            )
            if self.tools
            else ""
        )

    @property
    def file_description(self) -> str:
        """Returns a description of the files available to the agent."""
        if self.files:
            file_description = "You can work with the following files:\n"
            for file in self.files:
                name = getattr(file, "name", "Unnamed file")
                description = getattr(file, "description", "No description")
                file_description += f"<file>: {name} - {description} <\\file>\n"
            return file_description
        return ""

    @property
    def tool_names(self) -> str:
        """Returns a comma-separated list of tool names available to the agent."""
        return ",".join([self.sanitize_tool_name(tool.name) for tool in self.tools])

    @property
    def tool_by_names(self) -> dict[str, Node]:
        """Returns a dictionary mapping tool names to their corresponding Node objects."""
        return {self.sanitize_tool_name(tool.name): tool for tool in self.tools}

    def reset_run_state(self):
        """Resets the agent's run state."""
        self._intermediate_steps = {}
        self._run_depends = []

    def generate_prompt(self, block_names: list[str] | None = None, **kwargs) -> str:
        """Generates the prompt using specified blocks and variables."""
        temp_variables = self._prompt_variables.copy()
        temp_variables.update(kwargs)

        formatted_prompt_blocks = {}
        for block, content in self._prompt_blocks.items():
            if block_names is None or block in block_names:

                formatted_content = content.format(**temp_variables)
                if content:
                    formatted_prompt_blocks[block] = formatted_content

        prompt = Template(self.AGENT_PROMPT_TEMPLATE).render(formatted_prompt_blocks).strip()
        prompt = self._clean_prompt(prompt)
        return textwrap.dedent(prompt)

    def _clean_prompt(self, prompt_text):
        cleaned = re.sub(r"\n{3,}", "\n\n", prompt_text)
        return cleaned.strip()


class AgentManagerInputSchema(BaseModel):
    action: str = Field(..., description="Parameter to provide action to the manager")
    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_action(self, context):
        action = self.action
        if not action or action not in context.context.get("actions"):
            error_message = (
                f"Invalid or missing action: {action}. "  # nosec B608: Static message construction, not SQL-related.
                "Please select an action "
                f"from {context.context.get('actions')}"  # nosec B608: Static message construction, not SQL-related.
            )
            raise InvalidActionException(error_message)
        return self


class AgentManager(Agent):
    """Manager class that extends the Agent class to include specific actions."""

    _actions: dict[str, Callable] = PrivateAttr(default_factory=dict)
    name: str = "Agent Manager"
    input_schema: ClassVar[type[AgentManagerInputSchema]] = AgentManagerInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions()

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        data = super().to_dict(**kwargs)
        data["_actions"] = {
            k: getattr(action, "__name__", str(action))
            for k, action in self._actions.items()
        }
        return data

    def _init_actions(self):
        """Initializes the default actions for the manager."""
        self._actions = {"plan": self._plan, "assign": self._assign, "final": self._final}

    def add_action(self, name: str, action: Callable):
        """Adds a custom action to the manager."""
        self._actions[name] = action

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        return {"actions": list(self._actions.keys())}

    def execute(
        self, input_data: AgentManagerInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the manager agent with the given input data and action."""
        logger.info(f"Agent {self.name} - {self.id}: started with INPUT DATA:\n{input_data}")
        self.reset_run_state()
        config = config or RunnableConfig()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        action = input_data.action

        self._prompt_variables.update(dict(input_data))

        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        kwargs.pop("run_depends", None)
        _result_llm = self._actions[action](config=config, **kwargs)
        result = {"action": action, "result": _result_llm}

        execution_result = {
            "content": result,
            "intermediate_steps": self._intermediate_steps,
        }
        logger.info(f"Agent {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

        return execution_result

    def _plan(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'plan' action."""
        prompt = self._prompt_blocks.get("plan").format(**self._prompt_variables, **kwargs)

        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(
                content=llm_result, step="manager_planning", source=self.name, config=config, by_tokens=False, **kwargs
            )

        return llm_result

    def _assign(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'assign' action."""
        prompt = self._prompt_blocks.get("assign").format(**self._prompt_variables, **kwargs)
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(
                content=llm_result, step="manager_assigning", source=self.name, config=config, by_tokens=False, **kwargs
            )
        return llm_result

    def _final(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'final' action."""
        prompt = self._prompt_blocks.get("final").format(**self._prompt_variables, **kwargs)
        llm_result = self._run_llm(
            [Message(role=MessageRole.USER, content=prompt)], config, by_tokens=False, **kwargs
        ).output["content"]
        if self.streaming.enabled:
            return self.stream_content(
                content=llm_result,
                step="manager_final_output",
                source=self.name,
                config=config,
                by_tokens=False,
                **kwargs,
            )
        return llm_result
