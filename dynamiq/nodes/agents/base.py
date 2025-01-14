import io
import json
import re
import textwrap
from datetime import datetime
from enum import Enum
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, TypeAdapter, ValidationError, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.memory import Memory, MemoryRetrievalStrategy
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    AgentUnknownToolException,
    InvalidActionException,
    ToolExecutionException,
)
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger


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


class AgentInputSchema(BaseModel):
    input: str = Field(..., description="Parameter to provide input to the agent.")
    files: list[io.BytesIO | bytes] = Field(default=[], description="Parameter to provide files to the agent.")

    user_id: str = Field(default=None, description="Parameter to provide user ID.")
    session_id: str = Field(default=None, description="Parameter to provide session ID.")
    metadata: dict | list = Field(default={}, description="Parameter to provide metadata.")
    chat_history: list[Message] = Field(default=[], description="Parameter to provide chat history.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Agent(Node):
    """Base class for an AI Agent that interacts with a Language Model and tools."""

    DEFAULT_INTRODUCTION: ClassVar[str] = (
        "You are a helpful AI assistant designed to assist users with various tasks and queries."
        "Your goal is to provide accurate, helpful, and friendly responses to the best of your abilities."
    )

    llm: Node = Field(..., description="LLM used by the agent.")
    group: NodeGroup = NodeGroup.AGENTS
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    tools: list[Node] = []
    files: list[io.BytesIO | bytes] | None = None
    name: str = "Agent"
    role: str | None = None
    max_loops: int = 1
    memory: Memory | None = Field(None, description="Memory node for the agent.")
    memory_retrieval_strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.BOTH
    verbose: bool = Field(False, description="Whether to print verbose logs.")

    _prompt_blocks: dict[str, str] = PrivateAttr(default_factory=dict)
    _prompt_variables: dict[str, Any] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[AgentInputSchema]] = AgentInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._intermediate_steps: dict[int, dict] = {}
        self._run_depends: list[dict] = []
        self._init_prompt_blocks()

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "llm": True,
            "tools": True,
            "memory": True,
            "files": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        data["tools"] = [tool.to_dict(**kwargs) for tool in self.tools]
        data["memory"] = self.memory.to_dict(**kwargs) if self.memory else None
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
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
            "introduction": self.DEFAULT_INTRODUCTION,
            "role": self.role or "",
            "date": datetime.now().strftime("%d %B %Y"),
            "tools": "{tool_description}",
            "files": "{file_description}",
            "instructions": "",
            "output_format": "",
            "relevant_information": "{relevant_memory}",
            "conversation_history": "{conversation_history}",
            "request": "User request: {input}",
        }
        self._prompt_variables = {
            "tool_description": self.tool_description,
            "file_description": self.file_description,
            "user_input": "",
            "conversation_history": "",
            "relevant_memory": "",
        }

    def add_block(self, block_name: str, content: str):
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
        EXCLUDED_KEYS = {"user_id", "session_id", "input", "metadata"}

        custom_metadata = input_data.get("metadata", {}).copy()
        custom_metadata.update({k: v for k, v in input_data.items() if k not in EXCLUDED_KEYS})

        # Add user and session IDs if present
        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")

        if user_id:
            custom_metadata["user_id"] = user_id
        if session_id:
            custom_metadata["session_id"] = session_id

        return custom_metadata

    def execute(self, input_data: AgentInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the agent with the given input data.
        """
        logger.info(f"Agent {self.name} - {self.id}: started with input {dict(input_data)}")
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        custom_metadata = self._prepare_metadata(dict(input_data))

        chat_history = input_data.chat_history

        if chat_history:
            try:
                chat_history = TypeAdapter(list[Message]).validate_python(chat_history)
                chat_history = self._retrieve_chat_history(chat_history)
                self._prompt_variables["conversation_history"] = chat_history

            except ValidationError as e:
                raise TypeError(f"Invalid chat history: {e}")

        if self.memory:
            self.memory.add(role=MessageRole.USER, content=input_data.input, metadata=custom_metadata)
            self._retrieve_memory(dict(input_data))

        files = input_data.files
        if files:
            self.files = files
            self._prompt_variables["file_description"] = self.file_description

        self._prompt_variables.update(dict(input_data))
        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        kwargs.pop("run_depends", None)

        result = self._run_agent(config=config, **kwargs)
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
            all_session_messages_str = self.memory.get_search_results_as_string(query=None, filters=user_filters)
            self._prompt_variables["conversation_history"] = all_session_messages_str

        else:
            user_query = input_data.get("input", "")

            if self.memory_retrieval_strategy == MemoryRetrievalStrategy.RELEVANT:
                relevant_memory = self.memory.get_search_results_as_string(query=user_query, filters=user_filters)
                self._prompt_variables["relevant_memory"] = relevant_memory

            elif self.memory_retrieval_strategy == MemoryRetrievalStrategy.ALL:
                all_messages = self.memory.get_all_messages_as_string()
                self._prompt_variables["conversation_history"] = all_messages

            elif self.memory_retrieval_strategy == MemoryRetrievalStrategy.BOTH:
                relevant_memory = self.memory.get_search_results_as_string(query=user_query, filters=user_filters)
                all_messages = self.memory.get_all_messages_as_string()
                self._prompt_variables["relevant_memory"] = relevant_memory
                self._prompt_variables["conversation_history"] = all_messages

    def _run_llm(self, prompt: str, config: RunnableConfig | None = None, **kwargs) -> str:
        """Runs the LLM with a given prompt and handles streaming or full responses."""
        try:
            llm_result = self.llm.run(
                input_data={},
                config=config,
                prompt=Prompt(messages=[Message(role="user", content=prompt)]),
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict()]
            if llm_result.status != RunnableStatus.SUCCESS:
                error_message = f"LLM '{self.llm.name}' failed: {llm_result.output.get('content')}"
                raise ValueError({error_message})

            return llm_result.output["content"]

        except Exception as e:
            raise e

    def stream_content(self, content: str, source: str, step: str, config: RunnableConfig | None = None, **kwargs):
        if self.streaming.by_tokens:
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

    def stream_response(self, content: str, source: str, step: str, config: RunnableConfig | None = None, **kwargs):
        response_for_stream = StreamChunk(
            choices=[StreamChunkChoice(delta=StreamChunkChoiceDelta(content=content, source=source, step=step))]
        )

        self.run_on_node_execute_stream(
            callbacks=config.callbacks,
            chunk=response_for_stream.model_dump(),
            **kwargs,
        )
        return content

    def _run_agent(self, config: RunnableConfig | None = None, **kwargs) -> str:
        """Runs the agent with the generated prompt and handles exceptions."""
        formatted_prompt = self.generate_prompt()
        try:
            llm_result = self._run_llm(formatted_prompt, config=config, **kwargs)
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

    def _parse_action(self, output: str) -> tuple[str | None, str | None]:
        """Parses the action and its input from the output string."""
        try:
            action_match = re.search(
                r"Action:\s*(.*?)\nAction Input:\s*(({\n)?.*?)(?:[^}]*$)",
                output,
                re.DOTALL,
            )
            if action_match:
                action = action_match.group(1).strip()
                action_input = action_match.group(2).strip()
                if "```json" in action_input:
                    action_input = action_input.replace("```json", "").replace("```", "").strip()

                action_input = json.loads(action_input)
                return action, action_input
            else:
                raise ActionParsingException()
        except Exception as e:
            raise ActionParsingException(
                (
                    f"Error {e}: Unable to parse action and action input."
                    "Please rewrite using the correct Action/Action Input format"
                    "with action input as a valid dictionary."
                    "Ensure all quotes are included."
                ),
                recoverable=True,
            )

    def _extract_final_answer(self, output: str) -> str:
        """Extracts the final answer from the output string."""
        match = re.search(r"Answer:\s*(.*)", output, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _get_tool(self, action: str) -> Node:
        """Retrieves the tool corresponding to the given action."""
        tool = self.tool_by_names.get(action)
        if not tool:
            raise AgentUnknownToolException(
                f"Unknown tool: {action}."
                "Use only available tools and provide only the tool's name in the action field. "
                "Do not include any additional reasoning. "
                "Please correct the action field or state that you cannot answer the question."
            )
        return tool

    def _run_tool(self, tool: Node, tool_input: str, config, **kwargs) -> Any:
        """Runs a specific tool with the given input."""
        if self.files:
            if tool.is_files_allowed is True:
                tool_input["files"] = self.files

        tool_result = tool.run(
            input_data=tool_input,
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
        prompt = ""
        for block, content in self._prompt_blocks.items():
            if block_names is None or block in block_names:
                if content:
                    formatted_content = content.format(**temp_variables)
                    prompt += f"{block.upper()}:\n{formatted_content}\n\n"

        prompt = textwrap.dedent(prompt)
        lines = prompt.splitlines()
        stripped_lines = [line.strip() for line in lines if line.strip()]
        prompt = "\n".join(stripped_lines)
        prompt = "\n".join(" ".join(line.split()) for line in prompt.split("\n"))
        return prompt


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
        prompt = self.generate_prompt(block_names=["plan"])
        llm_result = self._run_llm(prompt, config, **kwargs)
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(content=llm_result, step="reasoning", source=self.name, config=config, **kwargs)
        return llm_result

    def _assign(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'assign' action."""
        prompt = self.generate_prompt(block_names=["assign"])
        llm_result = self._run_llm(prompt, config, **kwargs)
        if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
            return self.stream_content(content=llm_result, step="reasoning", source=self.name, config=config, **kwargs)
        return llm_result

    def _final(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'final' action."""
        prompt = self.generate_prompt(block_names=["final"])
        llm_result = self._run_llm(prompt, config, **kwargs)
        if self.streaming.enabled:
            return self.stream_content(content=llm_result, step="answer", source=self.name, config=config, **kwargs)
        return llm_result
