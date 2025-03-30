import io
import json
import re
import textwrap
import types
from datetime import datetime
from enum import Enum
from typing import Any, Callable, ClassVar, Union, get_args, get_origin

from jinja2 import Template
from litellm import get_supported_openai_params, supports_function_calling
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.memory import Memory, MemoryRetrievalStrategy
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.constants import (
    AGENT_PROMPT_TEMPLATE,
    REACT_BLOCK_INSTRUCTIONS,
    REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING,
    REACT_BLOCK_INSTRUCTIONS_NO_TOOLS,
    REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT,
    REACT_BLOCK_OUTPUT_FORMAT,
    REACT_BLOCK_TOOLS_NO_FORMATS,
    REACT_BLOCK_XML_INSTRUCTIONS,
    REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS,
    REACT_MAX_LOOPS_PROMPT,
    TOOL_MAX_TOKENS,
    TYPE_MAPPING,
)
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    AgentUnknownToolException,
    InvalidActionException,
    MaxLoopsExceededException,
    RecoverableAgentException,
    ToolExecutionException,
)
from dynamiq.nodes.agents.utils import create_message_from_input, process_tool_output_for_agent
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.prompts import Message, MessageRole, Prompt, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingMode
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import deep_merge


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
    """
    Stores intermediate steps for debugging or chain-of-thought style logging.
    """

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
    """
    Defines optional parameters for each tool, or global defaults.
    Merged by the agent at runtime.
    """

    global_params: dict[str, Any] = Field(default_factory=dict, alias="global")
    by_name_params: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="by_name")
    by_id_params: dict[str, dict[str, Any]] = Field(default_factory=dict, alias="by_id")


class AgentInputSchema(BaseModel):
    """
    Standard input to the Agent: user text + optional images/files + optional user/session IDs, etc.
    """

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
        """
        This ensures the user has provided all required parameters that appear in the prompt message templates.
        """
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
    """
    A  Agent class that:
      - Behaves like a simple one-shot if no tools or max_loops <= 1
      - Behaves like ReAct multi-step if tools and max_loops > 1
      - Supports multiple inference modes (DEFAULT, XML, STRUCTURED_OUTPUT, FUNCTION_CALLING).
      - Integrates the memory, streaming, and error-handling from the original base agent code.
    """

    AGENT_PROMPT_TEMPLATE: ClassVar[str] = AGENT_PROMPT_TEMPLATE

    llm: Node = Field(..., description="LLM used by the agent.")
    group: NodeGroup = NodeGroup.AGENTS
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    tools: list[Node] = []
    files: list[io.BytesIO | bytes] | None = None
    images: list[str | bytes | io.BytesIO] = None
    name: str = "Agent"
    verbose: bool = Field(False, description="Whether to print verbose logs.")

    memory: Memory | None = Field(None, description="Memory node for the agent.")
    memory_limit: int = Field(100, description="Maximum number of messages to retrieve from memory")
    memory_retrieval_strategy: MemoryRetrievalStrategy | None = MemoryRetrievalStrategy.ALL

    max_loops: int = Field(default=15, ge=2)
    tool_output_max_length: int = TOOL_MAX_TOKENS
    tool_output_truncate_enabled: bool = True
    behaviour_on_max_loops: Behavior = Field(
        default=Behavior.RAISE,
        description="How to handle reaching max loops without final answer: RAISE or RETURN partial.",
    )

    inference_mode: InferenceMode = InferenceMode.DEFAULT
    format_schema: list = Field(default_factory=list)

    input_message: Message | VisionMessage = Field(default=Message(role=MessageRole.USER, content="{{input}}"))
    role: str | None = ""

    _prompt_blocks: dict[str, str] = PrivateAttr(default_factory=dict)
    _prompt_variables: dict[str, Any] = PrivateAttr(default_factory=dict)

    _intermediate_steps: dict[int, dict] = PrivateAttr(default_factory=dict)
    _run_depends: list[dict] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[AgentInputSchema]] = AgentInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prompt = Prompt(messages=[])
        self._init_prompt_blocks()

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """
        Initialize LLM and tool components.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

        for tool in self.tools:
            if tool.is_postponed_component_init:
                tool.init_components(connection_manager)
            tool.is_optimized_for_agents = True

    @model_validator(mode="after")
    def validate_inference_mode(self):
        """
        Check if the underlying model supports the chosen inference mode
        (function-calling or structured-output, etc.).
        """
        if self.inference_mode == InferenceMode.FUNCTION_CALLING:
            if not supports_function_calling(model=self.llm.model):
                raise ValueError(f"Model {self.llm.model} does not support function calling mode.")
        if self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
            params = get_supported_openai_params(model=self.llm.model)
            if "response_format" not in params:
                raise ValueError(f"Model {self.llm.model} does not support structured JSON output.")
        return self

    def get_context_for_input_schema(self) -> dict:
        """
        Provides context for input schema validation:
        used by AgentInputSchema to check required parameters in the prompt.
        """
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
        """
        Converts instance to a dictionary (including LLM, Tools, Memory, etc.).
        """
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        data["tools"] = [tool.to_dict(**kwargs) for tool in self.tools]
        data["memory"] = self.memory.to_dict(**kwargs) if self.memory else None
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
        if self.images:
            data["images"] = [{"name": getattr(f, "name", f"image_{i}")} for i, f in enumerate(self.images)]
        return data

    def _init_prompt_blocks(self):
        """Initializes prompt blocks and generates mode-specific schemas."""
        # Base blocks and variables
        self._prompt_blocks = {
            "date": "{date}",
            "tools": "{tool_description}",
            "files": "{file_description}",
            "instructions": "",
            "output_format": "",
            "context": "",
        }
        self._prompt_variables = {
            "tool_description": self.tool_description,
            "file_description": self.file_description,
            "date": datetime.now().strftime("%d %B %Y"),
            "tools_name": self.tool_names,
            "input_formats": self.generate_input_formats(self.tools) if self.tools else "",
        }

        if not self.tools:
            if self.inference_mode == InferenceMode.XML:
                instructions = REACT_BLOCK_XML_INSTRUCTIONS_NO_TOOLS
            else:
                instructions = REACT_BLOCK_INSTRUCTIONS_NO_TOOLS
            self._prompt_blocks["tools"] = ""
        else:
            self._prompt_blocks["output_format"] = REACT_BLOCK_OUTPUT_FORMAT

            if self.inference_mode == InferenceMode.XML:
                instructions = REACT_BLOCK_XML_INSTRUCTIONS
            elif self.inference_mode == InferenceMode.FUNCTION_CALLING:
                instructions = REACT_BLOCK_INSTRUCTIONS_FUNCTION_CALLING
                self._prompt_blocks["tools"] = REACT_BLOCK_TOOLS_NO_FORMATS
                self.generate_function_calling_schemas()
            elif self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
                instructions = REACT_BLOCK_INSTRUCTIONS_STRUCTURED_OUTPUT
                self.generate_structured_output_schemas()
            else:
                instructions = REACT_BLOCK_INSTRUCTIONS

        self._prompt_blocks["instructions"] = instructions

    def set_block(self, block_name: str, content: str):
        """Update a named block of the system prompt."""
        self._prompt_blocks[block_name] = content

    def set_prompt_variable(self, variable_name: str, value: Any):
        """Set or update a variable used in prompt generation."""
        self._prompt_variables[variable_name] = value

    def generate_prompt(self, block_names: list[str] | None = None, **kwargs) -> str:
        """
        Renders the final system prompt using AGENT_PROMPT_TEMPLATE
        and any selected blocks (instructions, tools, etc.).
        """
        temp_vars = self._prompt_variables.copy()
        temp_vars.update(kwargs)

        temp_vars.setdefault("tools_name", self.tool_names)
        temp_vars.setdefault("input_formats", self.generate_input_formats(self.tools) if self.tools else "")
        temp_vars.setdefault("tool_description", self.tool_description)
        temp_vars.setdefault("file_description", self.file_description)
        temp_vars.setdefault("date", datetime.now().strftime("%d %B %Y"))

        formatted_blocks = {}
        for block, content in self._prompt_blocks.items():
            if block_names is None or block in block_names:
                formatted_content = content.format(**temp_vars) if content else ""
                formatted_blocks[block] = formatted_content

        prompt = Template(self.AGENT_PROMPT_TEMPLATE).render(formatted_blocks).strip()
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
        return textwrap.dedent(prompt)

    @property
    def tool_description(self) -> str:
        """Returns a description of the tools available to the agent."""
        if not self.tools:
            return ""
        lines = []
        for tool in self.tools:
            desc = tool.description.strip() if tool.description else "No description."
            lines.append(f"{tool.name}:\n <{tool.name}_description>\n{desc}\n<\\{tool.name}_description>")
        return "\n".join(lines)

    @property
    def file_description(self) -> str:
        """Returns a description of the user files provided to the agent."""
        if not self.files:
            return ""
        file_str = "You can work with the following files:\n"
        for file in self.files:
            name = getattr(file, "name", "Unnamed file")
            description = getattr(file, "description", "No description")
            file_str += f"<file>: {name} - {description} <\\file>\n"
        return file_str

    @property
    def tool_names(self) -> str:
        """Comma-separated list of available tool names."""
        return ",".join([self.sanitize_tool_name(t.name) for t in self.tools])

    @property
    def tool_by_names(self) -> dict[str, Node]:
        """Map sanitized tool name => Node object."""
        return {self.sanitize_tool_name(t.name): t for t in self.tools}

    def sanitize_tool_name(self, s: str) -> str:
        """Remove spaces and special chars from tool name to produce a simpler key."""
        s = s.replace(" ", "-")
        return re.sub(r"[^a-zA-Z0-9_-]", "", s)

    def generate_function_calling_schemas(self):
        """Generate the schemas for function calling, one function per tool, plus a finalize function.
        With improved error handling for type conversion.
        """
        self.format_schema = []
        final_answer_schema = {
            "type": "function",
            "strict": True,
            "function": {
                "name": "provide_final_answer",
                "description": "Use this when you want to provide the final answer to the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string", "description": "Reasoning for finishing up now."},
                        "answer": {"type": "string", "description": "The final answer to the user request."},
                    },
                    "required": ["thought", "answer"],
                },
            },
        }
        self.format_schema.append(final_answer_schema)

        for tool in self.tools:
            try:
                properties = {}
                for name, field_def in tool.input_schema.model_fields.items():
                    if field_def.json_schema_extra and not field_def.json_schema_extra.get(
                        "is_accessible_to_agent", True
                    ):
                        continue

                    desc = field_def.description or "No description."

                    try:
                        param_type = self._filter_format_type(field_def.annotation)

                        if param_type in TYPE_MAPPING:
                            properties[name] = {"type": TYPE_MAPPING[param_type], "description": desc}
                        else:
                            properties[name] = {"type": "string", "description": desc}
                    except Exception as e:
                        logger.warning(f"Error processing field {name} type for {tool.name}: {e}. Using string type.")
                        properties[name] = {"type": "string", "description": f"{desc} (Error processing type)"}

                schema = {
                    "type": "function",
                    "function": {
                        "name": self.sanitize_tool_name(tool.name),
                        "description": tool.description[:1024] if tool.description else f"Tool: {tool.name}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": {
                                    "type": "string",
                                    "description": "Reasoning or short message about using this tool.",
                                },
                                "action_input": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": list(properties.keys()),
                                    "additionalProperties": False,
                                    "description": f"Inputs for tool {tool.name}",
                                },
                            },
                            "required": ["thought", "action_input"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }
                self.format_schema.append(schema)
            except Exception as e:
                logger.error(f"Error generating schema for tool {tool.name}: {e}")
                fallback_schema = {
                    "type": "function",
                    "function": {
                        "name": self.sanitize_tool_name(tool.name),
                        "description": f"Tool: {tool.name} (Error: {str(e)})",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "thought": {"type": "string", "description": "Reasoning for using this tool."},
                                "action_input": {
                                    "type": "object",
                                    "properties": {
                                        "input": {"type": "string", "description": f"Input for {tool.name}"}
                                    },
                                    "required": ["input"],
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["thought", "action_input"],
                        },
                        "strict": False,
                    },
                }
                self.format_schema.append(fallback_schema)

    def generate_structured_output_schemas(self):
        """Create a single JSON schema for structured output with {thought, action, action_input}."""
        self.format_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": "plan_next_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "required": ["thought", "action", "action_input"],
                        "properties": {
                            "thought": {"type": "string", "description": "Chain-of-thought or reasoning."},
                            "action": {
                                "type": "string",
                                "description": f"Which tool to call among [{self.tool_names}] or 'finish' to finalize.",
                            },
                            "action_input": {
                                "type": "string",
                                "description": "JSON-encoded string input for "
                                "the chosen tool or final answer if 'finish'.",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            }


    def generate_input_formats(self, tools: list[Node]) -> str:
        """Generate formatted input descriptions for each tool."""
        input_formats = []
        for tool in tools:
            params = []
            for name, field in tool.input_schema.model_fields.items():
                if not field.json_schema_extra or field.json_schema_extra.get("is_accessible_to_agent", True):
                    if get_origin(field.annotation) in (Union, types.UnionType):
                        type_str = str(field.annotation)
                    else:
                        type_str = getattr(field.annotation, "__name__", str(field.annotation))

                    description = field.description or "No description"
                    params.append(f"{name} ({type_str}): {description}")

            input_formats.append(f" - {self.sanitize_tool_name(tool.name)}\n \t* " + "\n\t* ".join(params))
        return "\n".join(input_formats)

    def _filter_format_type(self, param_type: Any) -> Any:
        """
        For function-calling: if we have Optional[SomeType] or Union, pick a single type.
        Handles edge cases with better error handling.
        """
        try:
            if param_type is None:
                return str

            origin = get_origin(param_type)
            if origin is Union:
                args = get_args(param_type)
                for arg in args:
                    if arg is not type(None) and arg is not None:
                        return self._filter_format_type(arg)
                return str

            if isinstance(param_type, type):
                if param_type not in TYPE_MAPPING:
                    return str

            return param_type
        except Exception as e:
            logger.warning(f"Error in _filter_format_type: {e}. Defaulting to string type.")
            return str

    def execute(
        self,
        input_data: AgentInputSchema,
        input_message: Message | VisionMessage | None = None,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Public method to run the agent with the given input_data:
         1) Validate or gather memory
         2) Build final prompt
         3) Run either single-shot or multi-step logic (internal _run_agent)
         4) Optionally store the final answer to memory
        """
        logger.info(f"Agent {self.name} - {self.id}: started with input {dict(input_data)}")
        self.reset_run_state()
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        custom_metadata = self._prepare_metadata(dict(input_data))

        input_message = create_message_from_input(dict(input_data))
        input_message = input_message or self.input_message
        input_message = input_message.format_message(**dict(input_data))

        use_memory = self.memory and (input_data.user_id or input_data.session_id)
        if use_memory:
            history_messages = self._retrieve_memory(dict(input_data))
            if history_messages:
                history_messages.insert(
                    0,
                    Message(
                        role=MessageRole.SYSTEM,
                        content="Below is the previous conversation history. Use it as context.",
                    ),
                )
            memory_content = input_message.content
            if isinstance(input_message, VisionMessage):
                text_parts = [c.text for c in input_message.content if isinstance(c, VisionMessageTextContent)]
                memory_content = " ".join(text_parts) if text_parts else "Image input"
            self.memory.add(role=MessageRole.USER, content=memory_content, metadata=custom_metadata)
        else:
            history_messages = None

        if self.role:
            self._prompt_blocks["context"] = Template(self.role).render(**dict(input_data))

        if input_data.files:
            self.files = input_data.files
            self._prompt_variables["file_description"] = self.file_description

        if input_data.tool_params:
            kwargs["tool_params"] = input_data.tool_params

        self._prompt_variables.update(dict(input_data))

        result_str = self._run_agent(
            input_message,
            history_messages,
            config=config,
            **kwargs,
        )

        if use_memory:
            self.memory.add(role=MessageRole.ASSISTANT, content=result_str, metadata=custom_metadata)

        execution_result = {
            "content": result_str,
            "intermediate_steps": self._intermediate_steps,
        }
        logger.info(f"Agent {self.name} - {self.id}: finished. Result: {str(result_str)[:200]}...")
        return execution_result

    def _prepare_metadata(self, input_data: dict) -> dict:
        """
        Prepare a custom metadata dict from user input, excluding certain fields.
        """
        EXCLUDED_KEYS = {"user_id", "session_id", "input", "metadata", "files", "images", "tool_params"}
        meta = input_data.get("metadata", {}).copy()
        meta.update({k: v for k, v in input_data.items() if k not in EXCLUDED_KEYS})

        if "files" in meta:
            del meta["files"]
        if "images" in meta:
            del meta["images"]
        if "tool_params" in meta:
            del meta["tool_params"]

        if input_data.get("user_id"):
            meta["user_id"] = input_data["user_id"]
        if input_data.get("session_id"):
            meta["session_id"] = input_data["session_id"]

        return meta

    def reset_run_state(self):
        """Clear any intermediate step state from a previous run."""
        self._intermediate_steps = {}
        self._run_depends = []

    def retrieve_conversation_history(
        self,
        user_query: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = None,
        strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.ALL,
    ) -> list[Message]:
        """
        Retrieve conversation messages from memory, using a retrieval strategy (ALL, RELEVANT, HYBRID).
        """
        if not self.memory or not (user_id or session_id):
            return []

        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if session_id:
            filters["session_id"] = session_id

        limit = limit or self.memory_limit
        if strategy == MemoryRetrievalStrategy.RELEVANT and not user_query:
            logger.warning("RELEVANT strategy selected but no user_query given => fallback to ALL")
            strategy = MemoryRetrievalStrategy.ALL

        conversation = self.memory.get_agent_conversation(
            query=user_query,
            limit=limit,
            filters=filters,
            strategy=strategy,
        )

        if self.verbose:
            logger.debug(
                f"Agent {self.name} retrieved {len(conversation)} messages using {strategy.value} strategy. "
                f"First message role: {conversation[0].role.value if conversation else 'None'}"
            )
        return conversation

    def _retrieve_memory(self, input_data: dict) -> list[Message]:
        """
        Helper that calls retrieve_conversation_history with the user's ID, session ID, etc.
        """
        user_id = input_data.get("user_id")
        session_id = input_data.get("session_id")
        user_query = input_data.get("input", "")
        return self.retrieve_conversation_history(
            user_query=user_query,
            user_id=user_id,
            session_id=session_id,
            strategy=self.memory_retrieval_strategy,
        )

    def _run_llm(self, messages: list[Message], config: RunnableConfig | None = None, **kwargs) -> Any:
        """
        Call the LLM node with the given messages. Handle streaming or normal responses.
        """
        try:
            if self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
                kwargs["response_format"] = {"type": "json_object"}
            llm_result = self.llm.run(
                input_data={},
                config=config,
                prompt=Prompt(messages=messages),
                run_depends=self._run_depends,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict()]
            if llm_result.status != RunnableStatus.SUCCESS:
                raise ValueError(f"LLM '{self.llm.name}' failed: {llm_result.output.get('content')}")

            return llm_result
        except Exception as e:
            raise e

    def _run_tool(self, tool: Node, tool_input: dict, config: RunnableConfig, **kwargs) -> Any:
        """
        Runs a given tool with merged parameters from (tool_input + tool_params).
        """
        if self.files and tool.is_files_allowed is True:
            tool_input["files"] = self.files

        merged_input = tool_input.copy() if isinstance(tool_input, dict) else {"input": tool_input}
        raw_tool_params = kwargs.get("tool_params", ToolParams())
        if isinstance(raw_tool_params, dict):
            raw_tool_params = ToolParams.model_validate(raw_tool_params)

        debug_info = []
        if raw_tool_params.global_params:
            self._apply_tool_parameters(merged_input, raw_tool_params.global_params, "global", debug_info)

        name_params = raw_tool_params.by_name_params.get(tool.name) or raw_tool_params.by_name_params.get(
            self.sanitize_tool_name(tool.name), {}
        )
        if name_params:
            self._apply_tool_parameters(merged_input, name_params, f"name:{tool.name}", debug_info)

        id_params = raw_tool_params.by_id_params.get(tool.id, {})
        if id_params:
            self._apply_tool_parameters(merged_input, id_params, f"id:{tool.id}", debug_info)

        if self.verbose and debug_info:
            logger.debug("\n".join(debug_info))

        try:
            tool_result = tool.run(
                input_data=merged_input,
                config=config,
                run_depends=self._run_depends,
                recoverable_error=True,
            )
            self._run_depends = [NodeDependency(node=tool).to_dict()]

            if tool_result.status != RunnableStatus.SUCCESS:
                error_msg = f"Tool '{tool.name}' failed: {tool_result.output}"
                if tool_result.output.get("recoverable", False):
                    raise ToolExecutionException(error_msg)
                else:
                    raise ValueError(error_msg)

            tool_content = tool_result.output.get("content")

            return process_tool_output_for_agent(
                content=tool_content,
                max_tokens=self.tool_output_max_length,
                truncate=self.tool_output_truncate_enabled,
            )
        except RecoverableAgentException as e:
            return f"{type(e).__name__}: {str(e)}"
        except Exception as e:
            logger.error(f"Error executing tool {tool.name}: {e}")
            return f"Error executing tool {tool.name}: {str(e)}"

    def _apply_tool_parameters(self, merged_input: dict, params: dict, source: str, debug_info: list):
        """
        Helper that merges dict `params` into the existing `merged_input`.
        """
        for key, value in params.items():
            if key in merged_input and isinstance(value, dict) and isinstance(merged_input[key], dict):
                # deep merge
                merged_input[key] = deep_merge(value, merged_input[key])
                debug_info.append(f" - from {source}: deep-merged {key}")
            else:
                merged_input[key] = value
                debug_info.append(f" - from {source}: set {key}={value}")

    def _run_agent(
        self,
        input_message: Message,
        history_messages: list[Message] | None,
        config: RunnableConfig | None,
        **kwargs,
    ) -> str:
        """
        If no tools or max_loops <= 1 => do single-shot approach.
        Otherwise => do multi-step approach with the chosen inference_mode.
        """
        if not self.tools:
            system_message = Message(role=MessageRole.SYSTEM, content=self.generate_prompt())
        else:
            system_message = Message(
                role=MessageRole.SYSTEM,
                content=self.generate_prompt(
                    tools_name=self.tool_names, input_formats=self.generate_input_formats(self.tools)
                ),
            )

        messages = [system_message]
        if history_messages:
            messages.extend(history_messages)
        messages.append(input_message)
        self._prompt.messages = messages

        if not self.tools:
            logger.info(f"Agent {self.name} - {self.id}: Using simple approach with no tools.")
            return self._run_simple(config=config, **kwargs)

        logger.info(
            f"Agent {self.name} - {self.id}: Using ReAct approach with max_loops={self.max_loops} "
            f"and {len(self.tools)} tools."
        )
        return self._run_react(config=config, **kwargs)

    def _run_simple(self, config: RunnableConfig | None, **kwargs) -> str:
        llm_result = self._run_llm(
            messages=self._prompt.messages,
            config=config,
            inference_mode=self.inference_mode,
            schema=self.format_schema,
            **kwargs,
        )

        if self.inference_mode == InferenceMode.FUNCTION_CALLING:
            tool_calls = llm_result.output.get("tool_calls", {})
            if tool_calls:
                for _, call_info in tool_calls.items():
                    if call_info["function"]["name"] == "provide_final_answer":
                        final_ans = call_info["function"]["arguments"]["answer"]
                        if self.streaming.enabled:
                            self.stream_content(final_ans, source=self.name, step="final_answer", config=config)
                        return final_ans
                raise ValueError("Expected 'provide_final_answer' call not found.")
            raise ValueError("No tool calls in FUNCTION_CALLING mode.")

        content = llm_result.output.get("content", "")
        if self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
            try:
                data = json.loads(content)
                final_ans = data["answer"]
                if self.streaming.enabled:
                    self.stream_content(final_ans, source=self.name, step="final_answer", config=config)
                return final_ans
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Invalid JSON or missing 'answer' in STRUCTURED_OUTPUT: {e}")
        elif self.inference_mode == InferenceMode.XML:
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            final_text = match.group(1).strip() if match else content.strip()
            if not final_text:
                final_text = content.strip()
        else:
            match = re.search(r"(?s)Answer:\s*(.*)$", content)
            final_text = match.group(1).strip() if match else content.strip()
            if not final_text:
                final_text = content.strip()

        if self.streaming.enabled:
            self.stream_content(final_text, source=self.name, step="final_answer", config=config)
        return final_text

    def _run_react(
        self,
        config: RunnableConfig | None,
        **kwargs,
    ) -> str:
        """
        Handles the ReAct (multi-step) execution mode with tools.
        Iterates up to max_loops, parsing actions and executing tools as needed.

        Args:
            config (RunnableConfig | None): Configuration for the agent run.
            **kwargs: Additional parameters for running the agent.

        Returns:
            str: The final answer provided by the agent.

        Raises:
            MaxLoopsExceededException: If max loops are reached and behavior is set to RAISE.
        """
        if self.inference_mode in [InferenceMode.DEFAULT, InferenceMode.XML]:
            self.llm.stop = ["Observation:", "\nObservation:"]

        for loop_idx in range(1, self.max_loops + 1):
            llm_result = self._run_llm(
                messages=self._prompt.messages,
                config=config,
                inference_mode=self.inference_mode,
                schema=self.format_schema,
                **kwargs,
            )

            output_text = llm_result.output.get("content", "")
            tool_calls = llm_result.output.get("tool_calls", {})
            logger.info(f"Agent {self.name} - {self.id}: Loop {loop_idx} - LLM output: {output_text[:200]}...")
            if tool_calls:
                logger.info(f"Agent {self.name} - {self.id}: Loop {loop_idx} - Tool calls: {str(tool_calls)[:200]}")

            self.tracing_intermediate(loop_idx, self._prompt.messages, output_text or str(tool_calls))

            if self.inference_mode == InferenceMode.FUNCTION_CALLING and tool_calls:
                for _, call_info in tool_calls.items():
                    func_name = call_info["function"]["name"]
                    func_args = call_info["function"]["arguments"]
                    function_call_text = json.dumps({"function": func_name, "arguments": func_args})

                    if func_name == "provide_final_answer":
                        final_ans = func_args["answer"]
                        self.tracing_final(loop_idx, final_ans, config, kwargs)
                        if self.streaming.enabled:
                            self.stream_content(final_ans, source=self.name, step="answer", config=config, **kwargs)
                        return final_ans

                    thought = func_args.get("thought", "")
                    action_input = func_args.get("action_input", {})
                    self.log_reasoning(thought, func_name, action_input, loop_idx)

                    if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                        chunk_content = {"thought": thought, "action": func_name, "action_input": action_input}
                        self.stream_content(
                            chunk_content, source=self.name, step=f"reasoning_{loop_idx}", config=config
                        )

                    try:
                        tool = self._get_tool(func_name)
                        tool_result = self._run_tool(tool, action_input, config, **kwargs)
                    except RecoverableAgentException as e:
                        tool_result = f"{type(e).__name__}: {str(e)}"

                    obs_msg = f"\nObservation: {tool_result}\n"
                    self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=function_call_text))
                    self._prompt.messages.append(Message(role=MessageRole.USER, content=obs_msg))

                    self._intermediate_steps[loop_idx]["model_observation"].update(
                        AgentIntermediateStepModelObservation(
                            tool_using=func_name,
                            tool_input=action_input,
                            tool_output=tool_result,
                            updated=function_call_text,
                        ).model_dump()
                    )
                continue

            if self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
                try:
                    data = json.loads(output_text)
                    thought = data["thought"]
                    action = data["action"]
                    action_input_str = data["action_input"]
                    if action == "finish":
                        self.tracing_final(loop_idx, action_input_str, config, kwargs)
                        if self.streaming.enabled:
                            self.stream_content(
                                action_input_str, source=self.name, step="answer", config=config, **kwargs
                            )
                        return action_input_str
                    action_input = json.loads(action_input_str)
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Parsing failed: {e}")
                    self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=output_text))
                    self._prompt.messages.append(
                        Message(role=MessageRole.USER, content=f"Your output format was invalid: {str(e)}. Please fix.")
                    )
                    continue

            elif self.inference_mode == InferenceMode.XML:
                if "<answer>" in output_text:
                    final_ans = self._extract_final_answer_xml(output_text)
                    self.tracing_final(loop_idx, final_ans, config, kwargs)
                    if self.streaming.enabled:
                        self.stream_content(final_ans, source=self.name, step="answer", config=config, **kwargs)
                    return final_ans
                thought, action, action_input = self.parse_xml_and_extract_info(output_text)

            else:
                if "Answer:" in output_text:
                    final_ans = self._extract_final_answer(output_text)
                    self.tracing_final(loop_idx, final_ans, config, kwargs)
                    if self.streaming.enabled:
                        self.stream_content(final_ans, source=self.name, step="answer", config=config, **kwargs)
                    return final_ans
                thought, action, action_input = self._parse_action(output_text)

            self.log_reasoning(thought, action, action_input, loop_idx)
            if self.streaming.enabled and self.streaming.mode == StreamingMode.ALL:
                chunk_content = {"thought": thought, "action": action, "action_input": action_input}
                self.stream_content(chunk_content, source=self.name, step=f"reasoning_{loop_idx}", config=config)

            if action:
                try:
                    tool = self._get_tool(action)
                    tool_result = self._run_tool(tool, action_input, config, **kwargs)
                except RecoverableAgentException as e:
                    tool_result = f"{type(e).__name__}: {str(e)}"

                obs_msg = f"\nObservation: {tool_result}\n"
                self._prompt.messages.append(Message(role=MessageRole.ASSISTANT, content=output_text))
                self._prompt.messages.append(Message(role=MessageRole.USER, content=obs_msg))

                self._intermediate_steps[loop_idx]["model_observation"].update(
                    AgentIntermediateStepModelObservation(
                        tool_using=action,
                        tool_input=action_input,
                        tool_output=tool_result,
                        updated=output_text,
                    ).model_dump()
                )

        logger.warning(f"Agent {self.name} - {self.id}: Reached maximum loops ({self.max_loops})")
        if self.behaviour_on_max_loops == Behavior.RAISE:
            error_message = (
                f"Agent {self.name} (ID: {self.id}) reached max loops ({self.max_loops}) without a final answer. "
                f"Last response: {self._prompt.messages[-1].content}"
            )
            raise MaxLoopsExceededException(error_message)
        else:
            logger.info(f"Agent {self.name} - {self.id}: Attempting final summarization call after max loops.")
            final_system_message = Message(role=MessageRole.SYSTEM, content=REACT_MAX_LOOPS_PROMPT)
            final_messages = [final_system_message] + self._prompt.messages[1:]
            final_llm_result = self._run_llm(messages=final_messages, config=config, schema=None, stop=None, **kwargs)
            final_output_text = final_llm_result.output.get("content", "")

            final_answer = self.parse_xml_content(final_output_text, "answer") or final_output_text.strip()
            if not final_answer:
                final_answer = (
                    "Could not finalize answer within the maximum number of steps. "
                    "Consider increasing max_loops or revising the input."
                )

            self.tracing_intermediate(self.max_loops + 1, final_messages, final_output_text)
            self.tracing_final(self.max_loops + 1, final_answer, config, kwargs)
            if self.streaming.enabled:
                self.stream_content(final_answer, source=self.name, step="answer", config=config, **kwargs)
            return final_answer

    def parse_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML-like tags."""
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def parse_xml_and_extract_info(self, text: str) -> dict[str, Any]:
        """Parse XML-like structure and extract action and action_input."""
        output_content = self.parse_xml_content(text, "output")
        thought = self.parse_xml_content(output_content, "thought")
        action = self.parse_xml_content(output_content, "action")
        action_input_text = self.parse_xml_content(output_content, "action_input")

        if not thought or not action or not action_input_text:
            missing = []
            if not thought:
                missing.append("thought")
            if not action:
                missing.append("action")
            if not action_input_text:
                missing.append("action_input")

            error_message = (
                f"Error: Missing required XML tags: {', '.join(missing)}. "
                "Your response must include complete <output>, <thought>, <action>, "
                "and <action_input> tags."
            )
            raise ActionParsingException(error_message, recoverable=True)

        try:
            action_input = json.loads(action_input_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error: Unable to parse action and action input due to invalid JSON formatting. {e}")
            error_message = (
                "Error: Unable to parse action and action input due to invalid JSON formatting. "
                "Multiline strings are not allowed in JSON unless properly escaped. "
                "Ensure all newlines (\\n), quotes, and special characters are escaped. "
                "For example:\n\n"
                "Correct:\n"
                "{\n"
                '  "key": "Line 1\\nLine 2",\n'
                '  "code": "print(\\"Hello, World!\\")"\n'
                "}\n\n"
                "Incorrect:\n"
                "{\n"
                '  "key": "Line 1\nLine 2",\n'
                '  "code": "print("Hello, World!")"\n'
                "}\n\n"
                f"JSON Parsing Error Details: {e}"
            )
            raise ActionParsingException(error_message, recoverable=True)

        return thought, action, action_input

    def extract_output_and_answer_xml(self, text: str) -> tuple[str, Any]:
        """Extract output and answer from XML-like structure."""
        output = self.parse_xml_content(text, "output")
        answer = self.parse_xml_content(text, "answer")
        return {"output": output, "answer": answer}

    def _extract_final_answer_xml(self, llm_output: str) -> str:
        """Extract the final answer from the LLM output."""
        final_answer = self.extract_output_and_answer_xml(llm_output)
        return final_answer["answer"]

    def _parse_thought(self, output: str) -> tuple[str | None, str | None]:
        """Extracts thought from the output string."""
        thought_match = re.search(
            r"Thought:\s*(.*?)Action",
            output,
            re.DOTALL,
        )

        if thought_match:
            return thought_match.group(1).strip()

        return ""

    def _parse_action(self, output: str) -> tuple[str | None, str | None]:
        """Parses the action, its input, and thought from the output string."""
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
                return self._parse_thought(output), action, action_input
            else:
                raise ActionParsingException()
        except Exception as e:
            raise ActionParsingException(
                (
                    f"Error {e}: Unable to parse thought, action and action input or thought and answer."
                    "Please rewrite using the correct Thought/Action/Action Input format"
                    "with action input as a valid dictionary."
                    "Or in case of direct answer or clarification, use Thought/Answer format."
                    "Ensure all quotes are included."
                ),
                recoverable=True,
            )

    def _extract_final_answer(self, text: str) -> str:
        """
        Basic approach: find "Answer:" and return all text after it.
        """
        match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else text

    def _get_tool(self, action: str) -> Node:
        """
        Return the Node tool matching the sanitized action name.
        Raise if not found.
        """
        tool = self.tool_by_names.get(self.sanitize_tool_name(action))
        if not tool:
            raise AgentUnknownToolException(
                f"Unknown tool: '{action}'. Use only the available tools: {list(self.tool_by_names.keys())}"
            )
        return tool

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
        Stream data to the config callbacks. If by_tokens is True, split by spaces; else send as one chunk.
        """
        if (by_tokens is None and self.streaming.by_tokens) or by_tokens:
            return self.stream_by_tokens(content=content, source=source, step=step, config=config, **kwargs)
        return self.stream_response(content=content, source=source, step=step, config=config, **kwargs)

    def stream_by_tokens(
        self,
        content: str | dict,
        source: str,
        step: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ):
        """
        If content is str, split by spaces and stream. If dict, just stream once.
        """
        if isinstance(content, dict):
            return self.stream_response(content, source, step, config, **kwargs)

        tokens = content.split(" ")
        final_response = []
        for token in tokens:
            final_response.append(token)
            chunk = StreamChunk(
                choices=[
                    StreamChunkChoice(
                        delta=StreamChunkChoiceDelta(
                            content=" " + token,
                            source=source,
                            step=step,
                        )
                    )
                ]
            )
            self.run_on_node_execute_stream(
                callbacks=config.callbacks if config else [],
                chunk=chunk.model_dump(),
                **kwargs,
            )
        return " ".join(final_response)

    def stream_response(
        self,
        content: str | dict,
        source: str,
        step: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ):
        chunk = StreamChunk(
            choices=[
                StreamChunkChoice(
                    delta=StreamChunkChoiceDelta(
                        content=content,
                        source=source,
                        step=step,
                    )
                )
            ]
        )
        self.run_on_node_execute_stream(
            callbacks=config.callbacks if config else [],
            chunk=chunk.model_dump(),
            **kwargs,
        )
        return content

    def tracing_final(self, loop_num, final_answer, config, kwargs):
        """
        Logs the final answer for a given loop number.

        Args:
            loop_num (int): The loop iteration number.
            final_answer (str): The final answer to log.
            config (RunnableConfig): Configuration for the run.
            kwargs: Additional parameters.
        """
        self._intermediate_steps[loop_num]["final_answer"] = final_answer

    def tracing_intermediate(self, loop_num, formatted_prompt, llm_generated_output):
        """
        Logs intermediate steps for a given loop number.

        Args:
            loop_num (int): The loop iteration number.
            formatted_prompt (list[Message]): The prompt messages sent to the LLM.
            llm_generated_output (str): The raw output from the LLM.
        """
        self._intermediate_steps[loop_num] = AgentIntermediateStep(
            input_data={"prompt": formatted_prompt},
            model_observation=AgentIntermediateStepModelObservation(initial=llm_generated_output),
        ).model_dump()

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
