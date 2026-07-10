import base64
import binascii
import io
import json
import mimetypes
import re
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, ClassVar, Union
from urllib.parse import unquote_to_bytes
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.memory import Memory, MemoryRetrievalStrategy, MemorySaveMode
from dynamiq.memory.long_term import LongTermMemoryConfig
from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.nodes.agents.checkpoint import DEFAULT_HISTORY_OFFSET, AgentIterativeCheckpointMixin
from dynamiq.nodes.agents.exceptions import AgentUnknownToolException, InvalidActionException, ToolExecutionException
from dynamiq.nodes.agents.prompts.manager import AgentPromptManager
from dynamiq.nodes.agents.prompts.templates import AGENT_PROMPT_TEMPLATE
from dynamiq.nodes.agents.shared_session import SandboxSharingScope, SharedSession, _shared_session
from dynamiq.nodes.agents.utils import (
    TOOL_MAX_TOKENS,
    ToolCacheEntry,
    ToolOutputSandboxPersistenceConfig,
    bytes_to_data_url,
    convert_bytesio_to_file_info,
    extract_message_text,
    is_image_file,
    is_video_file,
    process_tool_output_with_sandbox_persistence,
)
from dynamiq.nodes.llms import BaseLLM
from dynamiq.nodes.node import NodeDependency, ensure_config
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.nodes.tools.file_tools import FileListTool, FileReadTool, FileSearchTool, FileWriteTool
from dynamiq.nodes.tools.mcp import MCPServer
from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME, ParallelToolCallsTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.nodes.tools.skills_tool import SkillsTool
from dynamiq.nodes.tools.todo_tools import TodoWriteTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import (
    Message,
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageFileContent,
    VisionMessageFileData,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.sandboxes.base import Sandbox, SandboxConfig
from dynamiq.skills.config import SkillsConfig
from dynamiq.skills.registries.dynamiq import Dynamiq
from dynamiq.skills.types import SkillMetadata
from dynamiq.skills.utils import ingest_skills_into_sandbox, normalize_sandbox_skills_base_path
from dynamiq.storages.file.base import FileStore, FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.types.cancellation import CanceledException, check_cancellation
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import deep_merge

# Per-call tool overlay (e.g. LTM tools bound to a request's user_id); isolated
# per thread / per asyncio task via ContextVar.
_run_extra_tools: ContextVar[list["Node"] | None] = ContextVar("dynamiq_agent_run_extra_tools", default=None)

# Per-call overlay of shared-sandbox-backed tools (later task); isolated per
# thread / per asyncio task via ContextVar, same pattern as `_run_extra_tools`.
_shared_sandbox_tools: ContextVar[list["Node"] | None] = ContextVar("dynamiq_shared_sandbox_tools", default=None)


class StreamChunkChoiceDelta(BaseModel):
    """Delta model for content chunks."""
    content: str | dict
    source: str
    step: str

    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        """Ensure source is always a string."""
        if not isinstance(v, str):
            raise ValueError(f"source must be a string, got {type(v).__name__}: {v}")
        return v

    def _recursive_serialize(self, obj, key_path: str = "", index: int = None):
        """Recursively serialize an object, converting any BytesIO objects to FileInfo objects."""
        if isinstance(obj, io.BytesIO):
            return convert_bytesio_to_file_info(obj, key_path, index).model_dump()

        elif isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                new_key_path = f"{key_path}.{k}" if key_path else k
                result[k] = self._recursive_serialize(v, new_key_path)
            return result

        elif isinstance(obj, list):
            result = []

            for i, item in enumerate(obj):
                new_key_path = f"{key_path}[{i}]" if key_path else f"item_{i}"
                result.append(self._recursive_serialize(item, new_key_path, i))
            return result

        else:
            return obj

    @model_serializer
    def serialize_content(self):
        """Serialize content dict, converting any BytesIO objects to base64 strings while preserving key structure."""
        if self.content is None or not isinstance(self.content, dict):
            return {"content": self.content, "source": self.source, "step": self.step}

        serialized_content = self._recursive_serialize(self.content)

        result = {
            "content": serialized_content,
            "source": self.source,
            "step": self.step,
        }

        return result


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


class ToolParams(BaseModel):
    global_params: dict[str, Any] = Field(default_factory=dict, alias="global")
    by_name_params: dict[str, Union[dict[str, Any], "ToolParams"]] = Field(default_factory=dict, alias="by_name")
    by_id_params: dict[str, Union[dict[str, Any], "ToolParams"]] = Field(default_factory=dict, alias="by_id")


class AgentInputSchema(BaseModel):
    input: str = Field(default="", description="Text input for the agent.")
    images: list[str | bytes | io.BytesIO] | None = Field(
        default=None, description="Image inputs (URLs, bytes, or file objects)."
    )
    videos: list[str | bytes | io.BytesIO] | None = Field(
        default=None,
        description="Video inputs (URLs, bytes, or file objects). Only usable with an LLM "
        "that supports native video input (see BaseLLM.is_video_input_supported).",
    )
    files: list[io.BytesIO | bytes] | None = Field(
        default=None,
        description="List of file paths to pass to the agent.",
        json_schema_extra={"map_from_storage": True, "is_accessible_to_agent": False},
    )

    user_id: str | None = Field(default=None, description="Parameter to provide user ID.")
    session_id: str | None = Field(default=None, description="Parameter to provide session ID.")
    metadata: dict = Field(default={}, description="Parameter to provide metadata in key-value pairs.")

    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    tool_params: ToolParams | None = Field(
        default_factory=ToolParams,
        description=(
            "Structured parameters for tools. Use 'global_params' for all tools, "
            "'by_name' for tool names, or 'by_id' for tool IDs. "
            "Values are dictionaries merged with tool inputs."
        ),
        json_schema_extra={"is_accessible_to_agent": False},
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
        messages = [Message(role=MessageRole.USER, content=ctx_msg)]
        if message := context.context.get("input_message"):
            messages.append(message)

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


class Agent(AgentIterativeCheckpointMixin, Node):
    """Base class for an AI Agent that interacts with a Language Model and tools."""

    AGENT_PROMPT_TEMPLATE: ClassVar[str] = AGENT_PROMPT_TEMPLATE

    llm: BaseLLM = Field(..., description="LLM used by the agent.")
    group: NodeGroup = NodeGroup.AGENTS
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=3600))
    tools: list[Node] = []
    files: list[io.BytesIO | bytes] | None = None
    is_files_allowed: bool = True
    images: list[str | bytes | io.BytesIO] = None
    videos: list[str | bytes | io.BytesIO] = None
    name: str = "Agent"
    max_loops: int = 1
    tool_output_max_length: int = TOOL_MAX_TOKENS
    tool_output_truncate_enabled: bool = True
    tool_output_sandbox_persistence: ToolOutputSandboxPersistenceConfig = Field(
        default_factory=ToolOutputSandboxPersistenceConfig,
        description="Configuration for saving large tool outputs to sandbox files.",
    )
    delegation_allowed: bool = Field(
        default=False,
        description="Allow returning a child agent tool's output directly via delegate_final flag.",
    )
    parallel_tool_calls_enabled: bool = Field(
        default=False,
        description="Enable multi-tool execution in a single step. "
        "When True, the agent can call multiple tools in parallel.",
    )
    memory: Memory | None = Field(None, description="Memory node for the agent.")
    memory_limit: int = Field(100, description="Maximum number of messages to retrieve from memory")
    memory_retrieval_strategy: MemoryRetrievalStrategy | None = MemoryRetrievalStrategy.ALL
    long_term_memory: LongTermMemoryConfig | None = Field(
        default=None,
        description=(
            "Long-term, fact-shaped, user-scoped memory config (enabled + backend + tools). "
            "Accessed via remember/recall tools. Independent of `memory` (short-term messages)."
        ),
    )
    verbose: bool = Field(False, description="Whether to print verbose logs.")
    file_store: FileStoreConfig = Field(
        default_factory=lambda: FileStoreConfig(enabled=False, backend=InMemoryFileStore()),
        description="Configuration for file storage used by the agent.",
    )
    sandbox: SandboxConfig | None = Field(default=None, description="Configuration for sandbox used by the agent.")
    share_sandbox_with_subagents: bool = Field(
        default=False,
        description="When enabled, subagents (SubAgentTool) share this agent's sandbox "
        "instead of each provisioning their own. Each subagent gets an isolated working directory.",
    )
    sandbox_sharing_scope: SandboxSharingScope = Field(
        default=SandboxSharingScope.ALL,
        description=(
            "When share_sandbox_with_subagents is on, which subagents join the shared sandbox: "
            "ALL (default) routes every subagent onto the shared sandbox, overriding a subagent's "
            "own sandbox; AUGMENT only shares to subagents that bring no sandbox of their own."
        ),
    )
    skills: SkillsConfig = Field(
        default_factory=SkillsConfig,
        description="Skills config. When enabled and source registry is set, skills are on (Dynamiq or FileSystem).",
    )

    input_message: Message | VisionMessage | None = None
    role: str | None = Field(
        default=None,
        description="""Agent basic instructions.
            Can be used to provide additional context or instructions to the agent.
            Accepts Jinja templates to provide additional parameters.""",
    )
    instructions: str | None = Field(
        default=None,
        description="Additional operational instructions appended to the operational instructions block.",
    )
    description: str | None = Field(default=None, description="Short human-readable description of the agent.")
    _mcp_servers: list[MCPServer] = PrivateAttr(default_factory=list)
    _excluded_tool_ids: set[str] = PrivateAttr(default_factory=set)
    _own_sandbox_tool_ids: set[str] = PrivateAttr(default_factory=set)
    _tool_cache: dict[ToolCacheEntry, Any] = {}
    _history_offset: int = PrivateAttr(
        default=DEFAULT_HISTORY_OFFSET,
    )
    # Original user input message preserved from compaction and used for memory fallback.
    _pinned_input: Message | VisionMessage | None = PrivateAttr(default=None)
    system_prompt_manager: AgentPromptManager = Field(default_factory=AgentPromptManager)
    _current_call_context: dict[str, Any] | None = PrivateAttr(default=None)
    _sandbox_is_shared: bool = PrivateAttr(default=False)
    # A borrowed per-agent view of an owner's shared sandbox; when set it is this
    # agent's effective sandbox_backend for the whole run (tools, uploads, output).
    _shared_sandbox_view: Sandbox | None = PrivateAttr(default=None)
    # Loop progress and pending-tool-call state are declared on AgentIterativeCheckpointMixin.

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[AgentInputSchema]] = AgentInputSchema
    _json_schema_fields: ClassVar[list[str]] = ["role", "description"]

    @classmethod
    def _generate_json_schema(
        cls, llms: dict[type[BaseLLM], list[str]] = {}, tools=list[type[Node]], **kwargs
    ) -> dict[str, Any]:
        """
        Generates full json schema for Agent with provided llms and tools.
        This schema is designed for compatibility with the WorkflowYamlParser,
        containing enough partial information to instantiate an Agent.
        Parameters name to be included in the schema are either defined in the _json_schema_fields class variable or
        passed via the fields parameter.

        It generates a schema using the provided LLMs and tools.

        Args:
            llms (dict[type[BaseLLM], list[str]]): Available llm providers and models.
            tools (list[type[Node]]): List of tools.

        Returns:
            dict[str, Any]: Generated json schema.
        """
        schema = super()._generate_json_schema(**kwargs)
        schema["properties"]["llm"] = {
            "anyOf": [
                {
                    "type": "object",
                    **llm._generate_json_schema(models=models, fields=["model", "temperature", "max_tokens"]),
                }
                for llm, models in llms.items()
            ],
            "additionalProperties": False,
        }

        schema["properties"]["tools"] = {
            "type": "array",
            "items": {"anyOf": [{"type": "object", **tool._generate_json_schema()} for tool in tools]},
        }

        schema["required"] += ["tools", "llm"]
        return schema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from dynamiq.nodes.tools.agent_tool import SubAgentTool
        self._run_depends: list[dict] = []
        self._prompt = Prompt(messages=[])
        # Added for backward compatibility with old Agent tools

        self.tools = [
            (
                SubAgentTool(agent=t, name=t.name, description=t.description or "")
                if isinstance(t, Agent) and not isinstance(t, SubAgentTool)
                else t
            )
            for t in self.tools
        ]

        expanded_tools = []
        for tool in self.tools:
            if isinstance(tool, MCPServer):
                self._mcp_servers.append(tool)
                subtools = tool.get_mcp_tools()
                expanded_tools.extend(subtools)
                self._excluded_tool_ids.update(subtool.id for subtool in subtools)
            else:
                expanded_tools.append(tool)

        self.tools = expanded_tools
        tools_sandbox = self._resolve_tools_sandbox()
        if self.file_store_backend and tools_sandbox:
            raise ValueError("file_store and sandbox cannot both be enabled for an Agent at the same time")

        if tools_sandbox:
            # Add sandbox tools when sandbox is enabled (not serialized; recreated from sandbox config on load)
            sandbox_tools = tools_sandbox.get_tools(llm=self.llm)
            self._excluded_tool_ids.update(t.id for t in sandbox_tools)
            self._own_sandbox_tool_ids.update(t.id for t in sandbox_tools)
            self.tools.extend(sandbox_tools)

        elif self.file_store_backend:
            # Add file tools when file store is enabled
            self.tools.extend(
                [
                    FileReadTool(file_store=self.file_store_backend, llm=self.llm),
                    FileSearchTool(file_store=self.file_store_backend),
                    FileListTool(file_store=self.file_store_backend),
                ]
            )
            if self.file_store.agent_file_write_enabled:
                self.tools.append(FileWriteTool(file_store=self.file_store_backend))

        if self.parallel_tool_calls_enabled:
            inference_mode = getattr(self, "inference_mode", None)
            use_native_parallel = inference_mode == InferenceMode.FUNCTION_CALLING
            if not use_native_parallel:
                self.tools = [t for t in self.tools if t.name != PARALLEL_TOOL_NAME]
                self.tools.append(ParallelToolCallsTool())

        if self._skills_should_init():
            self._init_skills()
        self._init_prompt_blocks()
        if self._skills_should_init():
            self._apply_skills_to_prompt()

    @model_validator(mode="after")
    def validate_input_fields(self):
        if self.input_message:
            self.input_message.role = MessageRole.USER

        return self

    def get_context_for_input_schema(self) -> dict:
        """Provides context for input schema that is required for proper validation."""
        role_for_validation = self.role or ""
        if role_for_validation and (
            "{% raw %}" not in role_for_validation and "{% endraw %}" not in role_for_validation
        ):
            role_for_validation = f"{{% raw %}}{role_for_validation}{{% endraw %}}"
        return {"input_message": self.input_message, "role": role_for_validation}

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "llm": True,
            "tools": True,
            "memory": True,
            "long_term_memory": True,
            "files": True,
            "images": True,
            "videos": True,
            "file_store": True,
            "skills": True,
            "sandbox": True,
            "system_prompt_manager": True,  # Runtime state container, not serializable
        }

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary."""
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)

        tools_to_serialize = [t for t in self.tools if t.id not in self._excluded_tool_ids]
        data["tools"] = [tool.to_dict(**kwargs) for tool in tools_to_serialize]
        data["tools"] = data["tools"] + [mcp_server.to_dict(**kwargs) for mcp_server in self._mcp_servers]

        data["memory"] = self.memory.to_dict(**kwargs) if self.memory else None
        data["long_term_memory"] = self.long_term_memory.to_dict(**kwargs) if self.long_term_memory else None
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
        if self.images:
            data["images"] = [{"name": getattr(f, "name", f"image_{i}")} for i, f in enumerate(self.images)]
        if self.videos:
            data["videos"] = [{"name": getattr(f, "name", f"video_{i}")} for i, f in enumerate(self.videos)]

        data["file_store"] = self.file_store.to_dict(**kwargs) if self.file_store else None
        data["sandbox"] = self.sandbox.to_dict(**kwargs) if self.sandbox else None
        data["skills"] = self.skills.to_dict(**kwargs)

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

        if self.long_term_memory and self.long_term_memory.backend.embedder.is_postponed_component_init:
            self.long_term_memory.backend.embedder.init_components(connection_manager)

        self._ensure_skills_ingested_for_sandbox()

    def _ensure_skills_ingested_for_sandbox(self) -> None:
        """When skills source is Dynamiq with sandbox_skills_base_path and sandbox is enabled, ingest skills at init."""
        if not self.sandbox_backend or not self._skills_should_init():
            return
        source = self.skills.source
        if source is None:
            return

        if not isinstance(source, Dynamiq) or not source.sandbox_skills_base_path:
            return
        try:
            if hasattr(self.sandbox_backend, "_ensure_sandbox"):
                self.sandbox_backend._ensure_sandbox()
            ingest_skills_into_sandbox(
                self.sandbox_backend,
                source,
                sandbox_skills_base_path=source.sandbox_skills_base_path,
            )
            logger.info("Agent %s: skills ingested into sandbox at init", self.name)
        except Exception as e:
            logger.warning("Agent %s: skills ingestion into sandbox failed: %s", self.name, e)

    def sanitize_tool_name(self, s: str):
        """Sanitize tool name to follow [^a-zA-Z0-9_-]."""
        s = s.replace(" ", "-")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", s)
        return sanitized

    def _init_prompt_blocks(self):
        """Initializes default prompt blocks and variables."""
        model_name = getattr(self.llm, "model", None)

        self.system_prompt_manager = AgentPromptManager(model_name=model_name, tool_description=self.tool_description)
        self.system_prompt_manager.setup_for_base_agent()

    def _skills_should_init(self) -> bool:
        """True if skills support should be initialized (enabled and source set)."""
        return self.skills.enabled and self.skills.source is not None

    def _init_skills(self) -> None:
        """Add SkillsTool to self.tools so it is included in function-calling and structured-output schemas."""

        source = self.skills.source
        if source is None:
            logger.warning("Skills config missing or invalid (source required); skipping skills init")
            return
        skills_tool = SkillsTool(skill_registry=source)
        self.tools.append(skills_tool)
        self._excluded_tool_ids.add(skills_tool.id)

    def _apply_skills_to_prompt(self) -> None:
        """Set skills block and tool_description on the prompt manager after _init_prompt_blocks()."""
        source = self.skills.source
        if source is None:
            return
        metadata = self.skills.get_skills_metadata()
        sandbox_base = normalize_sandbox_skills_base_path(getattr(source, "sandbox_skills_base_path", None))
        skills_summary = self._format_skills_summary(
            metadata, sandbox_skills_base_path=sandbox_base if sandbox_base else None
        )
        self.system_prompt_manager.set_block("skills", skills_summary)
        self.system_prompt_manager.set_initial_variable("tool_description", self.tool_description)
        if sandbox_base:
            self.system_prompt_manager.set_initial_variable("sandbox_skills_base_path", sandbox_base)
        logger.info(
            f"Agent {self.name} - {self.id}: initialized with {len(metadata)} skills "
            f"(source={source.__class__.__name__})"
        )

    def _format_skills_summary(self, metadata: list[SkillMetadata], sandbox_skills_base_path: str | None = None) -> str:
        """Format skills summary for prompt.

        When sandbox_skills_base_path is set (caller must pass an already-normalized path or None),
        each line includes the path to read the skill in the sandbox so the agent can go straight
        to SandboxShellTool without calling SkillsTool list.
        """
        if not metadata:
            return ""

        base = sandbox_skills_base_path or ""
        lines = []
        for skill in metadata:
            if base:
                skill_path = f"{base}/{skill.name}/SKILL.md"
                lines.append(f"- **{skill.name}**: {skill.description} — read: `{skill_path}`")
            else:
                lines.append(f"- **{skill.name}**: {skill.description}")
        return "\n".join(lines)

    def set_block(self, block_name: str, content: str):
        """Adds or updates a prompt block."""
        self.system_prompt_manager.set_block(block_name, content)

    def set_prompt_variable(self, variable_name: str, value: Any):
        """Sets or updates a prompt variable."""
        self.system_prompt_manager.set_variable(variable_name, value)

    def _prepare_metadata(self, input_data: AgentInputSchema) -> dict:
        """
        Prepare metadata from input data.

        Args:
            input_data: Agent input schema containing user information

        Returns:
            dict: Processed metadata
        """
        custom_metadata = input_data.metadata.copy()

        # Clean up any leaked fields
        if "files" in custom_metadata:
            del custom_metadata["files"]
        if "images" in custom_metadata:
            del custom_metadata["images"]
        if "videos" in custom_metadata:
            del custom_metadata["videos"]
        if "tool_params" in custom_metadata:
            del custom_metadata["tool_params"]

        if input_data.user_id:
            custom_metadata["user_id"] = input_data.user_id
        if input_data.session_id:
            custom_metadata["session_id"] = input_data.session_id

        return custom_metadata

    def _clear_todos_file(self) -> None:
        """Delete the persisted todos file and reset in-memory todo state.

        Invoked unconditionally from execute()'s finally block.
        Failures are logged but never propagate — cleanup must not break the agent run.
        """
        try:
            for tool in self.tools:
                if isinstance(tool, TodoWriteTool):
                    tool.clear()
            if getattr(self, "state", None) is not None:
                self.state.update_todos([])
        except Exception as e:
            logger.warning(f"Agent {self.name} - {self.id}: todo cleanup failed: {e}")

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
        # Convert to dict only for logging (to avoid logging BytesIO objects)
        log_data = input_data.model_dump()
        if log_data.get("images"):
            log_data["images"] = [f"image_{i}" for i in range(len(log_data["images"]))]
        if log_data.get("videos"):
            log_data["videos"] = [f"video_{i}" for i in range(len(log_data["videos"]))]
        if log_data.get("files"):
            log_data["files"] = [f"file_{i}" for i in range(len(log_data["files"]))]

        logger.info(f"Agent {self.name} - {self.id}: started with input {log_data}")
        self.reset_run_state()

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        custom_metadata = self._prepare_metadata(input_data)
        self._current_call_context = {
            "user_id": input_data.user_id,
            "session_id": input_data.session_id,
            "metadata": custom_metadata,
        }

        input_message = input_message or self.input_message or Message(role=MessageRole.USER, content=input_data.input)
        # Convert to dict for format_message, excluding fields that are unsafe for templates
        # (binary data like files/images, complex objects like tool_params, and input which is already handled)
        standard_fields = set(AgentInputSchema.model_fields.keys())
        extra_fields = input_data.model_dump(exclude=standard_fields)
        if extra_fields:
            input_message = input_message.format_message(**extra_fields)
        else:
            input_message = input_message.model_copy()
        input_message.static = True

        use_memory = self.memory and (input_data.user_id or input_data.session_id)

        ltm_tools = self._build_long_term_memory_tools(input_data)
        if ltm_tools:
            logger.info(
                "Agent %s - %s: attached %d long-term memory tools (%s)",
                self.name,
                self.id,
                len(ltm_tools),
                ", ".join(t.name for t in ltm_tools),
            )
        # Always set — a sub-agent without LTM would otherwise inherit the
        # parent's overlay via `ContextAwareThreadPoolExecutor`.
        ltm_token = _run_extra_tools.set(ltm_tools)
        # Session/borrow setup lives INSIDE the try so the finally always resets the ContextVars and
        # releases any borrowed view, even if setup raises. Tokens stay None until each set succeeds.
        shared_session_token = None
        sandbox_overlay_token = None
        try:
            shared_session_token = self._maybe_enter_shared_session(kwargs)
            # Borrowers resolve a per-agent view of the shared sandbox at run time (both factory- and
            # initialized-mode subagents). Always set the overlay (even to None) so a nested subagent
            # does not inherit this agent's overlay via ContextAwareThreadPoolExecutor.
            sandbox_overlay_token = _shared_sandbox_tools.set(self._maybe_borrow_shared_sandbox())
            if use_memory:
                history_messages = self._retrieve_memory(input_data)
                if len(history_messages) > 0:
                    history_messages.insert(
                        0,
                        Message(
                            role=MessageRole.SYSTEM,
                            content="Below is the previous conversation history. "
                            "Use this context to inform your response.",
                            static=True,
                        ),
                    )
            else:
                history_messages = None

            images = []
            videos = []
            for item in input_data.images or []:
                if not isinstance(item, str) and is_video_file(item):
                    videos.append(item)
                else:
                    images.append(item)
            for item in input_data.videos or []:
                if not isinstance(item, str) and is_image_file(item):
                    images.append(item)
                else:
                    videos.append(item)

            other_files = []
            for file in input_data.files or []:
                if is_image_file(file):
                    images.append(file)
                elif is_video_file(file):
                    videos.append(file)
                else:
                    other_files.append(file)

            media_url_references = []

            if videos and not self.llm.is_video_input_supported:
                logger.warning(
                    "Agent %s - %s: LLM '%s' does not support video input; treating %d video "
                    "attachment(s) as generic files instead.",
                    self.name, self.id, self.llm.model, len(videos),
                )
                for index, video in enumerate(videos):
                    self._append_unsupported_media_attachment(
                        video, other_files, media_url_references, media_type="video", index=index
                    )
                videos = []

            if images and not self.llm.is_vision_supported:
                logger.warning(
                    "Agent %s - %s: LLM '%s' does not support vision input; treating %d image "
                    "attachment(s) as generic files instead.",
                    self.name, self.id, self.llm.model, len(images),
                )
                for index, image in enumerate(images):
                    self._append_unsupported_media_attachment(
                        image, other_files, media_url_references, media_type="image", index=index
                    )
                images = []

            if media_url_references:
                input_message = self._inject_attached_media_references_into_message(
                    input_message, media_url_references
                )

            if other_files:
                normalized_files = self._ensure_named_files(other_files)
                file_paths = []
                if self.sandbox_backend:
                    file_paths = self._upload_files_to_sandbox(normalized_files)
                else:
                    if not self.file_store_backend:
                        self._setup_in_memory_file_store_and_tools()
                    if self.file_store_backend:
                        file_paths = self._upload_files_to_file_store(normalized_files)
                input_message = self._inject_attached_files_into_message(
                    input_message, normalized_files, file_paths=file_paths
                )

            if images or videos:
                input_message = self._inject_attached_media_into_message(input_message, images=images, videos=videos)

            if input_data.tool_params:
                kwargs["tool_params"] = input_data.tool_params

            self.system_prompt_manager.update_variables(dict(input_data))
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)

            try:
                result = self._run_agent(input_message, history_messages, config=config, **kwargs)
            except CanceledException:
                if use_memory:
                    try:
                        self._save_history_to_memory(custom_metadata)
                    except Exception as save_error:
                        logger.error(
                            f"Agent {self.name} - {self.id}: failed to save history to memory "
                            f"after cancel: {save_error}",
                        )
                        try:
                            self._append_user_input_to_memory(custom_metadata)
                        except Exception as save_error2:
                            logger.error(
                                f"Agent {self.name} - {self.id}: also failed to save user input "
                                f"after cancel: {save_error2}",
                            )
                raise
            except Exception:
                if use_memory:
                    try:
                        self._append_user_input_to_memory(custom_metadata)
                    except Exception as save_error:
                        logger.error(
                            f"Agent {self.name} - {self.id}: failed to save user input to memory "
                            f"after agent error: {save_error}",
                        )
                raise
            finally:
                self._current_call_context = None
                self._clear_todos_file()

            if use_memory:
                try:
                    self._save_history_to_memory(custom_metadata, final_output=result)
                except Exception as save_error:
                    logger.error(
                        "Agent %s - %s: failed to save history to memory: %s",
                        self.name,
                        self.id,
                        save_error,
                    )

            execution_result = {
                "content": result,
            }

            requested_paths = getattr(self, "_requested_output_files", None)

            if self.file_store_backend and requested_paths:
                try:
                    stored_files = self.file_store_backend.list_files_bytes(requested_paths)
                except Exception as e:
                    logger.warning(f"Agent {self.name} - {self.id}: failed to collect files from file store: {e}")
                    stored_files = []
                if stored_files:
                    execution_result["files"] = stored_files
                    logger.info(
                        f"Agent {self.name} - {self.id}: "
                        f"returning {len(stored_files)} requested file(s) from file store"
                    )

            if self.sandbox_backend and requested_paths:
                try:
                    sandbox_files = self.sandbox_backend.collect_files(file_paths=requested_paths)
                except Exception as e:
                    logger.warning(f"Agent {self.name} - {self.id}: failed to collect files from sandbox: {e}")
                    sandbox_files = []
                if sandbox_files:
                    existing_files = execution_result.get("files", [])
                    execution_result["files"] = existing_files + sandbox_files
                    logger.info(
                        f"Agent {self.name} - {self.id}: "
                        f"returning {len(sandbox_files)} requested file(s) from sandbox"
                    )

            logger.info(f"Node {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return execution_result
        finally:
            if sandbox_overlay_token is not None:
                _shared_sandbox_tools.reset(sandbox_overlay_token)
            self._release_shared_sandbox_view()
            _run_extra_tools.reset(ltm_token)
            self._exit_shared_session(shared_session_token)

    def retrieve_conversation_history(
        self,
        user_query: str = None,
        user_id: str = None,
        session_id: str = None,
        limit: int = None,
        strategy: MemoryRetrievalStrategy = MemoryRetrievalStrategy.ALL,
    ) -> list[Message]:
        """
        Retrieves conversation history for the agent using the specified strategy.

        Args:
            user_query: Current user input to find relevant context (for RELEVANT/HYBRID strategies)
            user_id: Optional user identifier
            session_id: Optional session identifier
            limit: Maximum number of messages to return (defaults to memory_limit)
            strategy: Which retrieval strategy to use (ALL, RELEVANT, or HYBRID)

        Returns:
            List of messages forming a valid conversation context
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
            logger.warning("RELEVANT strategy selected but no user_query provided - falling back to ALL")
            strategy = MemoryRetrievalStrategy.ALL

        conversation = self.memory.get_agent_conversation(
            query=user_query,
            limit=limit,
            filters=filters,
            strategy=strategy,
        )
        return conversation

    def _retrieve_memory(self, input_data: AgentInputSchema) -> list[Message]:
        """
        Args:
            input_data: Agent input schema containing user information

        Returns:
            list[Message]: List of messages forming a valid conversation context
        Retrieves memory messages when user_id and/or session_id are provided.
        """
        history_messages = self.retrieve_conversation_history(
            user_query=input_data.input,
            user_id=input_data.user_id,
            session_id=input_data.session_id,
            strategy=self.memory_retrieval_strategy,
        )
        logger.info("Agent %s - %s: retrieved %d messages from memory", self.name, self.id, len(history_messages))
        return history_messages

    def _build_long_term_memory_tools(self, input_data: "AgentInputSchema") -> list[Node]:
        """Construct per-run long-term-memory tools, or [] when LTM is off/absent.

        Raises if LTM is enabled but `input_data.user_id` is missing — the prompt
        already advertises tool blocks at that point, so silently dropping the
        tools would leave the LLM with an empty `tool_description`.
        """
        if self.long_term_memory is None or not self.long_term_memory.enabled:
            return []
        user_id = getattr(input_data, "user_id", None)
        if not user_id:
            raise ValueError(
                "long_term_memory is enabled but input_data.user_id is missing; "
                "pass user_id or disable long_term_memory for this call"
            )
        from dynamiq.nodes.tools.long_term_memory import build_long_term_memory_tools

        tools = build_long_term_memory_tools(
            backend=self.long_term_memory.backend,
            user_id=user_id,
        )
        for tool in tools:
            tool.is_optimized_for_agents = True
        return tools

    def _is_input_output_trace_message(self, message: Message) -> bool:
        """Return True when a message is an internal ReAct/tool-trace entry."""
        content = message.content.strip()

        if message.role == MessageRole.USER:
            return content.startswith("Observation:")

        if message.role == MessageRole.ASSISTANT:
            if content.startswith("Thought:") or content.startswith("Function call:"):
                return True

            if self.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
                try:
                    parsed_content = json.loads(content, strict=False)
                except (TypeError, ValueError):
                    parsed_content = None

                return isinstance(parsed_content, dict) and {"thought", "action"}.issubset(parsed_content)

            if self.inference_mode == InferenceMode.XML:
                return "<thought" in content and ("<action" in content or "<answer" in content)

        return False

    def _input_output_history(
        self,
        metadata: dict,
        snapshot_messages: list[Message],
        final_output: Any | None = None,
    ) -> list[Message]:
        """Return conversation history with internal trace removed.

        INPUT_OUTPUT mode is still snapshot-based: it rewrites the scoped memory
        slice with the current prompt state. The difference from FULL mode is that
        only user/assistant conversation turns are preserved while intermediate
        ReAct/tool messages are dropped.
        """
        history: list[Message] = []
        for message in snapshot_messages:
            if self._is_input_output_trace_message(message):
                continue
            history.append(
                Message(
                    role=message.role,
                    content=message.content,
                    metadata=metadata.copy(),
                    tool_calls=getattr(message, "tool_calls", None),
                    tool_call_id=getattr(message, "tool_call_id", None),
                    name=getattr(message, "name", None),
                )
            )

        if final_output is not None:
            final_output_text = str(final_output)
            if final_output_text:
                if history and history[-1].role == MessageRole.ASSISTANT:
                    history[-1] = Message(
                        role=MessageRole.ASSISTANT,
                        content=final_output_text,
                        metadata=metadata.copy(),
                    )
                else:
                    history.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=final_output_text,
                            metadata=metadata.copy(),
                        )
                    )
        return history

    def _fallback_input_output_pair(
        self,
        metadata: dict,
        snapshot_messages: list[Message],
        final_output: Any | None = None,
    ) -> list[Message]:
        """Return the current user input and the latest assistant response.

        This fallback path is used when scoped snapshot replacement is not
        available. It must not apply ReAct trace filtering to assistant
        messages in FULL mode, otherwise final assistant turns starting with
        ``Thought:`` are silently lost.
        """
        pair: list[Message] = []

        if self._pinned_input is not None:
            pair.append(
                Message(
                    role=MessageRole.USER,
                    content=extract_message_text(self._pinned_input),
                    metadata=metadata.copy(),
                )
            )

        assistant_content = str(final_output) if final_output is not None else None
        if assistant_content == "":
            assistant_content = None

        if assistant_content is None:
            last_assistant = next(
                (m for m in reversed(snapshot_messages) if m.role == MessageRole.ASSISTANT),
                None,
            )
            if last_assistant is not None:
                assistant_content = (
                    self._extract_final_answer_from_tool_calls(last_assistant) or last_assistant.content
                )

        if assistant_content is not None:
            pair.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    metadata=metadata.copy(),
                )
            )

        return pair

    @staticmethod
    def _extract_final_answer_from_tool_calls(message: Message) -> str | None:
        """Return the ``answer`` field from a ``provide_final_answer`` tool_call.

        Used by the 2-message fallback so FUNCTION_CALLING runs that never
        produced a ``final_output`` still persist the real answer text
        instead of the placeholder ``content`` string the agent loop wrote
        next to the tool_call.
        """
        if not message.tool_calls:
            return None
        for call in message.tool_calls:
            fn = call.get("function") or {}
            if fn.get("name") != "provide_final_answer":
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return None
            if isinstance(args, dict):
                answer = args.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return answer
        return None

    def _save_history_to_memory(
        self,
        metadata: dict,
        final_output: Any | None = None,
    ) -> None:
        """Snapshot prompt state into memory for current user/session scope.

        Replaces only the messages matching current ``user_id``/``session_id``
        filters. What gets written depends on ``memory.save_mode``: FULL stores
        every non-system message (incl. tool calls/observations); INPUT_OUTPUT
        stores only the user input and final assistant response.

        Args:
            metadata: Metadata dict forwarded to each ``memory.add`` call.
        """
        user_id = metadata.get("user_id")
        session_id = metadata.get("session_id")
        if not user_id and not session_id:
            logger.warning(
                f"Agent {self.name} - {self.id}: skipping memory snapshot save "
                f"because at least one of user_id or session_id is required",
            )
            return

        fully_scoped = bool(user_id and session_id)

        if getattr(self, "inference_mode", None) == InferenceMode.FUNCTION_CALLING:
            sanitized_dicts = BaseLLM._sanitize_fc_messages([m.to_dict() for m in self._prompt.messages])
            source_messages = Prompt.deserialize_messages(sanitized_dicts)
        else:
            source_messages = self._prompt.messages

        raw_snapshot_messages: list[Message] = []
        for msg in source_messages:
            if msg.role == MessageRole.SYSTEM:
                continue
            raw_snapshot_messages.append(
                Message(
                    role=msg.role,
                    content=extract_message_text(msg),
                    metadata=metadata.copy(),
                    tool_calls=getattr(msg, "tool_calls", None),
                    tool_call_id=getattr(msg, "tool_call_id", None),
                    name=getattr(msg, "name", None),
                )
            )

        if not fully_scoped:
            logger.info(
                f"Agent {self.name} - {self.id}: only one of user_id/session_id provided, "
                "using append-only save to avoid cross-scope data loss.",
            )
            self._append_fallback_messages(metadata, raw_snapshot_messages, final_output=final_output)
            return

        snapshot_messages = raw_snapshot_messages
        if self.memory.save_mode == MemorySaveMode.INPUT_OUTPUT:
            snapshot_messages = self._input_output_history(metadata, snapshot_messages, final_output=final_output)

        saved = len(snapshot_messages)
        scope_filters = {"user_id": user_id, "session_id": session_id}
        try:
            self.memory.replace_messages(filters=scope_filters, messages=snapshot_messages)
        except NotImplementedError:
            logger.warning(
                f"Agent {self.name} - {self.id}: backend does not support scoped delete, "
                "falling back to appending user input and assistant response.",
            )
            self._append_fallback_messages(metadata, raw_snapshot_messages, final_output=final_output)
            return

        logger.info(
            f"Agent {self.name} - {self.id}: saved {saved} message(s) to memory",
        )

    def _append_user_input_to_memory(self, metadata: dict) -> None:
        """Append only the user input to memory (used on agent failure).

        Unlike ``_save_history_to_memory`` this never deletes existing data,
        making it safe to call when the agent errored before producing a response.
        """
        if self._pinned_input is None:
            return
        pinned_content = extract_message_text(self._pinned_input)
        self.memory.add(role=MessageRole.USER, content=pinned_content, metadata=metadata.copy())
        logger.info(f"Agent {self.name} - {self.id}: saved user input to memory after agent error")

    def _append_fallback_messages(
        self,
        metadata: dict,
        snapshot_messages: list[Message],
        final_output: Any | None = None,
    ) -> None:
        """Append only the user input and last assistant response to memory (safe fallback)."""
        pair = self._fallback_input_output_pair(metadata, snapshot_messages, final_output=final_output)
        for m in pair:
            self.memory.add(role=m.role, content=m.content, metadata=m.metadata)
        logger.info(f"Agent {self.name} - {self.id}: saved {len(pair)} message(s) to memory (fallback)")

    def _run_llm(
        self, messages: list[Message | VisionMessage], config: RunnableConfig | None = None, **kwargs
    ) -> RunnableResult:
        """Runs the LLM with a given prompt and handles streaming or full responses.

        Args:
            messages (list[Message | VisionMessage]): Input messages for llm.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            kwargs: Additional keyword arguments.

        Returns:
            RunnableResult: Generated response.
        """
        try:
            check_cancellation(config)
            llm_result = self.llm.run(
                input_data={},
                config=config,
                prompt=Prompt(messages=messages),
                run_depends=deepcopy(self._run_depends),
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
            if llm_result.status == RunnableStatus.CANCELED:
                raise CanceledException()
            if llm_result.status != RunnableStatus.SUCCESS:
                error_message = f"LLM '{self.llm.name}' failed: {llm_result.error.message}"
                raise ValueError(error_message)

            return llm_result

        except Exception as e:
            raise e

    def stream_content(
        self,
        content: str | dict,
        source: str,
        step: str,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> str | dict:
        """
        Streams data.

        Args:
            content (str | dict): Data that will be streamed.
            source (str): Source of the content.
            step (str): Description of the step.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            str | dict: Streamed data.
        """
        if not isinstance(source, str):
            raise ValueError(
                f"stream_content source parameter must be a string, got {type(source).__name__}: {source}. "
                f"This likely indicates incorrect parameter passing from the calling code."
            )

        return self.stream_response(content=content, source=source, step=step, config=config, **kwargs)

    def stream_response(
        self, content: str | dict, source: str, step: str, config: RunnableConfig | None = None, **kwargs
    ):
        if not isinstance(source, str):
            raise ValueError(
                f"stream_response source parameter must be a string, got {type(source).__name__}: {source}. "
                f"This likely indicates a parameter ordering issue in the calling code."
            )

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

        self._pinned_input = input_message

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

    def _regenerate_node_ids(self, obj: Any) -> Any:
        """Recursively assign new IDs to cloned nodes and nested models."""
        if isinstance(obj, BaseModel):
            if hasattr(obj, "id"):
                setattr(obj, "id", str(uuid4()))

            for field_name in getattr(obj, "model_fields", {}):
                value = getattr(obj, field_name)
                if isinstance(value, list):
                    setattr(obj, field_name, [self._regenerate_node_ids(item) for item in value])
                elif isinstance(value, dict):
                    setattr(obj, field_name, {k: self._regenerate_node_ids(v) for k, v in value.items()})
                else:
                    setattr(obj, field_name, self._regenerate_node_ids(value))
            return obj
        if isinstance(obj, list):
            return [self._regenerate_node_ids(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._regenerate_node_ids(v) for k, v in obj.items()}
        return obj

    def _clone_tool_for_execution(
        self,
        tool: Node,
        config: RunnableConfig | None,
        *,
        clone: bool = True,
        override_source_ids: list[str] | None = None,
        target_id: str | None = None,
    ) -> tuple[Node, RunnableConfig]:
        """Prepare a tool for isolated execution: optionally clone, regenerate IDs, align config overrides."""
        base_config = ensure_config(config)
        original_id = tool.id
        if clone:
            try:
                tool = self._regenerate_node_ids(tool.clone())
            except Exception as e:
                logger.warning(f"Agent {self.name} - {self.id}: failed to clone tool {tool.name}: {e}")
                return tool, base_config
        else:
            try:
                self._regenerate_node_ids(tool)
            except Exception as e:
                logger.warning(f"Agent {self.name} - {self.id}: failed to regenerate IDs for tool {tool.name}: {e}")
                return tool, base_config

        if target_id:
            tool.id = target_id

        try:
            lookup_ids = [original_id] + (override_source_ids or [])
            override = None
            for oid in lookup_ids:
                override = base_config.nodes_override.get(oid)
                if override:
                    break
            if override and tool.id != original_id:
                base_config = base_config.model_copy(deep=False)
                base_config.nodes_override[tool.id] = override
        except Exception as e:
            logger.warning(f"Agent {self.name} - {self.id}: failed to align config override for tool {tool.name}: {e}")

        return tool, base_config

    @staticmethod
    def _extract_file_paths_from_input(tool: Node, merged_input: dict[str, Any]) -> list[str] | None:
        """Extract file path references from map_from_storage fields in tool input.

        Scans the tool's input schema for fields tagged with ``map_from_storage``
        and collects any string values the LLM provided for those fields.

        Returns:
            List of file path strings, or None if no paths were found.
        """
        paths: list[str] = []
        for field_name, field in tool.input_schema.model_fields.items():
            if not (field.json_schema_extra and field.json_schema_extra.get("map_from_storage", False)):
                continue
            value = merged_input.get(field_name)
            if value is None:
                continue
            if isinstance(value, str):
                paths.append(value)
            elif isinstance(value, (list, tuple)):
                paths.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, dict):
                paths.extend(v for v in value.values() if isinstance(v, str))
        return paths or None

    def _inject_files_into_tool(self, tool: Node, merged_input: dict[str, Any]) -> None:
        """Inject files from file store or sandbox into tool input when applicable.

        Sandbox files are only collected when the LLM explicitly references paths
        in ``map_from_storage`` fields.  Otherwise the file store is used, which
        means ``Python`` and ``PythonCodeExecutor`` tools always receive files
        from the file store (never from the sandbox).
        """
        if not tool.is_files_allowed:
            return

        file_paths = self._extract_file_paths_from_input(tool, merged_input)

        if self.sandbox_backend and file_paths:
            files = self.sandbox_backend.collect_files(file_paths=file_paths)
            files_map = {path: file for path, file in zip(file_paths, files)}
        elif self.file_store_backend:
            files = self.file_store_backend.list_files_bytes()
            files_map = {getattr(f, "name", f"file_{id(f)}"): f for f in files}
        else:
            return

        if not files:
            return

        for field_name, field in tool.input_schema.model_fields.items():
            if not (field.json_schema_extra and field.json_schema_extra.get("map_from_storage", False)):
                continue
            value = merged_input.get(field_name)
            if value is None:
                merged_input[field_name] = files
            elif isinstance(value, dict):
                merged_input[field_name] = {
                    k: files_map.get(v, v) if isinstance(v, str) else v for k, v in value.items()
                }
            elif isinstance(value, (list, tuple)):
                merged_input[field_name] = [files_map.get(v, v) if isinstance(v, str) else v for v in value]
            elif isinstance(value, str):
                merged_input[field_name] = files_map.get(value, value)

        if isinstance(tool, Python):
            merged_input["files"] = files

        if isinstance(tool, PythonCodeExecutor) and not tool.file_store and self.file_store_backend:
            tool.file_store = self.file_store_backend
            logger.debug(f"Agent {self.name} - {self.id}: injected file_store into PythonCodeExecutor tool {tool.name}")

    def _run_tool(
        self,
        tool: Node,
        tool_input: dict,
        config,
        update_run_depends: bool = True,
        collect_dependency: bool = False,
        delegate_final: bool = False,
        is_parallel: bool = False,
        tool_run_id: str | None = None,
        **kwargs,
    ) -> Any:
        """Runs a specific tool with the given input."""
        from dynamiq.nodes.tools.agent_tool import SubAgentTool
        merged_input = tool_input.copy() if isinstance(tool_input, dict) else {"input": tool_input}

        if not self.delegation_allowed:
            if delegate_final and self.verbose:
                logger.debug(
                    "Agent %s - %s: delegate_final ignored because delegation_allowed is False",
                    self.name,
                    self.id,
                )
            delegate_final = False
            if isinstance(merged_input, dict) and "delegate_final" in merged_input:
                if self.verbose:
                    logger.debug(
                        "Agent %s - %s: delegate_final removed from tool input because delegation_allowed is False",
                        self.name,
                        self.id,
                    )
                merged_input.pop("delegate_final", None)

        raw_tool_params = kwargs.get("tool_params", ToolParams())
        tool_params = (
            ToolParams.model_validate(raw_tool_params) if isinstance(raw_tool_params, dict) else raw_tool_params
        )

        is_child_agent = isinstance(tool, SubAgentTool)
        resolved_agent = None
        try:
            resolved_agent = tool.get_or_create_agent() if is_child_agent else None

            self._inject_files_into_tool(resolved_agent or tool, merged_input)

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
                name_params_any = (
                    tool_params.by_name_params.get(tool.name)
                    or tool_params.by_name_params.get(self.sanitize_tool_name(tool.name))
                    or (resolved_agent and tool_params.by_name_params.get(resolved_agent.name))
                    or (resolved_agent and tool_params.by_name_params.get(self.sanitize_tool_name(resolved_agent.name)))
                )
                if name_params_any:
                    if isinstance(name_params_any, ToolParams):
                        if self.verbose:
                            detail = "will apply to child agent" if is_child_agent else "ignored for non-agent tool"
                            debug_info.append(f"  - From name:{tool.name}: encountered nested ToolParams ({detail})")
                    elif isinstance(name_params_any, dict):
                        self._apply_parameters(merged_input, name_params_any, f"name:{tool.name}", debug_info)

                # 3. Apply parameters by tool ID (highest priority)
                id_params_any = tool_params.by_id_params.get(tool.id) or (
                    resolved_agent and tool_params.by_id_params.get(resolved_agent.id)
                )
                if id_params_any:
                    if isinstance(id_params_any, ToolParams):
                        if self.verbose:
                            detail = "will apply to child agent" if is_child_agent else "ignored for non-agent tool"
                            debug_info.append(f"  - From id:{tool.id}: encountered nested ToolParams ({detail})")
                    elif isinstance(id_params_any, dict):
                        self._apply_parameters(merged_input, id_params_any, f"id:{tool.id}", debug_info)

                if self.verbose and debug_info:
                    logger.debug("\n".join(debug_info))

            child_kwargs = kwargs | {"recoverable_error": True}

            if is_child_agent and self._current_call_context:
                child_context = self._build_child_agent_context(resolved_agent)
                for ctx_key in ("user_id", "session_id"):
                    if ctx_key not in merged_input and child_context.get(ctx_key):
                        merged_input[ctx_key] = child_context[ctx_key]
                if "metadata" not in merged_input and child_context.get("metadata"):
                    merged_input["metadata"] = child_context["metadata"]

            if is_child_agent and tool_params:
                nested_any = (
                    tool_params.by_id_params.get(tool.id)
                    or tool_params.by_id_params.get(getattr(resolved_agent, "id", ""))
                    or tool_params.by_name_params.get(tool.name)
                    or tool_params.by_name_params.get(self.sanitize_tool_name(tool.name))
                    or tool_params.by_name_params.get(getattr(resolved_agent, "name", ""))
                    or tool_params.by_name_params.get(self.sanitize_tool_name(getattr(resolved_agent, "name", "")))
                )
                if nested_any:
                    if isinstance(nested_any, ToolParams):
                        nested_tp = nested_any
                    elif isinstance(nested_any, dict):
                        nested_tp = ToolParams.model_validate(nested_any)
                    else:
                        nested_tp = None
                    if nested_tp:
                        child_kwargs = child_kwargs | {"tool_params": nested_tp}

            effective_delegate_final = delegate_final and is_child_agent
            if is_child_agent and isinstance(merged_input, dict) and "delegate_final" in merged_input:
                effective_delegate_final = effective_delegate_final or bool(merged_input.pop("delegate_final"))

            tool_to_run = resolved_agent if resolved_agent is not None else tool
            tool_config = ensure_config(config)
            if is_parallel and not is_child_agent:
                tool_to_run, tool_config = self._clone_tool_for_execution(tool_to_run, tool_config)
            if is_child_agent and tool.is_factory_mode:
                tool_to_run, tool_config = self._clone_tool_for_execution(
                    resolved_agent,
                    tool_config,
                    clone=False,
                    override_source_ids=[tool.id],
                    target_id=tool_run_id,
                )

            check_cancellation(config)
            tool_result = tool_to_run.run(
                input_data=merged_input,
                config=tool_config,
                run_depends=deepcopy(self._run_depends),
                **child_kwargs,
            )
            dependency_node = tool_to_run if tool_to_run is not tool else tool
            dependency_dict = NodeDependency(node=dependency_node).to_dict(for_tracing=True)
            if update_run_depends:
                self._run_depends = [dependency_dict]
            if tool_result.status == RunnableStatus.CANCELED:
                raise CanceledException()
            if tool_result.status != RunnableStatus.SUCCESS:
                error_message = f"Tool '{tool.name}' failed: {tool_result.error.to_dict()}"
                if tool_result.error.recoverable:
                    raise ToolExecutionException(error_message)
                else:
                    raise ValueError(error_message)

            if is_child_agent and tool.max_calls is not None:
                tool.increment_call_count()

            tool_result_output_content = tool_result.output.get("content")

            saved_files = self._handle_tool_generated_files(tool, tool_result)

            tool_result_content_processed = process_tool_output_with_sandbox_persistence(
                content=tool_result_output_content,
                tool_name=tool.name,
                tool_input=tool_input,
                sandbox=self.sandbox_backend,
                save_tool_output_to_sandbox=bool(self.sandbox_backend and tool.is_output_persisted_in_sandbox_allowed),
                sandbox_persistence_config=self.tool_output_sandbox_persistence,
                max_tokens=self.tool_output_max_length,
                truncate=self.tool_output_truncate_enabled and not effective_delegate_final,
            )

            if saved_files:
                paths = ", ".join(saved_files)
                tool_result_content_processed = f"{tool_result_content_processed}\n\nFiles saved: {paths}"

            output_files = tool_result.output.get("files", [])
            tool_output_meta = {k: v for k, v in tool_result.output.items() if k not in ("content", "files")}

            if not isinstance(tool, ContextManagerTool):
                self._tool_cache[ToolCacheEntry(action=tool.name, action_input=tool_input)] = (
                    tool_result_content_processed,
                    tool_output_meta,
                )
            if collect_dependency:
                return tool_result_content_processed, output_files, tool_output_meta, dependency_dict

            return tool_result_content_processed, output_files, tool_output_meta
        finally:
            if is_child_agent and tool.is_factory_mode and resolved_agent is not None:
                SubAgentTool.cleanup_factory_agent(resolved_agent)

    def _ensure_named_files(self, files: list[io.BytesIO | bytes]) -> list[io.BytesIO | bytes]:
        """Ensure all uploaded files have name and description attributes and store them in storage backend."""
        named = []
        for i, f in enumerate(files):
            if isinstance(f, bytes):
                bio = io.BytesIO(f)
                bio.name = f"file_{i}.bin"
                bio.description = "User-provided file"
                named.append(bio)
            elif isinstance(f, io.BytesIO):
                if not hasattr(f, "name"):
                    f.name = f"file_{i}"
                if not hasattr(f, "description"):
                    f.description = "User-provided file"
                named.append(f)
            else:
                named.append(f)
        return named

    @staticmethod
    def _extension_for_mime_type(mime_type: str) -> str:
        """Return a stable file extension for a MIME type."""
        if not mime_type:
            return ".bin"

        extension = mimetypes.guess_extension(mime_type)
        if extension:
            return extension

        if "/" not in mime_type:
            return ".bin"

        subtype = mime_type.rsplit("/", 1)[-1].split("+", 1)[0]
        return f".{subtype}" if subtype else ".bin"

    @classmethod
    def _data_url_to_file(cls, data_url: str, *, media_type: str, index: int) -> io.BytesIO:
        """Decode a data URL into a named BytesIO suitable for the generic file pipeline."""
        match = re.match(r"^data:([^;,]+)?((?:;[^,]*)*),(.*)$", data_url, flags=re.DOTALL)
        if not match:
            raise ValueError("invalid data URL")

        mime_type = match.group(1) or "application/octet-stream"
        params = [param.lower() for param in match.group(2).split(";") if param]
        payload = match.group(3)

        try:
            if "base64" in params:
                content = base64.b64decode(payload.strip(), validate=True)
            else:
                content = unquote_to_bytes(payload)
        except (binascii.Error, ValueError) as e:
            raise ValueError("invalid data URL payload") from e

        file_obj = io.BytesIO(content)
        file_obj.name = f"{media_type}_{index}{cls._extension_for_mime_type(mime_type)}"
        file_obj.description = f"User-provided {media_type} attachment decoded from data URL"
        file_obj.content_type = mime_type
        return file_obj

    @staticmethod
    def _local_media_path_to_file(path: str, *, media_type: str) -> io.BytesIO | None:
        """Read an existing local media path into a file object for generic file upload."""
        local_path = Path(path).expanduser()
        if not local_path.is_file():
            return None

        content = local_path.read_bytes()
        file_obj = io.BytesIO(content)
        file_obj.name = local_path.name
        file_obj.description = f"User-provided {media_type} attachment loaded from local path"
        file_obj.content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        return file_obj

    def _append_unsupported_media_attachment(
        self,
        attachment: str | io.BytesIO | bytes,
        other_files: list[io.BytesIO | bytes],
        media_url_references: list[str],
        *,
        media_type: str,
        index: int,
    ) -> None:
        """Route unsupported media to file upload or short textual references."""
        if not isinstance(attachment, str):
            other_files.append(attachment)
            return

        if not attachment.startswith("data:"):
            try:
                if file_obj := self._local_media_path_to_file(attachment, media_type=media_type):
                    other_files.append(file_obj)
                    return
            except OSError as e:
                logger.warning(
                    "Agent %s - %s: failed to read unsupported %s local path attachment '%s': %s",
                    self.name,
                    self.id,
                    media_type,
                    attachment,
                    e,
                )
            media_url_references.append(attachment)
            return

        try:
            other_files.append(self._data_url_to_file(attachment, media_type=media_type, index=index))
        except ValueError as e:
            logger.warning(
                "Agent %s - %s: failed to decode unsupported %s data URL attachment: %s",
                self.name,
                self.id,
                media_type,
                e,
            )
            media_url_references.append(f"{media_type} data URL attachment could not be decoded")

    @staticmethod
    def _split_upload_filename(file_name: str) -> tuple[str, str]:
        """Split a file name into stem and extension for suffixing."""
        stem, dot, extension = file_name.rpartition(".")
        if not dot or not stem:
            return file_name, ""
        return stem, f".{extension}"

    def _get_unique_upload_filename(
        self,
        file_name: str,
        seen_names: set[str],
        exists_check: Callable[[str], bool] | None = None,
    ) -> str:
        """Return a collision-free file name, preserving extension."""
        candidate = file_name
        stem, extension = self._split_upload_filename(file_name)
        suffix = 1

        while candidate in seen_names or (exists_check is not None and exists_check(candidate)):
            candidate = f"{stem}_{suffix}{extension}"
            suffix += 1

        seen_names.add(candidate)
        return candidate

    def _list_existing_sandbox_file_names(self) -> set[str]:
        """Best-effort list of existing sandbox file names for collision checks."""
        if not self.sandbox_backend:
            return set()

        try:
            existing_paths = self.sandbox_backend.list_files(target_dir=self.sandbox_backend.base_path)
        except Exception:
            return set()

        existing_names = set()
        for path in existing_paths:
            if isinstance(path, str):
                file_name = path.rsplit("/", 1)[-1]
                if file_name:
                    existing_names.add(file_name)
        return existing_names

    def _handle_tool_generated_files(self, tool: Node, tool_result: RunnableResult) -> list[str]:
        """
        Handle files generated by tools and store them in the file store and/or sandbox.

        Args:
            tool: The tool that generated the files
            tool_result: The result from the tool execution

        Returns:
            List of saved file paths (full sandbox paths or file-store keys).
        """
        if not self.file_store_backend and not self.sandbox_backend:
            return []

        if not (isinstance(tool_result.output, dict) and "files" in tool_result.output):
            return []

        tool_files = tool_result.output.get("files", [])
        if not tool_files:
            return []

        stored_files = []
        for file in tool_files:
            if isinstance(file, io.BytesIO):
                file_name = getattr(file, "name", f"file_{id(file)}.bin")
                file_description = getattr(file, "description", "Tool-generated file")
                content_type = getattr(file, "content_type", "application/octet-stream")

                content = file.read()
                file.seek(0)

                if self.sandbox_backend:
                    try:
                        dest = f"{self.sandbox_backend.base_path}/{file_name}"
                        self.sandbox_backend.upload_file(file_name, content, destination_path=dest)
                        stored_files.append(dest)
                    except Exception as e:
                        logger.warning(f"Failed to upload tool file '{file_name}' to sandbox: {e}")
                elif self.file_store_backend:
                    self.file_store_backend.store(
                        file_path=file_name,
                        content=content,
                        content_type=content_type,
                        metadata={"description": file_description, "source": "tool_generated"},
                        overwrite=True,
                    )
                    stored_files.append(file_name)

            elif isinstance(file, bytes):
                file_name = f"file_{id(file)}.bin"
                file_description = f"Tool-{tool.name}-generated file"
                content_type = "application/octet-stream"

                if self.sandbox_backend:
                    try:
                        dest = f"{self.sandbox_backend.base_path}/{file_name}"
                        self.sandbox_backend.upload_file(file_name, file, destination_path=dest)
                        stored_files.append(dest)
                    except Exception as e:
                        logger.warning(f"Failed to upload tool file '{file_name}' to sandbox: {e}")
                elif self.file_store_backend:
                    self.file_store_backend.store(
                        file_path=file_name,
                        content=file,
                        content_type=content_type,
                        metadata={"description": file_description, "source": "tool_generated"},
                        overwrite=True,
                    )
                    stored_files.append(file_name)
            else:
                logger.warning(f"Unsupported file type from tool '{tool.name}': {type(file)}")

        if stored_files:
            logger.info(f"Tool '{tool.name}' generated {len(stored_files)} file(s): {stored_files}")
        return stored_files

    def _upload_files_to_sandbox(self, normalized_files: list) -> list[str]:
        """Upload file-like objects to the sandbox backend."""
        file_paths = [""] * len(normalized_files)
        seen_names = self._list_existing_sandbox_file_names()
        for index, file_obj in enumerate(normalized_files):
            file_name = getattr(file_obj, "name", None)
            if file_name and hasattr(file_obj, "read"):
                try:
                    if hasattr(file_obj, "seek"):
                        file_obj.seek(0)
                    content = file_obj.read()
                    if isinstance(content, str):
                        content = content.encode("utf-8")
                    unique_file_name = self._get_unique_upload_filename(file_name, seen_names)
                    input_path = f"{self.sandbox_backend.base_path.rstrip('/')}/input/{unique_file_name}"
                    destination_path = self.sandbox_backend.upload_file(
                        unique_file_name, content, destination_path=input_path
                    )
                    file_paths[index] = destination_path
                except Exception as e:
                    logger.warning(f"Failed to upload file {file_name} to sandbox: {e}")
        return file_paths

    def _upload_files_to_file_store(self, normalized_files: list) -> list[str]:
        """Store file-like objects in the file store backend."""
        file_paths = [""] * len(normalized_files)
        seen_names: set[str] = set()

        def file_exists(candidate: str) -> bool:
            try:
                return bool(self.file_store_backend.exists(candidate))
            except Exception:
                return False

        for index, file_obj in enumerate(normalized_files):
            file_name = getattr(file_obj, "name", None)
            if not file_name or not hasattr(file_obj, "read"):
                continue
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                content = file_obj.read()
                if isinstance(content, str):
                    content = content.encode("utf-8")
                description = getattr(file_obj, "description", "User-provided file")
                unique_file_name = self._get_unique_upload_filename(file_name, seen_names, exists_check=file_exists)
                self.file_store_backend.store(
                    file_path=unique_file_name,
                    content=content,
                    content_type=getattr(file_obj, "content_type", "application/octet-stream"),
                    metadata={"description": description, "source": "user_upload"},
                    overwrite=False,
                )
                file_paths[index] = unique_file_name
            except Exception as e:
                logger.warning(f"Failed to store file {file_name} in file store: {e}")
        return file_paths

    def _setup_in_memory_file_store_and_tools(self) -> None:
        """Create in-memory file store and file tools when files are uploaded and no sandbox/file store exists."""
        self.file_store = FileStoreConfig(enabled=True, backend=InMemoryFileStore())
        self.tools.extend(
            [
                FileReadTool(file_store=self.file_store_backend, llm=self.llm),
                FileSearchTool(file_store=self.file_store_backend),
                FileListTool(file_store=self.file_store_backend),
            ]
        )
        new_tool_description = self.tool_description
        self.system_prompt_manager.set_initial_variable("tool_description", new_tool_description)
        if self.system_prompt_manager._prompt_blocks.get("tools") == "":
            from dynamiq.nodes.agents.agent import Agent

            if isinstance(self, Agent):
                from dynamiq.nodes.agents.prompts.manager import ReactPromptConfig
                from dynamiq.nodes.tools.agent_tool import SubAgentTool

                self.system_prompt_manager.build_react_prompt(
                    ReactPromptConfig(
                        inference_mode=self.inference_mode,
                        has_tools=True,
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

    def _inject_attached_files_into_message(
        self,
        input_message: Message | VisionMessage,
        files: list[io.BytesIO],
        file_paths: list[str] | None = None,
    ) -> Message | VisionMessage:
        if not files:
            return input_message

        if not isinstance(input_message, Message):
            return input_message

        file_lines = []

        normalized_paths = file_paths or []
        for index, f in enumerate(files):
            name = getattr(f, "name", None) or "unnamed_file"
            description = getattr(f, "description", "") or ""
            description = description.strip()
            saved_path = normalized_paths[index] if index < len(normalized_paths) else ""
            if not saved_path:
                saved_path = "File is not stored."

            saved_suffix = f" (saved as: {saved_path})"
            if description:
                file_lines.append(f"- {name}{saved_suffix}: {description}")
            else:
                file_lines.append(f"- {name}{saved_suffix}")

        if not file_lines:
            return input_message

        file_section = "\n".join(["\nAttached files available to you:"] + file_lines) + "\n"

        if isinstance(input_message.content, str):
            input_message.content = f"{input_message.content.rstrip()}{file_section}"
        else:
            input_message.content = input_message.content + file_section

        return input_message

    def _inject_attached_media_references_into_message(
        self, input_message: Message | VisionMessage, references: list[str]
    ) -> Message | VisionMessage:
        """Surface media URLs/paths as plain text when the LLM can't consume them as vision
        content. These are external references (not binary payloads), so they're rendered
        as text rather than routed through the file-upload pipeline, which only handles
        bytes/BytesIO content.
        """
        if not references or not isinstance(input_message, Message):
            return input_message

        lines = [f"- {ref}" for ref in references]
        section = (
            "\n".join(["\nMedia attachments not supported by this model (referenced by URL/path):"] + lines) + "\n"
        )

        if isinstance(input_message.content, str):
            input_message.content = f"{input_message.content.rstrip()}{section}"
        else:
            input_message.content = input_message.content + section

        return input_message

    def _inject_attached_media_into_message(
        self,
        input_message: Message | VisionMessage,
        images: list[str | io.BytesIO | bytes],
        videos: list[str | io.BytesIO | bytes],
    ) -> Message | VisionMessage:
        """Convert image/video attachments into vision content blocks the LLM can actually
        see, instead of leaving them as opaque text references.

        Only applies when input_message is still a plain Message -- a caller-provided
        VisionMessage (e.g. one built with its own Jinja placeholders, like `input_message=`
        on Agent construction) is left untouched, matching `_inject_attached_files_into_message`.
        Callers are expected to have already dropped any images/videos the configured LLM
        doesn't support (see `is_vision_supported`/`is_video_input_supported`).
        """
        if not images and not videos:
            return input_message

        if not isinstance(input_message, Message):
            return input_message

        content = []
        if input_message.content:
            content.append(VisionMessageTextContent(text=input_message.content))

        for image in images:
            try:
                if isinstance(image, str):
                    if image.startswith(("http://", "https://", "data:")):
                        image_url = image
                    else:
                        with open(image, "rb") as file:
                            image_url = bytes_to_data_url(file.read())
                else:
                    image_bytes = image.getvalue() if isinstance(image, io.BytesIO) else image
                    image_url = bytes_to_data_url(image_bytes)
                content.append(VisionMessageImageContent(image_url=VisionMessageImageURL(url=image_url)))
            except Exception as e:
                logger.error(f"Agent {self.name} - {self.id}: error processing image attachment: {str(e)}")

        for video in videos:
            try:
                if isinstance(video, str):
                    if video.startswith(("http://", "https://", "data:", "gs://")):
                        video_url = video
                    else:
                        with open(video, "rb") as file:
                            video_url = bytes_to_data_url(file.read())
                else:
                    video_bytes = video.getvalue() if isinstance(video, io.BytesIO) else video
                    video_url = bytes_to_data_url(video_bytes)
                content.append(VisionMessageFileContent(file=VisionMessageFileData(file_data=video_url)))
            except Exception as e:
                logger.error(f"Agent {self.name} - {self.id}: error processing video attachment: {str(e)}")

        if not content:
            return input_message

        return VisionMessage(content=content, role=input_message.role, static=input_message.static)

    @property
    def file_store_backend(self) -> FileStore | None:
        """Get the file store backend from the configuration if enabled."""
        return self.file_store.backend if self.file_store.enabled else None

    @property
    def sandbox_backend(self) -> Sandbox | None:
        """Get the effective sandbox backend for this agent.

        When this agent borrows an owner's shared sandbox, the per-agent view is the
        effective backend so uploads, output collection, skills ingestion and cleanup
        all target the same sandbox as the sandbox tools. Otherwise falls back to the
        agent's own configured backend (if enabled).
        """
        if self._shared_sandbox_view is not None:
            return self._shared_sandbox_view
        return self.sandbox.backend if self.sandbox and self.sandbox.enabled else None

    def _maybe_enter_shared_session(self, kwargs: dict):
        """Establish a shared execution session for this subtree if this agent owns one.

        Returns a ContextVar token to reset later, or None when nothing was set
        (flag off, no sandbox, or a session already exists and is inherited).
        """
        if _shared_session.get() is not None:
            return None  # inherit the ancestor's session
        if not self.share_sandbox_with_subagents or self.sandbox_backend is None:
            return None

        session = SharedSession(
            sandbox=self.sandbox_backend,
            share_sandbox=True,
            owner_run_id=str(kwargs.get("run_id") or self.id),
            sharing_scope=self.sandbox_sharing_scope,
        )
        return _shared_session.set(session)

    def _exit_shared_session(self, token) -> None:
        """Reset the ContextVar set by `_maybe_enter_shared_session`."""
        if token is not None:
            _shared_session.reset(token)

    def _resolve_tools_sandbox(self):
        """The sandbox this agent builds its OWN sandbox tools from at construction.

        Shared-sandbox borrowing is resolved at run time in ``execute()`` (see
        ``_maybe_borrow_shared_sandbox``), not here — that is what lets initialized-mode
        subagents share too. At construction ``_shared_sandbox_view`` is always None, so
        ``sandbox_backend`` returns exactly this agent's own configured backend.
        """
        return self.sandbox_backend

    def _configured_sandbox_backend(self) -> "Sandbox | None":
        """This agent's OWN configured sandbox backend, ignoring any borrowed shared view."""
        return self.sandbox.backend if self.sandbox and self.sandbox.enabled else None

    def _maybe_borrow_shared_sandbox(self) -> list[Node] | None:
        """Resolve a per-agent view of the owner's shared sandbox at run time.

        Called from ``execute()``. When this agent is a *borrower* under an active shared
        session, build sandbox tools on a fresh per-agent view of the shared sandbox and
        return them as a run-time overlay (published via ``_shared_sandbox_tools``). Returns
        None when this agent should not borrow:

        - no active/shareable session;
        - this agent *owns* the shared sandbox (its own tools already use it);
        - it uses a file store (file store and sandbox are mutually exclusive); or
        - scope is AUGMENT and it brings its own sandbox.

        Sets ``_shared_sandbox_view`` / ``_sandbox_is_shared`` as a side effect when it borrows.
        Works for both factory- and initialized-mode subagents because the ContextVar is
        always visible in ``execute()`` regardless of when the instance was constructed.
        """
        session = _shared_session.get()
        if session is None or not session.share_sandbox:
            return None

        own = self._configured_sandbox_backend()
        if own is not None and session.get_sandbox() is own:
            return None  # this agent owns the shared sandbox — nothing to borrow

        if self.file_store_backend is not None:
            return None  # file-store agents never join the shared sandbox

        if session.sharing_scope == SandboxSharingScope.AUGMENT and own is not None:
            return None  # augment: keep this subagent's own sandbox

        # Key the workdir on this agent's stable instance id, not a per-call random suffix, so a
        # reused initialized subagent lands in the SAME /work/<key> across calls within a run (its
        # relative-path files persist). Distinct instances still get distinct ids -> distinct
        # workdirs; factory subagents are rebuilt per call and so still rotate, as intended.
        key = f"{(self.sanitize_tool_name(self.name) or 'subagent').lower()}-{self.id}"
        view = session.sandbox_view_for(key)
        if view is None:
            return None

        # Build tools BEFORE committing any instance state, so a failure here leaves this agent
        # untouched: no latched view and the dedicated sandbox not yet released.
        tools = view.get_tools(llm=self.llm)
        for tool in tools:
            if tool.is_postponed_component_init:
                tool.init_components()
            tool.is_optimized_for_agents = True

        if own is not None:
            # scope=ALL routes this subagent onto the shared view instead of its own sandbox. We do
            # NOT tear the dedicated backend down here: this borrow path is shared by reused
            # initialized subagents, which must keep their own sandbox for later standalone use.
            # A *factory* subagent's orphaned backend is torn down in cleanup_factory_agent instead.
            logger.warning(
                "Agent %s - %s: sandbox_sharing_scope=ALL overrides this subagent's own sandbox; "
                "routing it onto the owner's shared sandbox instead.",
                self.name,
                self.id,
            )

        self._sandbox_is_shared = True
        self._shared_sandbox_view = view
        return tools

    def _release_shared_sandbox_view(self) -> None:
        """Disconnect and drop a per-call borrowed shared-sandbox view.

        ``kill=False`` keeps the underlying shared sandbox alive for the owner and other
        subagents; only this agent's client connection is dropped. ``_sandbox_is_shared``
        stays latched so ``cleanup_factory_agent`` never kills the shared sandbox.
        """
        view = self._shared_sandbox_view
        if view is None:
            return
        self._shared_sandbox_view = None
        try:
            view.close(kill=False)
        except Exception as e:
            logger.warning("Agent %s - %s: shared sandbox view close failed: %s", self.name, self.id, e)

    @property
    def _runtime_tools(self) -> list[Node]:
        """Instance tools plus any per-call overlays.

        Overlays: long-term-memory tools bound to user_id (``_run_extra_tools``) and shared-sandbox
        tools bound to a per-agent view (``_shared_sandbox_tools``). When a shared-sandbox overlay is
        active, this agent's OWN sandbox tools are hidden (scope=ALL override) so the model sees a
        single, consistent sandbox toolset.
        """
        tools = self.tools
        sandbox_overlay = _shared_sandbox_tools.get()
        # Only strip own sandbox tools when the overlay actually provides replacements. An empty
        # (but non-None) overlay must not leave the agent with no sandbox tools at all.
        if sandbox_overlay and self._own_sandbox_tool_ids:
            tools = [t for t in tools if t.id not in self._own_sandbox_tool_ids]
        result = list(tools)
        ltm_overlay = _run_extra_tools.get()
        if ltm_overlay:
            result += ltm_overlay
        if sandbox_overlay:
            result += sandbox_overlay
        return result

    @property
    def tool_description(self) -> str:
        """Returns a description of the tools available to the agent."""
        tools = self._runtime_tools
        return (
            "\n".join([f"- {tool.name}: {(tool.description or '').strip()}" for tool in tools])
            if tools
            else ""
        )

    @property
    def tool_names(self) -> str:
        """Returns a comma-separated list of tool names available to the agent."""
        return ",".join([self.sanitize_tool_name(tool.name) for tool in self._runtime_tools])

    @property
    def tool_by_names(self) -> dict[str, Node]:
        """Returns a dictionary mapping tool names to their corresponding Node objects."""
        return {self.sanitize_tool_name(tool.name): tool for tool in self._runtime_tools}

    def reset_run_state(self):
        """Resets the agent's run state.

        IterativeCheckpointMixin fields (_iteration_state, _has_restored_iteration)
        are preserved — they are consumed once by _run_agent() via get_start_iteration().
        SubAgentTool call counts are preserved on resume so the per-run limit stays correct.
        """
        self._run_depends = []
        self._tool_cache: dict[ToolCacheEntry, Any] = {}
        self._completed_loops = 0
        self.system_prompt_manager.reset()

        if not self.is_resumed:
            from dynamiq.nodes.tools.agent_tool import SubAgentTool

            for tool in self.tools:
                if isinstance(tool, SubAgentTool):
                    tool.reset_call_count()

    def generate_prompt(self, block_names: list[str] | None = None, **kwargs) -> str:
        """Generates the prompt using specified blocks and variables."""
        return self.system_prompt_manager.generate_prompt(block_names=block_names, **kwargs)

    def _build_child_agent_context(self, child_agent: "Agent") -> dict[str, Any]:
        """Return context for child agents with per-agent ids to isolate their memory."""
        if not self._current_call_context:
            return {}

        suffix_raw = getattr(child_agent, "name", None) or getattr(child_agent, "id", None) or "subagent"
        suffix_clean = self.sanitize_tool_name(str(suffix_raw)) or "subagent"
        child_context: dict[str, Any] = {}

        for ctx_key in ("user_id", "session_id"):
            base_val = self._current_call_context.get(ctx_key)
            if base_val:
                child_context[ctx_key] = f"{base_val}:{suffix_clean}"

        if metadata := self._current_call_context.get("metadata"):
            child_context["metadata"] = metadata

        return child_context

    def get_clone_attr_initializers(self) -> dict[str, Callable[[Node], Any]]:
        base = super().get_clone_attr_initializers()
        from dynamiq.prompts import Prompt

        base.update(
            {
                "_prompt": (lambda _self: Prompt(messages=[]) if Prompt else None),
            }
        )
        return base


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
    name: str = "agent-manager"
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
        self._actions = {
            "plan": self._plan,
            "assign": self._assign,
            "final": self._final,
            "handle_input": self._handle_input,
        }

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
        log_data = dict(input_data).copy()

        if log_data.get("images"):
            log_data["images"] = [f"image_{i}" for i in range(len(log_data["images"]))]

        if log_data.get("videos"):
            log_data["videos"] = [f"video_{i}" for i in range(len(log_data["videos"]))]

        if log_data.get("files"):
            log_data["files"] = [f"file_{i}" for i in range(len(log_data["files"]))]

        logger.info(f"Agent {self.name} - {self.id}: started with input {log_data}")
        self.reset_run_state()
        config = config or RunnableConfig()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        action = input_data.action

        self.system_prompt_manager.update_variables(dict(input_data))

        kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        kwargs.pop("run_depends", None)
        _result_llm = self._actions[action](config=config, **kwargs)
        result = {"action": action, "result": _result_llm}

        execution_result = {
            "content": result,
        }
        logger.info(f"Agent {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

        return execution_result

    def _plan(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'plan' action."""
        prompt = self.system_prompt_manager.render_block(
            "plan", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]

        return llm_result

    def _assign(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'assign' action."""
        prompt = self.system_prompt_manager.render_block(
            "assign", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]

        return llm_result

    def _final(self, config: RunnableConfig, **kwargs) -> str:
        """Executes the 'final' action."""
        prompt = self.system_prompt_manager.render_block(
            "final", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        if self.streaming.enabled:
            return self.stream_content(
                content=llm_result,
                step="manager_final_output",
                source=self.name,
                config=config,
                **kwargs,
            )
        return llm_result

    def _handle_input(self, config: RunnableConfig, **kwargs) -> str:
        """
        Executes the single 'handle_input' action to either respond or plan
        based on user request complexity.
        """
        prompt = self.system_prompt_manager.render_block(
            "handle_input", **(self.system_prompt_manager._prompt_variables | kwargs)
        )
        llm_result = self._run_llm([Message(role=MessageRole.USER, content=prompt)], config, **kwargs).output["content"]
        return llm_result
