# RFC-001-02: Industry Research - Deep Dive

**Status:** Final Draft v7.0 (Comprehensive)  
**Created:** January 6, 2026  
**Part:** 2 of 9

---

## 1. Research Methodology

We analyzed **12 major AI/ML workflow frameworks** across two categories:

### 1.1 Deep Code Analysis (Cloned & Reviewed)

| Framework | Repository | Files Analyzed |
|-----------|------------|----------------|
| **LangGraph** | `langchain-ai/langgraph` | `checkpoint/base/__init__.py`, `checkpoint/serde/jsonplus.py`, `checkpoint/postgres/__init__.py`, `types.py` |
| **CrewAI** | `crewaiinc/crewAI` | `flow/persistence/base.py`, `flow/persistence/sqlite.py` |
| **Google ADK** | `google/adk-python` | `sessions/session.py`, `sessions/base_session_service.py`, `sessions/database_session_service.py` |
| **Metaflow** | `Netflix/metaflow` | `runtime.py` |
| **Letta** | `letta-ai/letta` | `services/agent_serialization_manager.py` |
| **Manus AI** | (Documentation) | TiDB case study |

### 1.2 Documentation & API Analysis (Additional Competitors)

| Framework | Source | Key Feature |
|-----------|--------|-------------|
| **Microsoft AutoGen / Agent Framework** | Microsoft Docs | `save_state()` / `load_state()`, automatic checkpointing |
| **Temporal** | temporal.io | Event-sourced durable execution, Continue-As-New |
| **Prefect** | prefect.io | `persist_result=True`, cache policies |
| **Semantic Kernel (Microsoft)** | Microsoft Docs | Stateful steps, process framework checkpointing |
| **Haystack (deepset)** | haystack.deepset.ai | Breakpoints, snapshots, `PipelineRuntimeError` recovery |
| **Amazon Bedrock Agents** | AWS Docs | Session Management APIs, AgentCore Memory |

---

## 1.3 Additional Competitors Summary

### Temporal (Industry Gold Standard for Workflows)

**Approach:** Event-sourced durable execution
- Every state change recorded as immutable event
- Replay mechanism reconstructs state from Event History
- `Continue-As-New` for long-running workflows (checkpoint + fresh history)

**Key Insight:** Temporal's approach is the most robust but requires full event sourcing. Our DAG model is simpler but we adopt their principle of "never lose progress."

### Microsoft AutoGen / Agent Framework

**Approach:** Explicit state management with migration to automatic checkpointing
- AutoGen: `agent.save_state()` / `agent.load_state()` (manual)
- Agent Framework: Automatic persistence at superstep boundaries
- Captures: executor state, pending messages, shared state

**Key Insight:** Microsoft's migration from manual to automatic aligns with our opt-in approach. Their "superstep boundary" concept matches our node boundary checkpointing.

### Prefect

**Approach:** Result persistence + caching
- `@task(persist_result=True)` for checkpointing
- Cache policies (`INPUTS`, `TASK_SOURCE`) for avoiding re-execution
- State tracking (`Scheduled`, `Running`, `Completed`, `Failed`)

**Key Insight:** Prefect's per-task persistence mirrors our per-node approach. Their cache policy concept could enhance our tool caching in Agent.

### Haystack (deepset)

**Approach:** Breakpoints and snapshots
- `Breakpoint` class to pause at specific components
- Automatic snapshot on failure (`PipelineRuntimeError.pipeline_snapshot`)
- Resume from snapshot file

**Key Insight:** Haystack's breakpoint concept is very close to our design! Their JSON snapshot approach validates our serialization strategy.

### Amazon Bedrock Agents

**Approach:** Session Management APIs
- Checkpointing workflow stages for HITL
- IAM-based access control
- Encryption with KMS keys
- AgentCore Memory for short-term + long-term persistence

**Key Insight:** AWS's separation of short-term (session) and long-term (memory) aligns with our checkpoint vs Memory distinction. Their security model (IAM + KMS encryption) informs our future security roadmap.

---

## 2. LangGraph Deep Dive (Most Comprehensive)

**Repository:** `langchain-ai/langgraph`  
**Key Insight:** LangGraph has the most mature checkpoint system, designed for channel-based state with full HITL support.

### 2.1 Core Data Structures

```python
# From langgraph/checkpoint/base/__init__.py

class Checkpoint(TypedDict):
    """State snapshot at a given point in time."""
    
    v: int
    """The version of the checkpoint format. Currently 1."""
    
    id: str
    """The ID of the checkpoint. Both unique AND monotonically increasing,
    so can be used for sorting checkpoints from first to last."""
    
    ts: str
    """The timestamp of the checkpoint in ISO 8601 format."""
    
    channel_values: dict[str, Any]
    """The values of the channels at the time of the checkpoint.
    Mapping from channel name to deserialized channel snapshot value."""
    
    channel_versions: ChannelVersions  # dict[str, str | int | float]
    """The versions of the channels at the time of the checkpoint.
    Keys are channel names, values are monotonically increasing version strings."""
    
    versions_seen: dict[str, ChannelVersions]
    """Map from node ID to map from channel name to version seen.
    This keeps track of which versions each node has seen.
    Used to determine which nodes to execute next."""
    
    updated_channels: list[str] | None
    """The channels that were updated in this checkpoint."""


class CheckpointMetadata(TypedDict, total=False):
    """Metadata associated with a checkpoint."""
    
    source: Literal["input", "loop", "update", "fork"]
    """The source of the checkpoint:
    - "input": Created from an input to invoke/stream/batch
    - "loop": Created from inside the pregel loop
    - "update": Created from a manual state update
    - "fork": Created as a copy of another checkpoint"""
    
    step: int
    """The step number: -1 for first input, 0 for first loop, etc."""
    
    parents: dict[str, str]
    """IDs of parent checkpoints. Mapping from namespace to checkpoint ID."""


class CheckpointTuple(NamedTuple):
    """A tuple containing a checkpoint and its associated data."""
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: RunnableConfig | None = None
    pending_writes: list[PendingWrite] | None = None  # (task_id, channel, value)
```

**Key Design Decisions:**
- `versions_seen` enables detecting which nodes need re-execution
- `pending_writes` captures partial state during node execution
- Monotonically increasing `id` enables time-travel to any checkpoint

### 2.2 Backend Interface

```python
class BaseCheckpointSaver(Generic[V]):
    """Base class for creating a graph checkpointer."""
    
    serde: SerializerProtocol = JsonPlusSerializer()
    
    # === Synchronous Methods ===
    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple using the given configuration."""
        raise NotImplementedError
    
    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria."""
        raise NotImplementedError
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata."""
        raise NotImplementedError
    
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint."""
        raise NotImplementedError
    
    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints associated with a thread ID."""
        raise NotImplementedError
    
    # === Async Variants (all methods have async versions) ===
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None: ...
    async def alist(...) -> AsyncIterator[CheckpointTuple]: ...
    async def aput(...) -> RunnableConfig: ...
    async def aput_writes(...) -> None: ...
    async def adelete_thread(thread_id: str) -> None: ...
    
    def get_next_version(self, current: V | None, channel: None) -> V:
        """Generate the next version ID for a channel.
        Default: integer versions, incrementing by 1."""
        if current is None:
            return 1
        return current + 1
```

### 2.3 Advanced Serialization (JsonPlusSerializer)

LangGraph's `JsonPlusSerializer` handles complex Python objects using `ormsgpack` with extension types:

```python
class JsonPlusSerializer(SerializerProtocol):
    """Serializer that uses ormsgpack, with optional fallbacks."""
    
    def __init__(
        self,
        *,
        pickle_fallback: bool = False,  # Dangerous! Only for trusted data
        allowed_json_modules: Sequence[tuple[str, ...]] | Literal[True] | None = None,
    ) -> None: ...
    
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        """Serialize object to (type_tag, bytes)."""
        if obj is None:
            return "null", b""
        elif isinstance(obj, bytes):
            return "bytes", obj
        elif isinstance(obj, bytearray):
            return "bytearray", obj
        else:
            try:
                return "msgpack", _msgpack_enc(obj)
            except ormsgpack.MsgpackEncodeError:
                if self.pickle_fallback:
                    return "pickle", pickle.dumps(obj)
                raise
    
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        """Deserialize from (type_tag, bytes)."""
        type_, data_ = data
        if type_ == "null":
            return None
        elif type_ == "msgpack":
            return ormsgpack.unpackb(data_, ext_hook=self._unpack_ext_hook)
        # ... handle other types


# Extension types for complex objects:
EXT_CONSTRUCTOR_SINGLE_ARG = 0   # UUID, Decimal, set, frozenset, Enum
EXT_CONSTRUCTOR_POS_ARGS = 1    # pathlib.Path, date, timedelta, re.Pattern
EXT_CONSTRUCTOR_KW_ARGS = 2     # namedtuple, dataclass
EXT_METHOD_SINGLE_ARG = 3       # datetime.fromisoformat
EXT_PYDANTIC_V1 = 4             # Pydantic v1 models
EXT_PYDANTIC_V2 = 5             # Pydantic v2 models
EXT_NUMPY_ARRAY = 6             # NumPy arrays


def _msgpack_default(obj: Any) -> str | ormsgpack.Ext:
    """Custom serializer for complex types."""
    
    # Pydantic v2 models
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return ormsgpack.Ext(
            EXT_PYDANTIC_V2,
            _msgpack_enc((
                obj.__class__.__module__,
                obj.__class__.__name__,
                obj.model_dump(),
                "model_validate_json",
            )),
        )
    
    # Dataclasses
    elif dataclasses.is_dataclass(obj):
        return ormsgpack.Ext(
            EXT_CONSTRUCTOR_KW_ARGS,
            _msgpack_enc((
                obj.__class__.__module__,
                obj.__class__.__name__,
                {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)},
            )),
        )
    
    # NumPy arrays (with buffer)
    elif isinstance(obj, np.ndarray):
        order = "F" if obj.flags.f_contiguous else "C"
        meta = (obj.dtype.str, obj.shape, order, memoryview(obj) if obj.flags.c_contiguous else obj.tobytes())
        return ormsgpack.Ext(EXT_NUMPY_ARRAY, _msgpack_enc(meta))
    
    # ... handle UUID, datetime, Enum, etc.
```

**Key Insight for Dynamiq:** We can use similar patterns for serializing our Pydantic-based Messages and node state.

### 2.4 Human-in-the-Loop: The `interrupt()` Function

```python
# From langgraph/types.py

@final
@dataclass(init=False, slots=True)
class Interrupt:
    """Information about an interrupt that occurred in a node."""
    
    value: Any
    """The value associated with the interrupt (e.g., question to user)."""
    
    id: str
    """The ID of the interrupt. Can be used to resume directly."""


def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.
    
    The interrupt function enables human-in-the-loop workflows by:
    1. First invocation: Raises GraphInterrupt, halts execution, surfaces value to client
    2. Client resumes with Command(resume=...) 
    3. Subsequent invocations: Returns the resume value, continues execution
    
    Multiple interrupts in same node are matched by ORDER - resume values are
    consumed in the order interrupts were called.
    
    REQUIRES: Checkpointer must be enabled!
    """
    from langgraph.config import get_config
    from langgraph.errors import GraphInterrupt
    
    conf = get_config()["configurable"]
    scratchpad = conf[CONFIG_KEY_SCRATCHPAD]
    idx = scratchpad.interrupt_counter()  # Track interrupt index
    
    # Find previous resume values
    if scratchpad.resume:
        if idx < len(scratchpad.resume):
            conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
            return scratchpad.resume[idx]  # Return resume value
    
    # Find current resume value
    v = scratchpad.get_null_resume(True)
    if v is not None:
        scratchpad.resume.append(v)
        conf[CONFIG_KEY_SEND]([(RESUME, scratchpad.resume)])
        return v
    
    # No resume value found - raise interrupt
    raise GraphInterrupt((
        Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS]),
    ))


@dataclass(**_DC_KWARGS)
class Command(Generic[N], ToolOutputMixin):
    """Commands to update state and send messages to nodes.
    
    Used to resume from interrupts with Command(resume=...).
    """
    
    graph: str | None = None        # Target graph (None=current, PARENT=parent)
    update: Any | None = None       # State update to apply
    resume: dict[str, Any] | Any | None = None  # Resume value for interrupt
    goto: Send | Sequence[Send | N] | N = ()    # Next node(s) to execute
```

**Critical Pattern:** The `interrupt()` function uses a counter-based system to match resume values to interrupts. This allows multiple interrupts in the same node.

### 2.5 PostgreSQL Backend Implementation

```python
# From langgraph/checkpoint/postgres/__init__.py

class PostgresSaver(BasePostgresSaver):
    """Checkpointer that stores checkpoints in a Postgres database."""
    
    # SQL Migrations (run once on setup)
    MIGRATIONS = [
        """CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY)""",
        """CREATE TABLE IF NOT EXISTS checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            checkpoint JSONB NOT NULL,
            metadata JSONB NOT NULL,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        )""",
        """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            channel TEXT NOT NULL,
            version TEXT NOT NULL,
            type TEXT NOT NULL,
            blob BYTEA,
            PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
        )""",
        """CREATE TABLE IF NOT EXISTS checkpoint_writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            blob BYTEA NOT NULL,
            task_path TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
        )""",
        # ... additional migrations for indexes, columns
    ]
    
    def setup(self) -> None:
        """Set up the checkpoint database. MUST be called on first use."""
        with self._cursor() as cur:
            cur.execute(self.MIGRATIONS[0])
            # Run migrations based on current version
            results = cur.execute("SELECT v FROM checkpoint_migrations ORDER BY v DESC LIMIT 1")
            row = results.fetchone()
            version = row["v"] if row else -1
            
            for v, migration in zip(range(version + 1, len(self.MIGRATIONS)), self.MIGRATIONS[version + 1:]):
                cur.execute(migration)
                cur.execute("INSERT INTO checkpoint_migrations (v) VALUES (%s)", (v,))
    
    def put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig:
        """Save a checkpoint to the database."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        
        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        
        # Inline primitive values in checkpoint table
        # Store complex values in blobs table
        blob_values = {}
        for k, v in checkpoint["channel_values"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                pass  # Keep inline
            else:
                blob_values[k] = copy["channel_values"].pop(k)
        
        with self._cursor(pipeline=True) as cur:
            # Store blobs separately
            if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
                cur.executemany(
                    self.UPSERT_CHECKPOINT_BLOBS_SQL,
                    self._dump_blobs(thread_id, checkpoint_ns, blob_values, blob_versions),
                )
            
            # Store checkpoint metadata
            cur.execute(
                self.UPSERT_CHECKPOINTS_SQL,
                (thread_id, checkpoint_ns, checkpoint["id"], checkpoint_id,
                 Jsonb(copy), Jsonb(get_serializable_checkpoint_metadata(config, metadata))),
            )
        
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint["id"]}}
```

**Key Design Patterns:**
- **Separate blobs table:** Large channel values stored in `checkpoint_blobs`, not inline
- **Migration system:** `checkpoint_migrations` tracks schema version
- **Pipeline mode:** Uses PostgreSQL pipeline for batching
- **Thread deletion:** Cascade delete across all tables

---

## 3. Letta Deep Dive (Agent Serialization Excellence)

**Repository:** `letta-ai/letta`  
**Key Insight:** Letta has the most sophisticated agent export/import with ID remapping and validation.

### 3.1 ID Remapping Pattern

```python
# From letta/services/agent_serialization_manager.py

class AgentSerializationManager:
    """Manages export/import of agent files with ID remapping."""
    
    def __init__(self, ...):
        # ID mapping state for export
        self._db_to_file_ids: Dict[str, str] = {}
        
        # Counters for generating Stripe-style IDs
        self._id_counters: Dict[str, int] = {
            AgentSchema.__id_prefix__: 0,      # "agent"
            GroupSchema.__id_prefix__: 0,      # "group"
            BlockSchema.__id_prefix__: 0,      # "block"
            FileSchema.__id_prefix__: 0,       # "file"
            SourceSchema.__id_prefix__: 0,     # "source"
            ToolSchema.__id_prefix__: 0,       # "tool"
            MessageSchema.__id_prefix__: 0,    # "message"
            FileAgentSchema.__id_prefix__: 0,  # "file_agent"
            MCPServerSchema.__id_prefix__: 0,  # "mcp_server"
        }
    
    def _generate_file_id(self, entity_type: str) -> str:
        """Generate a Stripe-style ID: 'agent-0', 'tool-1', etc."""
        counter = self._id_counters[entity_type]
        file_id = f"{entity_type}-{counter}"
        self._id_counters[entity_type] += 1
        return file_id
    
    def _map_db_to_file_id(self, db_id: str, entity_type: str, allow_new: bool = True) -> str:
        """Map a database UUID to a file ID, creating if needed."""
        if db_id in self._db_to_file_ids:
            return self._db_to_file_ids[db_id]
        
        if not allow_new:
            raise AgentExportIdMappingError(db_id, entity_type)
        
        file_id = self._generate_file_id(entity_type)
        self._db_to_file_ids[db_id] = file_id
        return file_id
```

**Why This Matters:** Clean, human-readable IDs in exported files. No UUID collision issues on import.

### 3.2 Export Flow with Dependency Resolution

```python
async def export(self, agent_ids: List[str], actor: User) -> AgentFileSchema:
    """Export agents and related entities to portable format."""
    self._reset_state()
    
    agent_states = await self.agent_manager.get_agents_by_ids_async(agent_ids, actor)
    
    # === 1. Resolve Multi-Agent Groups ===
    groups = []
    group_agent_ids = []
    for agent_state in agent_states:
        if agent_state.multi_agent_group:
            groups.append(agent_state.multi_agent_group)
            group_agent_ids.extend(agent_state.multi_agent_group.agent_ids)
    
    # Add group member agents if not already included
    group_agent_ids = list(set(group_agent_ids) - set(agent_ids))
    if group_agent_ids:
        agent_states.extend(await self.agent_manager.get_agents_by_ids_async(group_agent_ids, actor))
    
    # === 2. Extract Unique Entities ===
    tool_set = self._extract_unique_tools(agent_states)
    block_set = self._extract_unique_blocks(agent_states)
    mcp_server_set = await self._extract_unique_mcp_servers(tool_set, actor)
    source_set, file_set = await self._extract_unique_sources_and_files_from_agents(agent_states, actor)
    
    # === 3. Map MCP Server IDs FIRST (tools depend on them) ===
    for mcp_server in mcp_server_set:
        self._map_db_to_file_id(mcp_server.id, MCPServerSchema.__id_prefix__)
    
    # === 4. Convert to Schemas with ID Remapping ===
    agent_schemas = [await self._convert_agent_state_to_schema(agent_state, actor) for agent_state in agent_states]
    tool_schemas = [self._convert_tool_to_schema(tool) for tool in tool_set]
    block_schemas = [self._convert_block_to_schema(block) for block in block_set]
    source_schemas = [self._convert_source_to_schema(source) for source in source_set]
    file_schemas = [self._convert_file_to_schema(file_metadata) for file_metadata in file_set]
    mcp_server_schemas = [self._convert_mcp_server_to_schema(mcp_server) for mcp_server in mcp_server_set]
    group_schemas = [self._convert_group_to_schema(group) for group in groups]
    
    # === 5. Return Complete Schema ===
    return AgentFileSchema(
        agents=agent_schemas,
        groups=group_schemas,
        blocks=block_schemas,
        files=file_schemas,
        sources=source_schemas,
        tools=tool_schemas,
        mcp_servers=mcp_server_schemas,
        metadata={"revision_id": await get_latest_alembic_revision()},
        created_at=datetime.now(timezone.utc),
    )
```

### 3.3 Comprehensive Validation

```python
def _validate_schema(self, schema: AgentFileSchema):
    """Validate schema before import."""
    errors = []
    
    # 1. ID Format Validation (must be "entity_type-N" format)
    errors.extend(self._validate_id_format(schema))
    
    # 2. Duplicate ID Detection (within and across entity types)
    errors.extend(self._validate_duplicate_ids(schema))
    
    # 3. File → Source Reference Validation
    errors.extend(self._validate_file_source_references(schema))
    
    # 4. FileAgent → File/Source/Agent Reference Validation
    errors.extend(self._validate_file_agent_references(schema))
    
    if errors:
        raise AgentFileImportError(f"Schema validation failed: {'; '.join(errors)}")


def _validate_id_format(self, schema: AgentFileSchema) -> List[str]:
    """Validate that all IDs follow expected format."""
    errors = []
    
    entity_checks = [
        (schema.agents, AgentSchema.__id_prefix__),
        (schema.groups, GroupSchema.__id_prefix__),
        (schema.blocks, BlockSchema.__id_prefix__),
        # ... all entity types
    ]
    
    for entities, expected_prefix in entity_checks:
        for entity in entities:
            if not entity.id.startswith(f"{expected_prefix}-"):
                errors.append(f"Invalid ID format: {entity.id} should start with '{expected_prefix}-'")
            else:
                # Check suffix is valid integer
                try:
                    suffix = entity.id[len(expected_prefix) + 1:]
                    int(suffix)
                except ValueError:
                    errors.append(f"Invalid ID format: {entity.id} should have integer suffix")
    
    return errors
```

### 3.4 Import with Dependency Order

```python
async def import_file(self, schema: AgentFileSchema, actor: User, ...) -> ImportResult:
    """Import in dependency order: MCP → Tools → Blocks → Sources → Files → Agents → Messages → FileAgents → Groups"""
    
    self._reset_state()
    self._validate_schema(schema)
    
    file_to_db_ids = {}  # Maps file IDs to new database IDs
    
    # 1. MCP Servers first (tools depend on them)
    for mcp_server_schema in schema.mcp_servers:
        created = await self.mcp_manager.create_or_update_mcp_server(...)
        file_to_db_ids[mcp_server_schema.id] = created.id
    
    # 2. Tools (may depend on MCP servers) - BULK UPSERT
    pydantic_tools = [Tool(**t.model_dump(exclude={"id"})) for t in schema.tools]
    created_tools = await self.tool_manager.bulk_upsert_tools_async(pydantic_tools, actor)
    # Map by name since tools are matched by name during upsert
    for tool_schema in schema.tools:
        created = created_tools_by_name.get(tool_schema.name)
        if created:
            file_to_db_ids[tool_schema.id] = created.id
    
    # 3. Blocks (no dependencies) - BATCH CREATE
    pydantic_blocks = [Block(**b.model_dump(exclude={"id"})) for b in schema.blocks]
    created_blocks = await self.block_manager.batch_create_blocks_async(pydantic_blocks, actor)
    for block_schema, created_block in zip(schema.blocks, created_blocks):
        file_to_db_ids[block_schema.id] = created_block.id
    
    # 4. Sources, 5. Files, 6. Agents, 7. Messages, 8. FileAgents, 9. Groups
    # ... (each with proper ID remapping using file_to_db_ids)
    
    # Start background file processing
    for file_schema in schema.files:
        if file_schema.content:
            safe_create_task(self._process_file_async(file_metadata, source_id, file_processor, actor))
    
    return ImportResult(success=True, imported_count=imported_count, id_mappings=file_to_db_ids)
```

**Key Insight for Dynamiq:** The import order matters! We must restore entities in dependency order.

---

## 4. Google ADK Deep Dive (Session-Based Simplicity)

**Repository:** `google/adk-python`  
**Key Insight:** ADK uses a simpler session-based model focused on conversational agents.

### 4.1 Session Model

```python
# From google/adk/sessions/session.py

class Session(BaseModel):
    """Represents a series of interactions between a user and agents."""
    
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True,
        alias_generator=alias_generators.to_camel,  # JSON uses camelCase
        populate_by_name=True,
    )
    
    id: str
    """The unique identifier of the session."""
    
    app_name: str
    """The name of the app."""
    
    user_id: str
    """The id of the user."""
    
    state: dict[str, Any] = Field(default_factory=dict)
    """The state of the session (merged app + user + session state)."""
    
    events: list[Event] = Field(default_factory=list)
    """The events of the session: user input, model response, function call/response, etc."""
    
    last_update_time: float = 0.0
    """The last update time of the session (for stale detection)."""
```

### 4.2 State Hierarchy (App → User → Session)

```python
# ADK has THREE state levels that merge together:

class State:
    """State prefixes for different scopes."""
    APP_PREFIX = "app:"     # Shared across all sessions
    USER_PREFIX = "user:"   # Shared across user's sessions
    TEMP_PREFIX = "temp:"   # Temporary, not persisted


def _merge_state(app_state: dict, user_state: dict, session_state: dict) -> dict:
    """Merge app, user, and session states into a single state dictionary."""
    merged_state = copy.deepcopy(session_state)
    for key in app_state.keys():
        merged_state[State.APP_PREFIX + key] = app_state[key]
    for key in user_state.keys():
        merged_state[State.USER_PREFIX + key] = user_state[key]
    return merged_state
```

**Key Pattern:** Prefix-based state isolation. `app:config` is global, `user:preferences` is per-user, everything else is session-local.

### 4.3 Event Appending with State Updates

```python
# From google/adk/sessions/base_session_service.py

class BaseSessionService(abc.ABC):
    
    async def append_event(self, session: Session, event: Event) -> Event:
        """Appends an event to a session object."""
        if event.partial:
            return event  # Don't persist partial events
        
        event = self._trim_temp_delta_state(event)  # Remove temp:* keys
        self._update_session_state(session, event)  # Apply state delta
        session.events.append(event)
        return event
    
    def _update_session_state(self, session: Session, event: Event) -> None:
        """Updates the session state based on the event."""
        if not event.actions or not event.actions.state_delta:
            return
        for key, value in event.actions.state_delta.items():
            if key.startswith(State.TEMP_PREFIX):
                continue  # Skip temporary state
            session.state.update({key: value})
```

### 4.4 Database Backend with Schema Versioning

```python
# From google/adk/sessions/database_session_service.py

class DatabaseSessionService(BaseSessionService):
    """Session service using SQLAlchemy with PostgreSQL/SQLite."""
    
    def __init__(self, db_url: str, **kwargs):
        # Create async engine
        self.db_engine = create_async_engine(db_url, **kwargs)
        
        # Schema version checking
        self._db_schema_version: Optional[str] = None
        self._table_creation_lock = asyncio.Lock()
    
    async def _prepare_tables(self):
        """Ensure database tables are ready (lazy initialization)."""
        if self._db_schema_version is not None:
            return
        
        async with self._db_schema_lock:
            # Check schema version
            async with self.db_engine.connect() as conn:
                self._db_schema_version = await conn.run_sync(
                    _schema_check_utils.get_db_schema_version_from_connection
                )
            
            # Create tables based on version
            async with self.db_engine.begin() as conn:
                if self._db_schema_version == LATEST_SCHEMA_VERSION:
                    await conn.run_sync(BaseV1.metadata.create_all)
                else:
                    await conn.run_sync(BaseV0.metadata.create_all)
    
    async def append_event(self, session: Session, event: Event) -> Event:
        """Persist event with stale session detection."""
        await self._prepare_tables()
        
        async with self.database_session_factory() as sql_session:
            storage_session = await sql_session.get(...)
            
            # Stale session detection
            if storage_session.update_timestamp_tz > session.last_update_time:
                raise ValueError("Stale session - please refresh and retry")
            
            # Extract and apply state deltas
            state_deltas = extract_state_delta(event.actions.state_delta)
            if state_deltas["app"]:
                storage_app_state.state = storage_app_state.state | state_deltas["app"]
            if state_deltas["user"]:
                storage_user_state.state = storage_user_state.state | state_deltas["user"]
            if state_deltas["session"]:
                storage_session.state = storage_session.state | state_deltas["session"]
            
            # Persist event
            sql_session.add(StorageEvent.from_event(session, event))
            await sql_session.commit()
```

**Key Insight:** ADK's stale session detection (`last_update_time` comparison) prevents race conditions.

---

## 5. Manus AI / OpenManus Investigation

### 5.1 Why Manus AI Matters

Manus AI has emerged as a leading example of production-grade AI agents capable of:
- Completing complex multi-step tasks autonomously
- Running for extended periods (hours) without failure
- Handling interruptions and resuming gracefully
- Multi-device interaction (start on web, continue on mobile)

This makes them a natural reference for checkpoint/resume design.

### 5.2 What We Found

#### OpenManus Repository

We attempted to clone and analyze [OpenManus](https://github.com/manusai/OpenManus):

```bash
$ git clone https://github.com/manusai/OpenManus
# Result: Repository exists but contains only README.md
# No actual checkpoint/state management code available
```

**Finding:** OpenManus is essentially an empty repository (placeholder). The actual Manus AI implementation is **proprietary/closed-source**.

#### TiDB Case Study (Public Documentation)

From Manus AI's public case study with TiDB Cloud:

| Aspect | Details |
|--------|---------|
| **Database** | TiDB Cloud (distributed MySQL-compatible) |
| **Key Feature** | High write throughput for frequent state saves |
| **Scale** | Millions of users, concurrent long-running agents |
| **State Pattern** | Event sourcing with periodic compaction |

**Key Quote (paraphrased):** *"Manus uses frequent checkpointing to ensure no work is lost. TiDB's distributed nature handles the write load."*

### 5.3 Inferred Architecture from Manus Behavior

By observing Manus AI's behavior, we can infer their checkpoint approach:

| Behavior | Implied Implementation |
|----------|----------------------|
| Seamless browser close/reopen | Server-side checkpoint persistence |
| Continue on different device | Checkpoint ID independent of client |
| Progress survives 24+ hours | Durable storage (not just Redis TTL) |
| "Context engineering" messaging | Structured state with relevance ranking |
| Reflective memory updates | Event-based state deltas |

### 5.4 What We're Adopting from Manus Philosophy

Even without code access, Manus's philosophy informs our design:

1. **Frequent Checkpoints** - Don't wait for failures; checkpoint proactively
2. **Client-Agnostic IDs** - Checkpoint ID travels with user, not session
3. **Graceful Degradation** - If context is too long, summarize intelligently
4. **Multi-Device by Default** - Design for continuation anywhere

### 5.5 Gap Analysis vs Manus

| Manus Capability | Our Proposed Solution | Gap |
|------------------|----------------------|-----|
| Real-time checkpoint | ✅ After each node | None |
| Multi-device resume | ✅ Via run_id + checkpoint_id | None |
| Hours-long execution | ✅ Connection refresh pattern | None |
| "Reflective memory" | ⚠️ Via Memory node, not automatic | Partial |
| Context summarization | ✅ Via Agent summarization_config | None |
| Distributed database | ⚠️ PostgreSQL (can scale with read replicas) | Minor |

**Conclusion:** While we can't directly adopt Manus code, our design achieves comparable capabilities for checkpoint/resume.

---

## 6. Framework Comparison Matrix (Detailed)

| Feature | LangGraph | Letta | Google ADK | CrewAI | Metaflow | Manus AI* | **Dynamiq** |
|---------|-----------|-------|------------|--------|----------|-----------|-------------|
| **Granularity** | Node | Agent | Session | Method | Run/Task | Unknown | Node + loop |
| **State Model** | Channels | Memory | Hierarchy | Dict | Artifacts | Events | NodeState |
| **HITL** | `interrupt()` | N/A | Events | Exception | N/A | Unknown | `PENDING_INPUT` |
| **Serialization** | ormsgpack | Pydantic | JSON | JSON | Pickle | Unknown | JSON+Pydantic |
| **Time Travel** | ✅ Full | ✅ Export | ❌ | ⚠️ Limited | ✅ Clone | Unknown | ✅ Full |
| **Parallel** | ✅ Channels | ❌ | ❌ | ⚠️ Basic | ✅ Foreach | Unknown | ✅ Map |
| **Isolation Key** | `thread_id` | `agent_id` | `app_name + user_id + session_id` | `flow_uuid` | `run_id` | `run_id + flow_id` |
| **Prod Backend** | PostgreSQL | PostgreSQL | PG/SQLite | SQLite | S3 | TiDB | PG/Redis |
| **Async** | ✅ Full | ✅ Full | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full | ✅ Full |
| **Validation** | Basic | Comprehensive | Basic | Basic | Basic | Unknown | Comprehensive |
| **Migrations** | ✅ SQL | ✅ Alembic | ✅ Schema | ❌ | ❌ | Unknown | ✅ Alembic |

*\*Manus AI is closed-source; data inferred from public documentation and observed behavior.*

---

## 6b. Extended Comparison (Additional Competitors)

| Feature | Temporal | AutoGen/AF | Prefect | Haystack | Bedrock | **Dynamiq** |
|---------|----------|------------|---------|----------|---------|-------------|
| **Approach** | Event-source | Supersteps | Result persist | Breakpoints | Session API | Node state |
| **Granularity** | Event | Superstep | Task | Component | Session | Node + loop |
| **Auto Recovery** | ✅ Full replay | ✅ Automatic | ⚠️ Manual config | ✅ Snapshot | ✅ Session | ✅ Full |
| **HITL** | ✅ Signals | ✅ Native | ⚠️ Manual | ⚠️ Breakpoint | ✅ Native | ✅ Native |
| **Encryption** | ⚠️ Config | ⚠️ Config | ⚠️ Config | ❌ | ✅ KMS | ⚠️ v2 |
| **Cloud Native** | ✅ Temporal Cloud | ✅ Azure | ✅ Prefect Cloud | ❌ | ✅ AWS | ⚠️ Self-host |

**Key Differences:**
- **Temporal**: Most robust (event-sourced) but heaviest; requires dedicated infrastructure
- **Microsoft AutoGen**: Similar superstep concept to our node boundaries
- **Prefect**: Per-task opt-in like ours (`persist_result=True` ≈ `checkpoint_enabled=True`)
- **Haystack**: Breakpoint/snapshot pattern very close to our design
- **Bedrock**: Enterprise-grade with encryption; informs our security roadmap

---

## 7. Key Patterns We're Adopting

### From LangGraph
1. ✅ **Checkpoint at node boundaries** (super-steps)
2. ✅ **`versions_seen` tracking** for determining which nodes to re-execute
3. ✅ **`put_writes` for intermediate state** (partial results during node execution)
4. ✅ **Separate blobs table** for large values
5. ✅ **Migration system** for schema evolution
6. ✅ **`interrupt()` pattern** adapted for our `PENDING_INPUT` status

### From Temporal
1. ✅ **"Never lose progress" principle** - Checkpoint frequently
2. ✅ **Continue-As-New concept** - For long-running workflows, checkpoint and continue fresh
3. ⚠️ **Event sourcing** - Too heavy for us, but we adopt atomic state snapshots

### From Microsoft AutoGen / Agent Framework
1. ✅ **Superstep boundaries** - Matches our node boundary checkpointing
2. ✅ **State isolation** - Separate executor-local vs shared state
3. ✅ **Opt-in migration path** - Manual → automatic (matches our approach)

### From Prefect
1. ✅ **Per-task persistence opt-in** (`persist_result=True`)
2. ✅ **Cache policies** - Informs our tool cache in Agent
3. ✅ **State machine** - Clear states (`Running`, `Failed`, `Completed`)

### From Haystack
1. ✅ **Breakpoint class** - Validates our checkpoint concept
2. ✅ **JSON snapshot files** - Validates our serialization approach
3. ✅ **Auto-snapshot on failure** - We adopt this (`status=FAILED` with checkpoint)

### From Amazon Bedrock
1. ✅ **Short-term vs long-term memory** - Checkpoint ≠ Memory (different concerns)
2. ⚠️ **KMS encryption** - Informs our v2 security roadmap
3. ✅ **Session Management APIs** - Validates our runtime integration approach

### From Letta
1. ✅ **Comprehensive validation** before operations
2. ✅ **Dependency-ordered import/restore** (respecting entity relationships)
3. ✅ **Background processing** for non-blocking operations
4. ⚠️ **ID remapping** (optional - we can use UUIDs directly)

### From Google ADK
1. ✅ **Stale detection** via `last_update_time`
2. ✅ **State hierarchy** (consider app/user/session prefixes for multi-tenant)
3. ✅ **Lazy table creation** with schema versioning
4. ✅ **Event-based state updates**

### From CrewAI
1. ✅ **Separate `pending_feedback` table/status** for HITL
2. ✅ **Exception-based HITL pause** (adapted to our model)

### From Metaflow
1. ✅ **Clone-based resume** - never re-execute completed nodes
2. ✅ **Selective re-execution** (`steps_to_rerun` pattern)

---

## 8. What We're NOT Adopting (and Why)

| Pattern | Framework | Why Not |
|---------|-----------|---------|
| **Channel-based state** | LangGraph | Dynamiq uses DAG execution, not channels |
| **Full ID remapping** | Letta | Complexity not needed for single-system use |
| **Pickle serialization** | Metaflow | Security risk, not portable |
| **State prefixes** | ADK | Over-engineering for our use case |
| **Method decorators** | CrewAI | We prefer explicit node-level control |

---

## 9. Implementation Priority

Based on this research, our implementation phases are:

| Phase | Component | Inspiration From |
|-------|-----------|------------------|
| 1 | Core models + File backend | All frameworks |
| 2 | Agent checkpoint support | LangGraph interrupt pattern |
| 3 | PostgreSQL backend | LangGraph PostgresSaver |
| 4 | Validation + Migration | Letta validation, LangGraph migrations |
| 5 | HITL integration | CrewAI pending_feedback + existing runtime |

---

## 10. How Dynamiq Compares: Executive Summary

### 10.1 Where We Excel

| Advantage | Details | vs Competition |
|-----------|---------|----------------|
| **Agent Loop Checkpointing** | Mid-loop checkpoint inside agents | LangGraph only checkpoints at node boundaries |
| **Map Node Parallelism** | Track completed iterations, only re-run failed | Metaflow requires full restart of foreach |
| **Existing HITL Infrastructure** | Leverage runtime's `hitl_input_pump`, `RunInputEvent` table | CrewAI requires new infrastructure |
| **Unified DAG Model** | Single checkpoint covers entire DAG state | LangGraph's channel model is more complex |
| **Pydantic-Native** | Full Pydantic serialization, already in codebase | LangGraph needs custom serializers |
| **Flexible Backends** | File (dev), SQLite (test), Redis (cache), PostgreSQL (prod) | Most frameworks support 1-2 backends |

### 10.2 Where We Match Industry Leaders

| Capability | Dynamiq | LangGraph | Letta | Google ADK |
|------------|---------|-----------|-------|------------|
| Time travel to any checkpoint | ✅ | ✅ | ✅ | ❌ |
| Full async support | ✅ | ✅ | ✅ | ✅ |
| Production-grade storage | ✅ PostgreSQL | ✅ PostgreSQL | ✅ PostgreSQL | ✅ PostgreSQL |
| HITL with pause/resume | ✅ | ✅ | ⚠️ Limited | ✅ |
| Schema migrations | ✅ Alembic | ✅ SQL | ✅ Alembic | ✅ Versioned |
| Comprehensive validation | ✅ | ⚠️ Basic | ✅ | ⚠️ Basic |

### 10.3 Unique Dynamiq Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│              DYNAMIQ CHECKPOINT DIFFERENTIATORS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. MID-AGENT LOOP CHECKPOINTING                               │
│     Agent at loop 5/15? Checkpoint and resume at loop 6.       │
│     No other framework does this.                              │
│                                                                 │
│  2. ITERATION-LEVEL MAP CHECKPOINTING                          │
│     100 items, 50 done, failure at 51?                         │
│     Resume only items 51-100, keep 1-50 results.               │
│                                                                 │
│  3. NESTED HITL SUPPORT                                        │
│     Agent → Tool → HumanFeedback works seamlessly.             │
│     Orchestrator → Agent → Tool → HumanFeedback works too.     │
│                                                                 │
│  4. EXISTING RUNTIME INTEGRATION                               │
│     No new WebSocket endpoints needed.                         │
│     Checkpoints complement existing HITL infrastructure.       │
│                                                                 │
│  5. YAML + CHECKPOINT SYNERGY                                  │
│     Workflow from YAML + State from checkpoint = Full restore  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 10.4 Known Gaps (Honest Assessment)

| Gap | Details | Mitigation |
|-----|---------|------------|
| **No ormsgpack optimization** | LangGraph's binary serialization is faster | JSON is portable; optimize later if needed |
| **No channel-based state** | LangGraph's channels enable fine-grained control | Our DAG model is simpler and sufficient |
| **No "reflective memory"** | Manus AI automatically summarizes context | Can be added via Memory node configuration |
| **No distributed storage native** | Manus uses TiDB for horizontal scale | PostgreSQL with read replicas handles most cases |

### 10.5 Competitive Positioning Summary

```
                    CHECKPOINT SOPHISTICATION
                              ▲
                              │
              Manus AI ◆      │      ◆ Dynamiq (Proposed)
              (inferred)      │         
                              │      
                   LangGraph ◆│
                              │
                              │   ◆ Letta
                    CrewAI ◆  │
                              │
               Metaflow ◆     │      ◆ Google ADK
                              │
                              └──────────────────────────▶
                                  PRODUCTION MATURITY

Legend:
◆ Framework position (approximate)
```

**Dynamiq's Position:** We achieve near-LangGraph sophistication with better agent-level granularity, while maintaining Dynamiq's simpler DAG execution model.

---

**Previous:** [01-EXECUTIVE-SUMMARY.md](./01-EXECUTIVE-SUMMARY.md)  
**Next:** [03-RUNTIME-INTEGRATION.md](./03-RUNTIME-INTEGRATION.md)
