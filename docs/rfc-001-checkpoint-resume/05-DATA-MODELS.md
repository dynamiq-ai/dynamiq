# RFC-001-05: Data Models

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 5 of 9

---

## 1. Overview

This document defines the Pydantic models, protocols, and serialization strategies for checkpoint/resume.

**Design Principles:**
1. Use Pydantic for type safety and validation
2. JSON-serializable by default (for all backends)
3. Minimal state - only what's needed for resume
4. Forward-compatible versioning

---

## 2. Checkpoint Protocol

### 2.1 CheckpointableNode Protocol

```python
# dynamiq/checkpoint/protocol.py

from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class CheckpointableNode(Protocol):
    """
    Protocol for nodes that support checkpointing.
    
    Nodes implementing this protocol can have their internal state
    saved to and restored from checkpoints.
    
    Example:
        >>> class MyNode(Node, CheckpointMixin):
        ...     _counter: int = 0
        ...     
        ...     def get_checkpoint_state(self) -> dict:
        ...         return {"counter": self._counter}
        ...     
        ...     def restore_from_checkpoint(self, state: dict) -> None:
        ...         self._counter = state.get("counter", 0)
        ...         self._is_resumed = True
    """
    
    def get_checkpoint_state(self) -> dict[str, Any]:
        """
        Extract serializable state for checkpointing.
        
        Returns:
            Dictionary containing all state needed for resume.
            Must be JSON-serializable (primitives, lists, dicts).
            
        Note:
            Do NOT include:
            - Configuration that's already in workflow definition
            - External resource handles (connections, clients)
            - Large transient data that can be reconstructed
        """
        ...
    
    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """
        Restore internal state from a checkpoint.
        
        Args:
            state: Dictionary previously returned by get_checkpoint_state()
            
        Note:
            After restoration, the node should be able to continue
            execution from where it left off.
        """
        ...
```

### 2.2 CheckpointMixin

```python
# dynamiq/checkpoint/protocol.py

class CheckpointMixin:
    """
    Default checkpoint implementation for nodes.
    
    Provides default no-op implementations that can be overridden.
    Also provides the _is_resumed flag for nodes to check.
    
    Usage:
        class MyNode(Node, CheckpointMixin):
            # Node automatically gets checkpoint support
            pass
    """
    
    _is_resumed: bool = False
    
    def get_checkpoint_state(self) -> dict[str, Any]:
        """Default: no internal state to checkpoint."""
        return {}
    
    def restore_from_checkpoint(self, state: dict[str, Any]) -> None:
        """Default: just mark as resumed."""
        self._is_resumed = True
    
    @property
    def is_resumed(self) -> bool:
        """Check if this node was restored from a checkpoint."""
        return self._is_resumed
    
    def reset_resumed_flag(self) -> None:
        """Reset the resumed flag after handling resume logic."""
        self._is_resumed = False
```

---

## 3. Checkpoint Status

```python
# dynamiq/checkpoint/models.py

from enum import Enum

class CheckpointStatus(str, Enum):
    """
    Status of a checkpoint.
    
    Attributes:
        ACTIVE: Normal execution, checkpoint created after node
        PAUSED: Intentionally paused (e.g., for debugging)
        COMPLETED: Workflow completed successfully
        FAILED: Workflow failed, checkpoint preserved for retry
        PENDING_INPUT: Waiting for human input (HITL)
    """
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING_INPUT = "pending_input"
```

---

## 4. Node Checkpoint State

```python
# dynamiq/checkpoint/models.py

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

class NodeCheckpointState(BaseModel):
    """
    Checkpoint state for a single node.
    
    Captures both the execution result (input/output) and
    any internal state specific to the node type.
    
    Attributes:
        node_id: Unique identifier of the node
        node_type: Class name of the node (for debugging)
        status: Execution status (success, failure, skip, etc.)
        input_data: Input data passed to the node
        output_data: Output data from the node
        error: Error information if failed
        internal_state: Node-specific state from get_checkpoint_state()
        started_at: When node execution started
        completed_at: When node execution completed
    """
    
    node_id: str
    node_type: str
    status: str  # RunnableStatus value
    
    # Execution data
    input_data: Any | None = None
    output_data: Any | None = None
    error: dict | None = None
    
    # Node-specific internal state (from get_checkpoint_state)
    internal_state: dict = Field(default_factory=dict)
    
    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    class Config:
        extra = "allow"  # Allow additional fields for future compatibility
```

---

## 5. Flow Checkpoint

```python
# dynamiq/checkpoint/models.py

from dynamiq.utils import generate_uuid

class PendingInputContext(BaseModel):
    """
    Context for a workflow waiting for human input.
    
    Captures all information needed to resume after
    human provides input.
    
    Attributes:
        node_id: ID of the node waiting for input
        prompt: The question/prompt shown to user
        timestamp: When input was requested
        metadata: Additional context (tool name, etc.)
    """
    node_id: str
    prompt: str
    timestamp: datetime
    metadata: dict = Field(default_factory=dict)


class FlowCheckpoint(BaseModel):
    """
    Complete checkpoint for a flow execution.
    
    This is the top-level model that captures the entire
    state of a workflow at a point in time.
    
    Attributes:
        id: Unique checkpoint identifier
        flow_id: ID of the Flow being executed
        workflow_id: ID of the parent Workflow (if any)
        run_id: Runtime execution ID
        status: Current checkpoint status
        node_states: State of each node, keyed by node_id
        completed_node_ids: List of nodes that have completed
        pending_node_ids: List of nodes waiting to execute
        original_input: Input data for resume
        original_config: RunnableConfig for resume
        pending_input_context: HITL context if waiting for input
        created_at: When checkpoint was created
        updated_at: When checkpoint was last updated
        version: Schema version for forward compatibility
        dynamiq_version: Library version
        metadata: Additional debugging info
        parent_checkpoint_id: For checkpoint chains (time travel)
    """
    
    # Identification
    id: str = Field(default_factory=generate_uuid)
    flow_id: str
    workflow_id: str | None = None
    run_id: str
    
    # State
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    node_states: dict[str, NodeCheckpointState] = Field(default_factory=dict)
    completed_node_ids: list[str] = Field(default_factory=list)
    pending_node_ids: list[str] = Field(default_factory=list)
    
    # Original execution context (for resume)
    original_input: Any = None
    original_config: dict | None = None
    
    # HITL support
    pending_input_context: PendingInputContext | None = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Versioning
    version: str = "1.0"
    dynamiq_version: str | None = None
    
    # Metadata
    metadata: dict = Field(default_factory=dict)
    parent_checkpoint_id: str | None = None
    
    def mark_node_complete(
        self, 
        node_id: str, 
        state: NodeCheckpointState
    ) -> None:
        """
        Mark a node as completed and store its state.
        
        Args:
            node_id: The node that completed
            state: The node's checkpoint state
        """
        self.node_states[node_id] = state
        
        if node_id not in self.completed_node_ids:
            self.completed_node_ids.append(node_id)
        
        if node_id in self.pending_node_ids:
            self.pending_node_ids.remove(node_id)
        
        self.updated_at = datetime.utcnow()
    
    def mark_pending_input(
        self, 
        node_id: str, 
        prompt: str, 
        metadata: dict | None = None
    ) -> None:
        """
        Mark checkpoint as waiting for human input.
        
        Args:
            node_id: The node waiting for input
            prompt: The prompt shown to user
            metadata: Additional context
        """
        self.status = CheckpointStatus.PENDING_INPUT
        self.pending_input_context = PendingInputContext(
            node_id=node_id,
            prompt=prompt,
            timestamp=datetime.utcnow(),
            metadata=metadata or {},
        )
        self.updated_at = datetime.utcnow()
    
    def clear_pending_input(self) -> None:
        """Clear pending input context after input received."""
        self.pending_input_context = None
        self.status = CheckpointStatus.ACTIVE
        self.updated_at = datetime.utcnow()
    
    def is_node_completed(self, node_id: str) -> bool:
        """Check if a node has completed."""
        return node_id in self.completed_node_ids
    
    def get_node_output(self, node_id: str) -> Any | None:
        """Get output of a completed node."""
        if node_id in self.node_states:
            return self.node_states[node_id].output_data
        return None
```

---

## 6. Checkpoint Configuration

```python
# dynamiq/checkpoint/models.py

class CheckpointConfig(BaseModel):
    """
    Configuration for checkpoint behavior.
    
    This config is attached to a Flow to enable checkpointing.
    
    Attributes:
        enabled: Whether checkpointing is active
        backend: Storage backend instance
        checkpoint_after_node: Create checkpoint after each node
        checkpoint_on_failure: Create checkpoint when workflow fails
        checkpoint_mid_agent_loop: Checkpoint during long agent loops
        max_checkpoints: Maximum checkpoints to keep per flow_id
        retention_days: Delete checkpoints older than this
        exclude_nodes: Node IDs to skip checkpointing
        
    Example:
        >>> from dynamiq.checkpoint.backends import FileCheckpointBackend
        >>> 
        >>> config = CheckpointConfig(
        ...     enabled=True,
        ...     backend=FileCheckpointBackend(".checkpoints"),
        ...     max_checkpoints=10,
        ... )
        >>> 
        >>> flow = Flow(nodes=[...], checkpoint_config=config)
    """
    
    enabled: bool = False
    backend: Any | None = None  # CheckpointBackend instance
    
    # Checkpoint triggers
    checkpoint_after_node: bool = True
    checkpoint_on_failure: bool = True
    checkpoint_mid_agent_loop: bool = False
    
    # Retention
    max_checkpoints: int = 10
    retention_days: int | None = None
    
    # Filtering
    exclude_nodes: list[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
```

---

## 7. Serialization Strategy

### 7.1 JSON Serialization

All models use Pydantic's built-in JSON serialization:

```python
# Serialize
checkpoint = FlowCheckpoint(flow_id="flow-1", run_id="run-1")
json_str = checkpoint.model_dump_json()

# Deserialize
checkpoint = FlowCheckpoint.model_validate_json(json_str)
```

### 7.2 Handling Complex Types

For types that aren't directly JSON-serializable:

```python
# dynamiq/checkpoint/serializers.py

import json
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

def checkpoint_json_encoder(obj: Any) -> Any:
    """
    Custom JSON encoder for checkpoint serialization.
    
    Handles:
    - datetime → ISO format string
    - UUID → string
    - Enum → value
    - Pydantic models → dict
    - Sets → lists
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize_checkpoint(checkpoint: FlowCheckpoint) -> str:
    """Serialize checkpoint to JSON string."""
    return json.dumps(
        checkpoint.model_dump(),
        default=checkpoint_json_encoder,
        ensure_ascii=False,
    )


def deserialize_checkpoint(json_str: str) -> FlowCheckpoint:
    """Deserialize checkpoint from JSON string."""
    data = json.loads(json_str)
    return FlowCheckpoint(**data)
```

### 7.3 Binary Serialization (Optional)

For high-performance scenarios:

```python
# dynamiq/checkpoint/serializers.py

try:
    import orjson
    
    def fast_serialize(checkpoint: FlowCheckpoint) -> bytes:
        """Fast binary serialization using orjson."""
        return orjson.dumps(
            checkpoint.model_dump(),
            default=checkpoint_json_encoder,
        )
    
    def fast_deserialize(data: bytes) -> FlowCheckpoint:
        """Fast binary deserialization using orjson."""
        return FlowCheckpoint(**orjson.loads(data))

except ImportError:
    # Fallback to standard json
    def fast_serialize(checkpoint: FlowCheckpoint) -> bytes:
        return serialize_checkpoint(checkpoint).encode("utf-8")
    
    def fast_deserialize(data: bytes) -> FlowCheckpoint:
        return deserialize_checkpoint(data.decode("utf-8"))
```

---

## 8. Validation

### 8.1 Checkpoint Validation

```python
# dynamiq/checkpoint/validation.py

from typing import List

def validate_checkpoint(checkpoint: FlowCheckpoint) -> List[str]:
    """
    Validate checkpoint integrity.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Check required fields
    if not checkpoint.flow_id:
        errors.append("Missing flow_id")
    if not checkpoint.run_id:
        errors.append("Missing run_id")
    
    # Check node state consistency
    for node_id in checkpoint.completed_node_ids:
        if node_id not in checkpoint.node_states:
            errors.append(f"Completed node {node_id} missing from node_states")
    
    # Check pending input consistency
    if checkpoint.status == CheckpointStatus.PENDING_INPUT:
        if not checkpoint.pending_input_context:
            errors.append("Status is PENDING_INPUT but no pending_input_context")
        elif checkpoint.pending_input_context.node_id not in checkpoint.node_states:
            # This is okay - node might not have completed yet
            pass
    
    # Version compatibility
    if checkpoint.version not in ["1.0"]:
        errors.append(f"Unknown checkpoint version: {checkpoint.version}")
    
    return errors


def validate_node_state(state: NodeCheckpointState) -> List[str]:
    """Validate node checkpoint state."""
    errors = []
    
    if not state.node_id:
        errors.append("Missing node_id")
    if not state.node_type:
        errors.append("Missing node_type")
    if not state.status:
        errors.append("Missing status")
    
    return errors
```

### 8.2 Before-Save Validation

```python
# In CheckpointBackend.save()

def save(self, checkpoint: FlowCheckpoint) -> str:
    """Save checkpoint with validation."""
    errors = validate_checkpoint(checkpoint)
    if errors:
        raise ValueError(f"Invalid checkpoint: {errors}")
    
    # ... actual save logic ...
```

---

## 9. Integration with Existing Pydantic Patterns

### 9.1 Alignment with Dynamiq's Existing Models

Looking at existing Dynamiq code:

```python
# Existing pattern in dynamiq/nodes/node.py
class Node(BaseModel):
    def to_dict(self, **kwargs) -> dict:
        return self.model_dump(**kwargs)
```

Our checkpoint models follow the same pattern:

```python
class FlowCheckpoint(BaseModel):
    def to_dict(self, **kwargs) -> dict:
        """Consistent with Node.to_dict()"""
        return self.model_dump(**kwargs)
```

### 9.2 Alignment with YAML Serialization

The existing YAML serialization (`WorkflowYAMLDumper`) uses similar patterns:

```python
# From dynamiq/serializers/dumpers/yaml.py
node_data = node.to_dict(include_secure_params=True, by_alias=True)
```

Our checkpoints can be included in YAML exports:

```python
checkpoint_data = checkpoint.model_dump(mode="json")
# Can be included in workflow YAML if needed
```

---

## 10. Module Structure

```
dynamiq/checkpoint/
├── __init__.py
├── protocol.py          # CheckpointableNode, CheckpointMixin
├── models.py            # FlowCheckpoint, NodeCheckpointState, etc.
├── serializers.py       # JSON/binary serialization
├── validation.py        # Checkpoint validation
└── backends/
    ├── __init__.py
    ├── base.py          # CheckpointBackend ABC
    ├── file.py          # FileCheckpointBackend
    ├── sqlite.py        # SQLiteCheckpointBackend
    ├── redis.py         # RedisCheckpointBackend
    └── postgres.py      # PostgresCheckpointBackend
```

---

**Previous:** [04-NODE-ANALYSIS.md](./04-NODE-ANALYSIS.md)  
**Next:** [06-STORAGE-BACKENDS.md](./06-STORAGE-BACKENDS.md)
