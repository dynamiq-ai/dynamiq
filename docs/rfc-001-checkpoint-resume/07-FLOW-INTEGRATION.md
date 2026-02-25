# RFC-001-07: Flow Integration

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 7 of 9

---

## 1. Overview

This document details the modifications to `Flow.run_sync()` and related methods to support checkpoint/resume functionality.

**Goal:** Minimal, surgical changes that preserve existing behavior while adding opt-in checkpointing.

---

## 2. Flow Class Modifications

### 2.1 New Attributes

```python
# dynamiq/flows/flow.py

from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointConfig, CheckpointStatus
from dynamiq.checkpoint.protocol import CheckpointableNode

class Flow(BaseFlow):
    """
    Flow with checkpoint/resume support.
    
    Checkpointing is opt-in via checkpoint_config. When enabled:
    - Checkpoints are created after each node completes
    - Workflows can be resumed from any checkpoint
    - HITL workflows persist across sessions
    
    Example:
        >>> from dynamiq.checkpoint.backends import FileCheckpointBackend
        >>> 
        >>> flow = Flow(
        ...     nodes=[agent1, agent2],
        ...     checkpoint_config=CheckpointConfig(
        ...         enabled=True,
        ...         backend=FileCheckpointBackend(".checkpoints"),
        ...     ),
        ... )
        >>> 
        >>> # Run with automatic checkpointing
        >>> result = flow.run_sync(input_data={"query": "..."})
        >>> 
        >>> # Resume from checkpoint if needed
        >>> checkpoint = flow.get_latest_checkpoint()
        >>> result = flow.run_sync(input_data=None, resume_from=checkpoint.id)
    """
    
    # Checkpoint configuration (opt-in)
    checkpoint_config: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Configuration for checkpoint/resume functionality"
    )
    
    # Private attributes for checkpoint state
    _checkpoint: FlowCheckpoint | None = PrivateAttr(default=None)
    _original_input: Any = PrivateAttr(default=None)
```

### 2.2 Modified run_sync()

```python
def run_sync(
    self,
    input_data: Any,
    config: RunnableConfig = None,
    *,
    resume_from: str | FlowCheckpoint | None = None,
    **kwargs
) -> RunnableResult:
    """
    Run flow with optional checkpoint resume.
    
    Args:
        input_data: Input data for the flow. If resuming, this can be None
                    to use the checkpoint's original_input.
        config: Runnable configuration
        resume_from: Checkpoint ID or FlowCheckpoint to resume from.
                     If provided, the flow resumes from that checkpoint.
        **kwargs: Additional arguments passed to node execution
        
    Returns:
        RunnableResult with flow output
        
    Raises:
        ValueError: If resume_from is provided but checkpoint not found
        
    Example:
        >>> # Normal run
        >>> result = flow.run_sync(input_data={"query": "test"})
        >>> 
        >>> # Resume from checkpoint
        >>> result = flow.run_sync(
        ...     input_data=None,  # Uses checkpoint's original_input
        ...     resume_from="checkpoint-123"
        ... )
    """
    
    # ========== CHECKPOINT RESUME HANDLING ==========
    if resume_from:
        self._checkpoint = self._load_checkpoint(resume_from)
        if not self._checkpoint:
            raise ValueError(f"Checkpoint not found: {resume_from}")
        
        # Restore workflow state from checkpoint
        self._restore_from_checkpoint(self._checkpoint)
        
        # Use checkpoint's original input if none provided
        if input_data is None:
            input_data = self._checkpoint.original_input
        
        # Add resume metadata for tracing
        kwargs["resumed_from_checkpoint"] = self._checkpoint.id
        kwargs["original_run_id"] = self._checkpoint.run_id
        
        logger.info(
            f"Resuming from checkpoint {self._checkpoint.id}, "
            f"skipping {len(self._checkpoint.completed_node_ids)} completed nodes"
        )
    else:
        # Fresh run - reset state
        self.reset_run_state()
        self._checkpoint = None
    
    # Store original input for checkpoint
    self._original_input = input_data
    
    # ========== INITIALIZE CHECKPOINT ==========
    run_id = uuid4()
    
    if self._should_checkpoint() and not self._checkpoint:
        self._checkpoint = FlowCheckpoint(
            flow_id=self.id,
            run_id=str(run_id),
            original_input=input_data,
            original_config=config.model_dump() if config else None,
        )
        
        # Save initial checkpoint
        self._save_checkpoint()
    
    # ========== MAIN EXECUTION LOOP ==========
    try:
        while self._ts.is_active():
            ready_nodes = self._get_nodes_ready_to_run(input_data=input_data)
            
            # Skip already-completed nodes on resume
            if self._checkpoint:
                ready_nodes = [
                    n for n in ready_nodes 
                    if n.id not in self._checkpoint.completed_node_ids
                ]
            
            if not ready_nodes:
                time.sleep(0.003)
                continue
            
            # Execute ready nodes
            results = self._run_executor.execute(
                ready_nodes, input_data, config, **kwargs
            )
            
            self._results.update(results)
            self._ts.done(*results.keys())
            
            # ========== CHECKPOINT AFTER BATCH ==========
            if self._should_checkpoint():
                self._update_checkpoint(results, CheckpointStatus.ACTIVE)
            
            time.sleep(0.003)
        
        # ========== SUCCESS ==========
        if self._should_checkpoint():
            self._update_checkpoint({}, CheckpointStatus.COMPLETED)
            self._cleanup_old_checkpoints()
        
        return self._build_result(RunnableStatus.SUCCESS)
        
    except HumanInputRequiredException as e:
        # ========== HITL PAUSE ==========
        if self._should_checkpoint():
            self._checkpoint.mark_pending_input(
                node_id=e.node_id,
                prompt=e.prompt,
                metadata=e.metadata,
            )
            self._save_checkpoint()
            logger.info(
                f"Checkpoint saved with PENDING_INPUT status, "
                f"checkpoint_id={self._checkpoint.id}"
            )
        raise
        
    except Exception as e:
        # ========== FAILURE ==========
        if self._should_checkpoint_on_failure():
            self._update_checkpoint({}, CheckpointStatus.FAILED)
            logger.info(
                f"Checkpoint saved on failure, "
                f"checkpoint_id={self._checkpoint.id}"
            )
        raise
```

### 2.3 Checkpoint Helper Methods

```python
def _load_checkpoint(
    self, 
    resume_from: str | FlowCheckpoint
) -> FlowCheckpoint | None:
    """
    Load checkpoint from ID or return if already a checkpoint.
    
    Args:
        resume_from: Checkpoint ID string or FlowCheckpoint instance
        
    Returns:
        The checkpoint if found, None otherwise
    """
    if isinstance(resume_from, FlowCheckpoint):
        return resume_from
    
    if self.checkpoint_config.backend:
        return self.checkpoint_config.backend.load(resume_from)
    
    return None


def _restore_from_checkpoint(self, checkpoint: FlowCheckpoint) -> None:
    """
    Restore flow state from checkpoint.
    
    This method:
    1. Restores completed node results
    2. Restores internal state of complex nodes
    3. Reinitializes the topological sorter with completed nodes marked
    
    Args:
        checkpoint: The checkpoint to restore from
    """
    # Reset results dict
    self._results = {}
    
    for node_id, node_state in checkpoint.node_states.items():
        # Restore completed results to skip re-execution
        if node_state.status in ("success", "failure", "skip"):
            self._results[node_id] = RunnableResult(
                status=RunnableStatus(node_state.status),
                input=node_state.input_data,
                output=node_state.output_data,
                error=(
                    RunnableResultError(**node_state.error) 
                    if node_state.error else None
                ),
            )
        
        # Restore internal node state (for complex nodes)
        node = self._node_by_id.get(node_id)
        if (node 
            and isinstance(node, CheckpointableNode) 
            and node_state.internal_state):
            node.restore_from_checkpoint(node_state.internal_state)
            logger.debug(f"Restored internal state for node {node_id}")
    
    # Reinitialize topological sorter
    self._ts = self.init_node_topological_sorter(nodes=self.nodes)
    
    # Mark completed nodes as done
    for node_id in checkpoint.completed_node_ids:
        if node_id in [n.id for n in self.nodes]:
            try:
                self._ts.done(node_id)
            except ValueError:
                # Node might not be in active set
                pass
    
    logger.info(
        f"Restored from checkpoint: "
        f"{len(checkpoint.completed_node_ids)} nodes completed, "
        f"{len(checkpoint.pending_node_ids)} nodes pending"
    )


def _should_checkpoint(self) -> bool:
    """Check if checkpointing is enabled and configured."""
    return (
        self.checkpoint_config.enabled 
        and self.checkpoint_config.backend is not None
        and self.checkpoint_config.checkpoint_after_node
    )


def _should_checkpoint_on_failure(self) -> bool:
    """Check if should checkpoint on failure."""
    return (
        self.checkpoint_config.enabled 
        and self.checkpoint_config.backend is not None
        and self.checkpoint_config.checkpoint_on_failure
    )


def _update_checkpoint(
    self, 
    new_results: dict[str, RunnableResult],
    status: CheckpointStatus
) -> None:
    """
    Update checkpoint with new results.
    
    Args:
        new_results: Results from nodes that just completed
        status: New checkpoint status
    """
    if not self._checkpoint:
        return
    
    for node_id, result in new_results.items():
        node = self._node_by_id.get(node_id)
        if not node:
            continue
        
        # Skip excluded nodes
        if node_id in self.checkpoint_config.exclude_nodes:
            continue
        
        # Get internal state if node supports checkpointing
        internal_state = {}
        if isinstance(node, CheckpointableNode):
            internal_state = node.get_checkpoint_state()
        
        # Create node state
        node_state = NodeCheckpointState(
            node_id=node_id,
            node_type=type(node).__name__,
            status=result.status.value,
            input_data=result.input,
            output_data=result.output,
            error=result.error.to_dict() if result.error else None,
            internal_state=internal_state,
            completed_at=datetime.utcnow(),
        )
        
        self._checkpoint.mark_node_complete(node_id, node_state)
    
    self._checkpoint.status = status
    self._save_checkpoint()


def _save_checkpoint(self) -> None:
    """Save current checkpoint to backend."""
    if self._checkpoint and self.checkpoint_config.backend:
        try:
            self.checkpoint_config.backend.save(self._checkpoint)
            logger.debug(f"Checkpoint saved: {self._checkpoint.id}")
        except Exception as e:
            # Log but don't fail execution
            logger.warning(f"Failed to save checkpoint: {e}")


def _cleanup_old_checkpoints(self) -> None:
    """Remove old checkpoints beyond max_checkpoints."""
    if not self.checkpoint_config.backend:
        return
    
    try:
        deleted = self.checkpoint_config.backend.cleanup(
            self.id,
            keep_count=self.checkpoint_config.max_checkpoints,
            older_than_days=self.checkpoint_config.retention_days,
        )
        if deleted > 0:
            logger.debug(f"Cleaned up {deleted} old checkpoints")
    except Exception as e:
        logger.warning(f"Failed to cleanup checkpoints: {e}")
```

### 2.4 Convenience Methods

```python
# Public API for checkpoint management

def list_checkpoints(self, limit: int = 10) -> list[FlowCheckpoint]:
    """
    List checkpoints for this flow.
    
    Args:
        limit: Maximum number of checkpoints to return
        
    Returns:
        List of checkpoints, newest first
        
    Example:
        >>> checkpoints = flow.list_checkpoints(limit=5)
        >>> for cp in checkpoints:
        ...     print(f"{cp.id}: {cp.status}")
    """
    if not self.checkpoint_config.backend:
        return []
    return self.checkpoint_config.backend.list_by_flow(self.id, limit=limit)


def get_latest_checkpoint(
    self, 
    status: CheckpointStatus | None = None
) -> FlowCheckpoint | None:
    """
    Get the most recent checkpoint.
    
    Args:
        status: Optional status filter
        
    Returns:
        The latest checkpoint if found
        
    Example:
        >>> # Get latest successful checkpoint
        >>> cp = flow.get_latest_checkpoint(status=CheckpointStatus.COMPLETED)
    """
    if not self.checkpoint_config.backend:
        return None
    return self.checkpoint_config.backend.get_latest(self.id, status=status)


def delete_checkpoint(self, checkpoint_id: str) -> bool:
    """
    Delete a specific checkpoint.
    
    Args:
        checkpoint_id: The checkpoint to delete
        
    Returns:
        True if deleted, False if not found
    """
    if not self.checkpoint_config.backend:
        return False
    return self.checkpoint_config.backend.delete(checkpoint_id)


def clear_all_checkpoints(self) -> int:
    """
    Delete all checkpoints for this flow.
    
    Returns:
        Number of checkpoints deleted
    """
    if not self.checkpoint_config.backend:
        return 0
    return self.checkpoint_config.backend.cleanup(self.id, keep_count=0)
```

---

## 3. Workflow Class Modifications

```python
# dynamiq/workflow/workflow.py

class Workflow(BaseModel):
    """
    Workflow with checkpoint-aware run method.
    """
    
    def run(
        self,
        input_data: Any,
        config: RunnableConfig = None,
        *,
        resume_from: str | FlowCheckpoint | None = None,
        **kwargs
    ) -> RunnableResult:
        """
        Run workflow with optional checkpoint resume.
        
        Delegates to flow.run_sync() with checkpoint support.
        
        Args:
            input_data: Input data (None if resuming)
            config: Runnable configuration
            resume_from: Checkpoint to resume from
            **kwargs: Additional arguments
            
        Returns:
            RunnableResult with workflow output
        """
        return self.flow.run_sync(
            input_data=input_data,
            config=config,
            resume_from=resume_from,
            **kwargs
        )
    
    async def run_async(
        self,
        input_data: Any,
        config: RunnableConfig = None,
        *,
        resume_from: str | FlowCheckpoint | None = None,
        **kwargs
    ) -> RunnableResult:
        """
        Async run with checkpoint support.
        """
        return await self.flow.run_async(
            input_data=input_data,
            config=config,
            resume_from=resume_from,
            **kwargs
        )
    
    # Checkpoint convenience methods
    def list_checkpoints(self, limit: int = 10) -> list[FlowCheckpoint]:
        """List checkpoints for this workflow's flow."""
        return self.flow.list_checkpoints(limit=limit)
    
    def get_latest_checkpoint(
        self, 
        status: CheckpointStatus | None = None
    ) -> FlowCheckpoint | None:
        """Get most recent checkpoint."""
        return self.flow.get_latest_checkpoint(status=status)
    
    def resume(
        self,
        checkpoint_id: str | None = None,
        input_data: Any = None,
        **kwargs
    ) -> RunnableResult:
        """
        Resume from the latest or specified checkpoint.
        
        Convenience method that handles checkpoint lookup.
        
        Args:
            checkpoint_id: Specific checkpoint to resume from.
                           If None, uses the latest checkpoint.
            input_data: Override input data (optional)
            **kwargs: Additional arguments
            
        Returns:
            RunnableResult
            
        Raises:
            ValueError: If no checkpoint found
        """
        if checkpoint_id:
            resume_from = checkpoint_id
        else:
            checkpoint = self.get_latest_checkpoint()
            if not checkpoint:
                raise ValueError("No checkpoint found to resume from")
            resume_from = checkpoint.id
        
        return self.run(
            input_data=input_data,
            resume_from=resume_from,
            **kwargs
        )
```

---

## 4. Integration with Existing Features

### 4.1 Tracing Integration

```python
# In Flow.run_sync, when resuming:

if resume_from:
    # Preserve tracing continuity
    kwargs["resumed_from_checkpoint"] = checkpoint.id
    kwargs["original_run_id"] = checkpoint.run_id
    
    # Start new trace but reference original
    if config and hasattr(config, 'callbacks'):
        for callback in config.callbacks:
            if hasattr(callback, 'on_resume'):
                callback.on_resume(
                    checkpoint_id=checkpoint.id,
                    original_run_id=checkpoint.run_id,
                )
```

### 4.2 Error Handling Integration

```python
# In Flow.run_sync, existing error handling works with checkpoints:

try:
    # ... execution ...
except Exception as e:
    # Checkpoint preserves state for retry
    if self._should_checkpoint_on_failure():
        self._update_checkpoint({}, CheckpointStatus.FAILED)
    
    # Existing error handling continues
    if self.error_handling.behavior == Behavior.RAISE:
        raise
    # ...
```

### 4.3 Parallel Execution

```python
# Checkpointing works with parallel execution:
# Each node result is captured individually

results = self._run_executor.execute(
    ready_nodes,  # May be multiple nodes running in parallel
    input_data, 
    config, 
    **kwargs
)

# All completed nodes are checkpointed together
self._update_checkpoint(results, CheckpointStatus.ACTIVE)
```

---

## 5. Usage Examples

### 5.1 Basic Checkpointing

```python
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.flows.flow import CheckpointConfig
from dynamiq.checkpoint.backends import FileCheckpointBackend
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI

# Create backend
backend = FileCheckpointBackend(".checkpoints")

# Create flow with checkpointing
flow = Flow(
    nodes=[
        Agent(id="agent-1", llm=OpenAI(model="gpt-4")),
    ],
    checkpoint_config=CheckpointConfig(
        enabled=True,
        backend=backend,
    ),
)

workflow = Workflow(flow=flow)

# Run - checkpoints created automatically
result = workflow.run(input_data={"query": "Research AI trends"})
```

### 5.2 Resume After Failure

```python
try:
    result = workflow.run(input_data={"query": "Complex task"})
except Exception as e:
    print(f"Workflow failed: {e}")
    
    # Get checkpoint
    checkpoint = workflow.get_latest_checkpoint()
    if checkpoint:
        print(f"Resuming from checkpoint {checkpoint.id}")
        print(f"Completed nodes: {checkpoint.completed_node_ids}")
        
        # Resume
        result = workflow.resume()
```

### 5.3 Human-in-the-Loop

```python
from dynamiq.checkpoint.exceptions import HumanInputRequiredException

try:
    result = workflow.run(input_data={"query": "Review this"})
except HumanInputRequiredException as e:
    # Workflow paused for human input
    checkpoint = workflow.get_latest_checkpoint()
    print(f"Waiting for input: {e.prompt}")
    print(f"Checkpoint: {checkpoint.id}")
    
    # Store checkpoint ID for later
    save_checkpoint_id_for_user(checkpoint.id)

# Later, when user provides input...
checkpoint_id = get_saved_checkpoint_id()
result = workflow.resume(checkpoint_id=checkpoint_id)
```

### 5.4 Production with PostgreSQL

```python
from dynamiq.checkpoint.backends import PostgresCheckpointBackend

backend = PostgresCheckpointBackend(
    connection_string=os.environ["DATABASE_URL"],
)

flow = Flow(
    nodes=[...],
    checkpoint_config=CheckpointConfig(
        enabled=True,
        backend=backend,
        max_checkpoints=20,
        retention_days=30,
    ),
)
```

---

## 6. Backward Compatibility

### 6.1 Default Behavior

```python
# Existing code works unchanged:
flow = Flow(nodes=[...])  # checkpoint_config.enabled=False by default
workflow = Workflow(flow=flow)
result = workflow.run(input_data={"query": "test"})
# No checkpoints created, no behavior change
```

### 6.2 API Compatibility

| API | Before | After | Breaking? |
|-----|--------|-------|-----------|
| `Flow(nodes=...)` | Works | Works | No |
| `flow.run_sync(input_data)` | Works | Works | No |
| `workflow.run(input_data)` | Works | Works | No |
| `flow.run_sync(resume_from=...)` | N/A | New | No |
| `workflow.resume()` | N/A | New | No |

### 6.3 No Required Changes

Existing users:
- Don't need to change any code
- Don't need to add checkpoint configuration
- Don't need to handle new exceptions (HITL exception is opt-in)

---

**Previous:** [06-STORAGE-BACKENDS.md](./06-STORAGE-BACKENDS.md)  
**Next:** [08-TESTING-MIGRATION.md](./08-TESTING-MIGRATION.md)
