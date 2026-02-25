# RFC-001-03: Runtime Integration

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 3 of 9

---

## 1. Overview

This document explains how checkpoint/resume integrates with the existing Dynamiq runtime infrastructure, specifically:

- WebSocket streaming
- Server-Sent Events (SSE)
- Human-in-the-Loop (HITL) via database polling
- Multi-device/multi-pod support

**Critical Insight:** The runtime already has robust HITL infrastructure. Checkpointing **complements** this - it doesn't replace it.

---

## 2. Current Runtime Architecture

### 2.1 High-Level Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Runtime   │────▶│   Worker    │
│ (WebSocket/ │     │  (FastAPI)  │     │ (execute_   │
│    SSE)     │◀────│             │◀────│   run.py)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   ▼                   │
       │            ┌─────────────┐            │
       │            │  PostgreSQL │            │
       │            │  - runs     │            │
       └───────────▶│  - stream_  │◀───────────┘
                    │    chunks   │
                    │  - run_     │
                    │    input_   │
                    │    events   │
                    └─────────────┘
```

### 2.2 Key Database Tables

| Table | Purpose |
|-------|---------|
| `runs` | Run metadata, status, input/output |
| `stream_chunks` | Streaming events (approval, messages, etc.) |
| `run_input_events` | HITL input from any device |
| `threads` | Groups related runs |

### 2.3 HITL Data Flow (Current)

```
1. Agent tool emits approval event
   ↓
2. StreamPersister saves to stream_chunks
   ↓
3. Client receives via SSE/WebSocket
   ↓
4. User provides input
   ↓
5. Client sends input via WS message or REST API
   ↓
6. Runtime saves to run_input_events
   ↓
7. hitl_input_pump polls run_input_events
   ↓
8. Input forwarded to workflow's input_queue
   ↓
9. Tool receives input, continues execution
```

---

## 3. How Checkpointing Integrates

### 3.1 Checkpoint + HITL: Not Conflicting, Complementary

**Key Realization:** Checkpoints and HITL serve different purposes:

| Aspect | HITL (Existing) | Checkpointing (New) |
|--------|-----------------|---------------------|
| **Purpose** | Collect human input | Persist execution state |
| **Trigger** | Tool needs approval | Node completes |
| **Storage** | `run_input_events` | Checkpoint backend |
| **Resume** | Input pump forwards to queue | Restore node states |

**They work together:**
1. Checkpoint captures "workflow is waiting for input at tool X"
2. If runtime crashes, we restore from checkpoint
3. Resume re-emits the approval event via streaming
4. User provides input through existing HITL mechanism
5. Workflow continues

### 3.2 Integration Points

```python
# In execute_run.py (conceptual changes)

async def execute_run(...):
    # NEW: Check for checkpoint resume
    checkpoint = None
    if resume_from_checkpoint_id:
        checkpoint = await load_checkpoint(resume_from_checkpoint_id)
        workflow = restore_workflow_from_checkpoint(workflow, checkpoint)
    
    # EXISTING: Set up HITL
    hitl_sync_queue = Queue()
    hitl_stop_event = asyncio.Event()
    hitl_bridge_task = asyncio.create_task(
        hitl_input_pump(run_id, hitl_sync_queue, hitl_stop_event)
    )
    
    # EXISTING: Execute workflow
    result = await workflow.run_async(input_data=input_data, config=run_config)
    
    # NEW: Save final checkpoint on completion
    if checkpoint_config.enabled:
        await save_checkpoint(workflow, status=CheckpointStatus.COMPLETED)
```

### 3.3 When HITL Node Resumes

```
Resume from Checkpoint
       ↓
Workflow restored with pending_input_context
       ↓
Tool marked with _is_resumed = True
       ↓
Tool re-emits approval event via streaming
       ↓
Client sees approval event (same as first time)
       ↓
User provides input (same flow as before)
       ↓
Workflow continues
```

---

## 4. Detailed HITL + Checkpoint Scenarios

### 4.1 Scenario: Browser Closed, User Returns Later

**Without Checkpointing (Current):**
1. User starts workflow with HITL
2. Workflow reaches approval point
3. Approval event sent to client
4. **User closes browser**
5. WebSocket disconnects
6. Workflow times out waiting for input
7. **Run fails** ❌

**With Checkpointing (Proposed):**
1. User starts workflow with HITL
2. Workflow reaches approval point
3. Checkpoint saved with `status=PENDING_INPUT`
4. Approval event sent to client
5. **User closes browser**
6. WebSocket disconnects
7. Workflow times out (configurable behavior)
8. Run marked as `PENDING_INPUT` (not failed)
9. **Later:** User returns, opens workflow
10. Client calls resume endpoint with checkpoint_id
11. Workflow restored from checkpoint
12. Tool re-emits approval event
13. User provides input
14. Workflow completes ✅

### 4.2 Scenario: Switch Devices Mid-Workflow

**Current:** Already supported! Input via `run_input_events` table.

**With Checkpointing:** Same flow, but if worker pod dies during the wait:
1. New worker picks up run
2. Restores from checkpoint
3. Continues waiting for input (or re-emits approval)

### 4.3 Scenario: Worker Pod Crashes

**Without Checkpointing:**
1. Run marked as FAILED
2. User must restart from beginning

**With Checkpointing:**
1. Checkpoint exists with last successful state
2. Reaper/retry mechanism picks up run
3. Restores from checkpoint
4. Only re-executes nodes after the checkpoint

---

## 5. Runtime API Changes

### 5.1 New Endpoints

```python
# In app/api/v2/runs.py

@router.post("/runs/{run_id}/resume")
async def resume_run(
    run_id: UUID,
    checkpoint_id: str | None = None,  # If None, use latest
    input_data: dict | None = None,    # Override input if needed
) -> RunResponse:
    """
    Resume a run from a checkpoint.
    
    If checkpoint_id is not provided, uses the latest checkpoint for this run.
    If the run is in PENDING_INPUT state, this also resumes from that point.
    """
    ...

@router.get("/runs/{run_id}/checkpoints")
async def list_checkpoints(
    run_id: UUID,
    limit: int = 10,
) -> list[CheckpointSummary]:
    """
    List available checkpoints for a run.
    
    Returns checkpoint metadata without full state (for performance).
    """
    ...

@router.get("/runs/{run_id}/checkpoints/{checkpoint_id}")
async def get_checkpoint(
    run_id: UUID,
    checkpoint_id: str,
) -> CheckpointDetail:
    """
    Get detailed checkpoint information including node states.
    """
    ...
```

### 5.2 Run Status Changes

```python
class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_INPUT = "pending_input"  # NEW: Waiting for human input
```

### 5.3 Database Schema Additions

```sql
-- New table for checkpoints
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    flow_id TEXT NOT NULL,
    status TEXT NOT NULL,  -- active, completed, failed, pending_input
    
    -- Checkpoint data (JSONB for flexibility)
    node_states JSONB NOT NULL DEFAULT '{}',
    completed_node_ids TEXT[] NOT NULL DEFAULT '{}',
    original_input JSONB,
    
    -- HITL context
    pending_input_context JSONB,  -- node_id, prompt, timestamp
    
    -- Metadata
    version TEXT NOT NULL DEFAULT '1.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Indexes
    CONSTRAINT checkpoints_run_id_idx UNIQUE (run_id, id)
);

CREATE INDEX checkpoints_run_flow_idx ON checkpoints(run_id, flow_id);
CREATE INDEX checkpoints_status_idx ON checkpoints(run_id, status);
```

---

## 6. Streaming Event Changes

### 6.1 New Event Types

```python
# Checkpoint-related streaming events

class CheckpointCreatedEvent(BaseStreamingEvent):
    """Emitted when a checkpoint is created."""
    event: str = "checkpoint_created"
    data: dict  # {"checkpoint_id": "...", "status": "active", "node_id": "..."}

class CheckpointResumedEvent(BaseStreamingEvent):
    """Emitted when resuming from a checkpoint."""
    event: str = "checkpoint_resumed"
    data: dict  # {"checkpoint_id": "...", "skipped_nodes": [...]}
```

### 6.2 Enhanced Approval Event

```python
# Existing approval event with checkpoint context

class ApprovalEvent(BaseStreamingEvent):
    event: str = "approval"
    data: dict  # {
                #   "entity_id": "tool-123",
                #   "prompt": "Approve this action?",
                #   "checkpoint_id": "...",  # NEW: For resume reference
                #   "is_resumed": False,     # NEW: True if re-emitted after resume
                # }
```

---

## 7. Worker Changes

### 7.1 execute_run.py Modifications

```python
# Key changes to execute_run.py

async def execute_run(
    run_id: UUID,
    attempt_id: UUID,
    session: AsyncSession,
    nexus_client: Any,
    resume_from_checkpoint: str | None = None,  # NEW
) -> None:
    """Execute a run with checkpoint support."""
    
    run = await session.get(Run, run_id)
    
    # NEW: Handle checkpoint resume
    if resume_from_checkpoint:
        checkpoint = await load_checkpoint(session, resume_from_checkpoint)
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {resume_from_checkpoint}")
        
        # Log resume event
        logger.info(f"Resuming run {run_id} from checkpoint {checkpoint.id}")
        
        # Emit resume event to stream
        if stream_to_db:
            await _emit_checkpoint_event(
                session, run_id, attempt_id, "checkpoint_resumed",
                {"checkpoint_id": checkpoint.id, "skipped_nodes": checkpoint.completed_node_ids}
            )
    
    # Initialize workflow
    workflow = await init_workflow_by_uri(run.workflow_uri, tracing_handler)
    
    # NEW: Restore workflow state from checkpoint
    if resume_from_checkpoint and checkpoint:
        restore_workflow_from_checkpoint(workflow, checkpoint)
        input_data = checkpoint.original_input  # Use original input
    else:
        input_data = run.input
    
    # NEW: Configure checkpointing
    if checkpoint_config_enabled:
        workflow.flow.checkpoint_config = CheckpointConfig(
            enabled=True,
            backend=PostgresCheckpointBackend(session),
            checkpoint_after_node=True,
        )
    
    # ... rest of execution (mostly unchanged)
    
    # NEW: On HITL timeout, save pending_input checkpoint instead of failing
    try:
        result = await asyncio.wait_for(
            workflow.run_async(input_data=input_data, config=run_config),
            timeout=hitl_timeout_seconds,
        )
    except asyncio.TimeoutError:
        if checkpoint_config_enabled:
            # Save checkpoint with PENDING_INPUT status
            checkpoint = await save_checkpoint(
                workflow, 
                status=CheckpointStatus.PENDING_INPUT,
                pending_input_context=get_pending_input_context(workflow),
            )
            run.status = RunStatus.PENDING_INPUT.value
            run.metadata = {"checkpoint_id": checkpoint.id}
            await session.commit()
            return  # Don't fail, just pause
        raise
```

### 7.2 Checkpoint Helper Functions

```python
async def load_checkpoint(
    session: AsyncSession, 
    checkpoint_id: str
) -> FlowCheckpoint | None:
    """Load checkpoint from database."""
    result = await session.execute(
        select(CheckpointModel).where(CheckpointModel.id == checkpoint_id)
    )
    row = result.scalar_one_or_none()
    if row:
        return FlowCheckpoint(**row.to_dict())
    return None


def restore_workflow_from_checkpoint(
    workflow: Workflow, 
    checkpoint: FlowCheckpoint
) -> None:
    """Restore workflow state from checkpoint."""
    flow = workflow.flow
    
    # Restore node states
    for node_id, node_state in checkpoint.node_states.items():
        node = flow._node_by_id.get(node_id)
        if not node:
            continue
        
        # Mark completed nodes
        if node_state.status in ("success", "skip"):
            flow._results[node_id] = RunnableResult(
                status=RunnableStatus(node_state.status),
                input=node_state.input_data,
                output=node_state.output_data,
            )
        
        # Restore internal state
        if hasattr(node, 'restore_from_checkpoint') and node_state.internal_state:
            node.restore_from_checkpoint(node_state.internal_state)
    
    # Mark nodes as done in topological sorter
    for node_id in checkpoint.completed_node_ids:
        if node_id in flow._ts._active:
            flow._ts.done(node_id)


def get_pending_input_context(workflow: Workflow) -> dict | None:
    """Extract pending input context from workflow if HITL is waiting."""
    # Find the node that's waiting for input
    for node in workflow.flow.nodes:
        if hasattr(node, '_pending_prompt') and node._pending_prompt:
            return {
                "node_id": node.id,
                "prompt": node._pending_prompt,
                "timestamp": datetime.utcnow().isoformat(),
            }
    return None
```

---

## 8. Client Integration

### 8.1 JavaScript/TypeScript Example

```typescript
// Client-side handling of checkpoint events

const eventSource = new EventSource(`/runs/${runId}/stream`);

eventSource.addEventListener('checkpoint_created', (event) => {
  const data = JSON.parse(event.data);
  console.log(`Checkpoint created: ${data.checkpoint_id}`);
  // Store for potential resume
  localStorage.setItem(`checkpoint_${runId}`, data.checkpoint_id);
});

eventSource.addEventListener('approval', async (event) => {
  const data = JSON.parse(event.data);
  
  if (data.is_resumed) {
    console.log('Approval re-emitted after resume');
  }
  
  // Show approval dialog to user
  const approved = await showApprovalDialog(data.prompt);
  
  // Send response
  await fetch(`/runs/${runId}/input`, {
    method: 'POST',
    body: JSON.stringify({ content: approved ? 'APPROVE' : 'REJECT' }),
  });
});

// Resume from checkpoint if user returns
async function checkAndResume(runId: string) {
  const response = await fetch(`/runs/${runId}`);
  const run = await response.json();
  
  if (run.status === 'pending_input') {
    // Resume the run
    await fetch(`/runs/${runId}/resume`, { method: 'POST' });
    // Reconnect to stream
    connectToStream(runId);
  }
}
```

### 8.2 Python SDK Example

```python
from dynamiq_runtime import RuntimeClient

client = RuntimeClient(base_url="http://localhost:8000")

# Start a run
run = client.create_run(workflow_uri="workflows/agent.yaml", input={"query": "..."})

# If run is paused for input
if run.status == "pending_input":
    # Get checkpoint info
    checkpoints = client.list_checkpoints(run.id)
    latest = checkpoints[0]
    
    print(f"Workflow waiting for input at: {latest.pending_input_context['prompt']}")
    
    # Provide input and resume
    client.send_input(run.id, content="APPROVE")
    
    # Or resume from checkpoint explicitly
    resumed_run = client.resume_run(run.id, checkpoint_id=latest.id)
```

---

## 9. Failure Scenarios and Recovery

### 9.1 Worker Pod Crash During Execution

```
1. Worker crashes mid-execution
2. Run stays in RUNNING status (stale)
3. Reaper detects stale run (no heartbeat)
4. Reaper checks for checkpoint
   - If exists: Mark run as RESUMABLE
   - If not: Mark run as FAILED
5. New worker picks up RESUMABLE run
6. Resumes from checkpoint
7. Continues execution
```

### 9.2 Database Connection Lost

```
1. Checkpoint save fails
2. Workflow continues (checkpoint is optional)
3. Next checkpoint attempt retries
4. If persistent failure: Log warning, continue without checkpointing
```

### 9.3 Checkpoint Backend Full

```
1. Save fails with storage error
2. Cleanup old checkpoints (beyond max_checkpoints)
3. Retry save
4. If still failing: Disable checkpointing, log error
```

---

## 10. Performance Considerations

### 10.1 Checkpoint Overhead

| Operation | Expected Time | Mitigation |
|-----------|---------------|------------|
| Serialize node state | 1-10ms | Use msgpack/orjson |
| Write to PostgreSQL | 5-20ms | Batch writes, async |
| Read checkpoint | 5-20ms | Index on run_id |
| Restore workflow | 1-5ms | In-memory operation |

### 10.2 Recommendations

1. **Don't checkpoint every LLM token** - Only checkpoint at node boundaries
2. **Use async saves** - Don't block workflow execution
3. **Compress large states** - For agents with long conversation history
4. **Set reasonable retention** - Default 10 checkpoints per flow

---

## 11. Summary

**Key Takeaways:**

1. **HITL and checkpointing are complementary** - Checkpoints capture state, HITL handles input
2. **Minimal runtime changes** - Most logic is in the library, runtime just orchestrates
3. **Backward compatible** - Existing runs work unchanged
4. **Graceful degradation** - Checkpoint failures don't crash workflows
5. **Multi-device support preserved** - Database-based input still works

---

**Previous:** [02-INDUSTRY-RESEARCH.md](./02-INDUSTRY-RESEARCH.md)  
**Next:** [04-NODE-ANALYSIS.md](./04-NODE-ANALYSIS.md)
