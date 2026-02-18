# RFC-001-08: Testing & Migration

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 8 of 9

---

## 1. Overview

This document covers:
- Testing strategy for checkpoint/resume
- Migration guide for existing users
- Backward compatibility guarantees
- Implementation timeline

---

## 2. Testing Strategy

### 2.1 Test Categories

| Category | Description | Coverage Target |
|----------|-------------|-----------------|
| **Unit Tests** | Individual components | 95%+ |
| **Integration Tests** | Component interactions | 90%+ |
| **End-to-End Tests** | Full workflow scenarios | 85%+ |
| **Performance Tests** | Throughput and latency | Baseline + regression |

### 2.2 Unit Tests

#### 2.2.1 Backend Tests

```python
# tests/unit/checkpoint/test_backends.py

import pytest
from datetime import datetime, timedelta
from dynamiq.checkpoint.backends.file import FileCheckpointBackend
from dynamiq.checkpoint.backends.sqlite import SQLiteCheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint, CheckpointStatus

# Parametrize to test all backends with same tests
@pytest.fixture(params=["file", "sqlite"])
def backend(request, tmp_path):
    if request.param == "file":
        return FileCheckpointBackend(str(tmp_path / "checkpoints"))
    elif request.param == "sqlite":
        return SQLiteCheckpointBackend(str(tmp_path / "test.db"))


@pytest.fixture
def sample_checkpoint():
    return FlowCheckpoint(
        flow_id="test-flow",
        run_id="test-run",
        status=CheckpointStatus.ACTIVE,
        original_input={"query": "test input"},
    )


class TestCheckpointBackend:
    """Test suite for all checkpoint backends."""
    
    def test_save_and_load(self, backend, sample_checkpoint):
        """Basic save and load functionality."""
        checkpoint_id = backend.save(sample_checkpoint)
        loaded = backend.load(checkpoint_id)
        
        assert loaded is not None
        assert loaded.id == sample_checkpoint.id
        assert loaded.flow_id == sample_checkpoint.flow_id
        assert loaded.original_input == sample_checkpoint.original_input
    
    def test_load_nonexistent(self, backend):
        """Loading nonexistent checkpoint returns None."""
        loaded = backend.load("nonexistent-id")
        assert loaded is None
    
    def test_update(self, backend, sample_checkpoint):
        """Update existing checkpoint."""
        backend.save(sample_checkpoint)
        
        sample_checkpoint.status = CheckpointStatus.COMPLETED
        backend.update(sample_checkpoint)
        
        loaded = backend.load(sample_checkpoint.id)
        assert loaded.status == CheckpointStatus.COMPLETED
    
    def test_delete(self, backend, sample_checkpoint):
        """Delete checkpoint."""
        backend.save(sample_checkpoint)
        
        assert backend.delete(sample_checkpoint.id) is True
        assert backend.load(sample_checkpoint.id) is None
        
        # Delete nonexistent returns False
        assert backend.delete("nonexistent") is False
    
    def test_list_by_flow(self, backend):
        """List checkpoints for a flow."""
        flow_id = "test-flow"
        
        # Create multiple checkpoints
        for i in range(5):
            cp = FlowCheckpoint(
                flow_id=flow_id,
                run_id=f"run-{i}",
                status=CheckpointStatus.ACTIVE,
            )
            backend.save(cp)
        
        checkpoints = backend.list_by_flow(flow_id, limit=3)
        
        assert len(checkpoints) == 3
        # Should be newest first
        assert checkpoints[0].run_id == "run-4"
    
    def test_list_by_flow_with_status_filter(self, backend):
        """Filter checkpoints by status."""
        flow_id = "test-flow"
        
        # Create checkpoints with different statuses
        for status in [CheckpointStatus.ACTIVE, CheckpointStatus.COMPLETED]:
            cp = FlowCheckpoint(
                flow_id=flow_id,
                run_id=f"run-{status.value}",
                status=status,
            )
            backend.save(cp)
        
        active = backend.list_by_flow(
            flow_id, 
            status=CheckpointStatus.ACTIVE
        )
        assert len(active) == 1
        assert active[0].status == CheckpointStatus.ACTIVE
    
    def test_get_latest(self, backend):
        """Get most recent checkpoint."""
        flow_id = "test-flow"
        
        for i in range(3):
            cp = FlowCheckpoint(
                flow_id=flow_id,
                run_id=f"run-{i}",
            )
            backend.save(cp)
        
        latest = backend.get_latest(flow_id)
        assert latest is not None
        assert latest.run_id == "run-2"
    
    def test_cleanup(self, backend):
        """Cleanup old checkpoints."""
        flow_id = "test-flow"
        
        # Create 10 checkpoints
        for i in range(10):
            cp = FlowCheckpoint(
                flow_id=flow_id,
                run_id=f"run-{i}",
            )
            backend.save(cp)
        
        # Keep only 3
        deleted = backend.cleanup(flow_id, keep_count=3)
        
        assert deleted == 7
        remaining = backend.list_by_flow(flow_id, limit=100)
        assert len(remaining) == 3
```

#### 2.2.2 Model Tests

```python
# tests/unit/checkpoint/test_models.py

import pytest
from datetime import datetime
from dynamiq.checkpoint.models import (
    FlowCheckpoint, 
    NodeCheckpointState, 
    CheckpointStatus,
    PendingInputContext,
)

class TestFlowCheckpoint:
    def test_create_default(self):
        """Create checkpoint with defaults."""
        cp = FlowCheckpoint(flow_id="flow-1", run_id="run-1")
        
        assert cp.id is not None
        assert cp.status == CheckpointStatus.ACTIVE
        assert cp.node_states == {}
        assert cp.completed_node_ids == []
    
    def test_mark_node_complete(self):
        """Mark a node as complete."""
        cp = FlowCheckpoint(flow_id="flow-1", run_id="run-1")
        
        state = NodeCheckpointState(
            node_id="node-1",
            node_type="Agent",
            status="success",
            output_data={"result": "done"},
        )
        
        cp.mark_node_complete("node-1", state)
        
        assert "node-1" in cp.node_states
        assert "node-1" in cp.completed_node_ids
    
    def test_mark_pending_input(self):
        """Mark checkpoint as waiting for input."""
        cp = FlowCheckpoint(flow_id="flow-1", run_id="run-1")
        
        cp.mark_pending_input(
            node_id="human-tool-1",
            prompt="Please approve this action",
            metadata={"tool": "HumanFeedback"},
        )
        
        assert cp.status == CheckpointStatus.PENDING_INPUT
        assert cp.pending_input_context is not None
        assert cp.pending_input_context.node_id == "human-tool-1"
        assert cp.pending_input_context.prompt == "Please approve this action"
    
    def test_serialization_roundtrip(self):
        """Serialize and deserialize checkpoint."""
        cp = FlowCheckpoint(
            flow_id="flow-1",
            run_id="run-1",
            original_input={"query": "test", "nested": {"value": 123}},
        )
        
        # Serialize
        json_str = cp.model_dump_json()
        
        # Deserialize
        loaded = FlowCheckpoint.model_validate_json(json_str)
        
        assert loaded.flow_id == cp.flow_id
        assert loaded.original_input == cp.original_input
```

#### 2.2.3 Node Checkpoint Tests

```python
# tests/unit/checkpoint/test_node_checkpointing.py

import pytest
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Message

class TestAgentCheckpoint:
    @pytest.fixture
    def agent(self):
        return Agent(
            id="test-agent",
            llm=OpenAI(model="gpt-4"),
        )
    
    def test_get_checkpoint_state_empty(self, agent):
        """Empty agent has minimal state."""
        state = agent.get_checkpoint_state()
        
        assert isinstance(state, dict)
        assert "prompt_messages" in state
        assert state["prompt_messages"] == []
    
    def test_get_checkpoint_state_with_history(self, agent):
        """Agent with conversation history."""
        # Simulate conversation
        agent._prompt.messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        agent._intermediate_steps = {0: {"action": "search"}}
        agent._current_loop = 2
        
        state = agent.get_checkpoint_state()
        
        assert len(state["prompt_messages"]) == 2
        assert state["intermediate_steps"] == {0: {"action": "search"}}
        assert state["current_loop"] == 2
    
    def test_restore_from_checkpoint(self, agent):
        """Restore agent from checkpoint state."""
        state = {
            "prompt_messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "intermediate_steps": {0: {"action": "search"}},
            "current_loop": 3,
            "run_depends": ["dep-1"],
        }
        
        agent.restore_from_checkpoint(state)
        
        assert len(agent._prompt.messages) == 2
        assert agent._intermediate_steps == {0: {"action": "search"}}
        assert agent._resume_from_loop == 3
        assert agent._is_resumed is True


class TestMapCheckpoint:
    def test_partial_completion_resume(self):
        """Map node resumes with only pending iterations."""
        from dynamiq.nodes.operators import Map
        
        map_node = Map(id="test-map")
        
        # Simulate partial completion
        map_node._completed_iterations = {
            0: {"result": "done-0"},
            1: {"result": "done-1"},
        }
        map_node._total_iterations = 5
        
        state = map_node.get_checkpoint_state()
        
        assert state["completed_iterations"] == {0: {"result": "done-0"}, 1: {"result": "done-1"}}
        assert state["total_iterations"] == 5
```

### 2.3 Integration Tests

```python
# tests/integration/checkpoint/test_flow_resume.py

import pytest
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.flows.flow import CheckpointConfig
from dynamiq.checkpoint.backends.sqlite import SQLiteCheckpointBackend
from dynamiq.checkpoint.models import CheckpointStatus
from dynamiq.nodes import Node
from dynamiq.runnables import RunnableStatus

class FailOnceNode(Node):
    """Node that fails once then succeeds."""
    _call_count: int = 0
    
    def execute(self, input_data, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            raise RuntimeError("Simulated failure")
        return {"result": "success"}


class CountingNode(Node):
    """Node that counts executions."""
    execution_count: int = 0
    
    def execute(self, input_data, **kwargs):
        CountingNode.execution_count += 1
        return {"count": CountingNode.execution_count}


class TestFlowResume:
    @pytest.fixture
    def backend(self, tmp_path):
        return SQLiteCheckpointBackend(str(tmp_path / "test.db"))
    
    def test_resume_after_failure(self, backend):
        """Resume flow after node failure."""
        # Create flow with failing node
        node1 = CountingNode(id="node-1")
        node2 = FailOnceNode(id="node-2")
        node2.depends = [{"node": node1}]
        
        flow = Flow(
            nodes=[node1, node2],
            checkpoint_config=CheckpointConfig(
                enabled=True,
                backend=backend,
            ),
        )
        
        # First run - should fail
        with pytest.raises(RuntimeError):
            flow.run_sync(input_data={"query": "test"})
        
        # Verify checkpoint created
        checkpoint = backend.get_latest(flow.id)
        assert checkpoint is not None
        assert checkpoint.status == CheckpointStatus.FAILED
        assert "node-1" in checkpoint.completed_node_ids
        
        # Reset counting
        CountingNode.execution_count = 0
        
        # Resume - should succeed
        result = flow.run_sync(
            input_data=None,
            resume_from=checkpoint.id
        )
        
        assert result.status == RunnableStatus.SUCCESS
        # node-1 should NOT be re-executed
        assert CountingNode.execution_count == 0
    
    def test_completed_nodes_skipped(self, backend):
        """Completed nodes are not re-executed on resume."""
        # Reset
        CountingNode.execution_count = 0
        
        node1 = CountingNode(id="node-1")
        node2 = CountingNode(id="node-2")
        node2.depends = [{"node": node1}]
        
        flow = Flow(
            nodes=[node1, node2],
            checkpoint_config=CheckpointConfig(
                enabled=True,
                backend=backend,
            ),
        )
        
        # First run
        flow.run_sync(input_data={"query": "test"})
        assert CountingNode.execution_count == 2
        
        # Get checkpoint (completed)
        checkpoint = backend.get_latest(flow.id)
        
        # Reset
        CountingNode.execution_count = 0
        
        # Resume from completed checkpoint
        flow.run_sync(input_data=None, resume_from=checkpoint.id)
        
        # No nodes should be re-executed
        assert CountingNode.execution_count == 0
    
    def test_checkpoint_list_and_cleanup(self, backend):
        """Test checkpoint listing and cleanup."""
        flow = Flow(
            nodes=[CountingNode(id="node-1")],
            checkpoint_config=CheckpointConfig(
                enabled=True,
                backend=backend,
                max_checkpoints=3,
            ),
        )
        
        # Run multiple times
        for i in range(5):
            CountingNode.execution_count = 0
            flow.run_sync(input_data={"run": i})
        
        # Should only have 3 checkpoints after cleanup
        checkpoints = flow.list_checkpoints()
        assert len(checkpoints) <= 3
```

### 2.4 End-to-End Tests

```python
# tests/e2e/checkpoint/test_hitl_resume.py

import pytest
import asyncio
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.flows.flow import CheckpointConfig
from dynamiq.checkpoint.backends.sqlite import SQLiteCheckpointBackend
from dynamiq.checkpoint.models import CheckpointStatus
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool

class TestHITLResume:
    """End-to-end tests for Human-in-the-Loop resume."""
    
    @pytest.fixture
    def backend(self, tmp_path):
        return SQLiteCheckpointBackend(str(tmp_path / "test.db"))
    
    @pytest.mark.asyncio
    async def test_hitl_checkpoint_and_resume(self, backend):
        """Test HITL workflow with checkpoint and resume."""
        # Create agent with human feedback tool
        human_tool = HumanFeedbackTool(
            id="human-feedback",
            input_method="stream",
        )
        
        agent = Agent(
            id="agent-1",
            llm=MockLLM(),  # Mock LLM for testing
            tools=[human_tool],
        )
        
        flow = Flow(
            nodes=[agent],
            checkpoint_config=CheckpointConfig(
                enabled=True,
                backend=backend,
            ),
        )
        
        # Start workflow - should pause at HITL
        with pytest.raises(HumanInputRequiredException) as exc_info:
            await flow.run_async(input_data={"query": "Review this"})
        
        # Verify checkpoint with PENDING_INPUT
        checkpoint = backend.get_latest(flow.id)
        assert checkpoint.status == CheckpointStatus.PENDING_INPUT
        assert checkpoint.pending_input_context is not None
        
        # Simulate user providing input
        human_tool._pending_response = "Approved"
        
        # Resume
        result = await flow.run_async(
            input_data=None,
            resume_from=checkpoint.id
        )
        
        # Should complete
        assert result.status == "success"
```

### 2.5 Performance Tests

```python
# tests/performance/checkpoint/test_checkpoint_perf.py

import pytest
import time
from dynamiq.checkpoint.backends.file import FileCheckpointBackend
from dynamiq.checkpoint.backends.sqlite import SQLiteCheckpointBackend
from dynamiq.checkpoint.models import FlowCheckpoint

class TestCheckpointPerformance:
    @pytest.fixture(params=["file", "sqlite"])
    def backend(self, request, tmp_path):
        if request.param == "file":
            return FileCheckpointBackend(str(tmp_path / "checkpoints"))
        else:
            return SQLiteCheckpointBackend(str(tmp_path / "test.db"))
    
    def test_save_performance(self, backend, benchmark):
        """Benchmark checkpoint save operation."""
        checkpoint = FlowCheckpoint(
            flow_id="perf-test",
            run_id="run-1",
            original_input={"data": "x" * 1000},  # 1KB input
        )
        
        result = benchmark(backend.save, checkpoint)
        
        # Should complete in under 50ms
        assert benchmark.stats.stats.mean < 0.05
    
    def test_load_performance(self, backend, benchmark):
        """Benchmark checkpoint load operation."""
        checkpoint = FlowCheckpoint(
            flow_id="perf-test",
            run_id="run-1",
        )
        checkpoint_id = backend.save(checkpoint)
        
        result = benchmark(backend.load, checkpoint_id)
        
        # Should complete in under 20ms
        assert benchmark.stats.stats.mean < 0.02
    
    def test_large_checkpoint_performance(self, backend):
        """Test with large checkpoint (1MB)."""
        # Create large checkpoint
        large_data = {"data": "x" * (1024 * 1024)}  # 1MB
        checkpoint = FlowCheckpoint(
            flow_id="perf-test",
            run_id="run-1",
            original_input=large_data,
        )
        
        start = time.time()
        backend.save(checkpoint)
        save_time = time.time() - start
        
        start = time.time()
        loaded = backend.load(checkpoint.id)
        load_time = time.time() - start
        
        # Should complete in under 500ms
        assert save_time < 0.5
        assert load_time < 0.5
```

---

## 3. Migration Guide

### 3.1 For Library Users

#### Step 1: No Changes Required (Backward Compatible)

Your existing code continues to work:

```python
# Before (still works)
flow = Flow(nodes=[agent1, agent2])
result = workflow.run(input_data={"query": "test"})
```

#### Step 2: Enable Checkpointing (Optional)

Add checkpoint configuration when ready:

```python
# After (optional enhancement)
from dynamiq.checkpoint.backends import FileCheckpointBackend

flow = Flow(
    nodes=[agent1, agent2],
    checkpoint_config=CheckpointConfig(
        enabled=True,
        backend=FileCheckpointBackend(".checkpoints"),
    ),
)
```

#### Step 3: Use Resume Functionality

Handle resume in your application:

```python
# Handle failures with resume
try:
    result = workflow.run(input_data=input_data)
except Exception as e:
    checkpoint = workflow.get_latest_checkpoint()
    if checkpoint:
        result = workflow.resume()
```

### 3.2 For Runtime Team

#### Step 1: Add Database Migration

```python
# migrations/versions/XXXXXX_add_checkpoints.py

def upgrade():
    op.create_table(
        'checkpoints',
        sa.Column('id', sa.Text(), primary_key=True),
        sa.Column('run_id', sa.Text(), sa.ForeignKey('runs.id')),
        sa.Column('flow_id', sa.Text(), nullable=False),
        sa.Column('status', sa.Text(), nullable=False),
        sa.Column('data', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True)),
    )
    
    op.create_index('idx_checkpoints_run', 'checkpoints', ['run_id'])
    op.create_index('idx_checkpoints_flow', 'checkpoints', ['flow_id'])

def downgrade():
    op.drop_table('checkpoints')
```

#### Step 2: Update execute_run.py

See [03-RUNTIME-INTEGRATION.md](./03-RUNTIME-INTEGRATION.md) for detailed changes.

#### Step 3: Add API Endpoints

```python
# New endpoints in app/api/v2/runs.py

@router.post("/runs/{run_id}/resume")
@router.get("/runs/{run_id}/checkpoints")
@router.get("/runs/{run_id}/checkpoints/{checkpoint_id}")
```

### 3.3 For Node Developers

If you're creating custom nodes with internal state:

```python
from dynamiq.nodes import Node
from dynamiq.checkpoint.protocol import CheckpointMixin

class MyStatefulNode(Node, CheckpointMixin):
    """Custom node with checkpoint support."""
    
    _internal_counter: int = 0
    _accumulated_results: list = []
    
    def execute(self, input_data, **kwargs):
        self._internal_counter += 1
        result = self._process(input_data)
        self._accumulated_results.append(result)
        return {"result": result}
    
    def get_checkpoint_state(self) -> dict:
        """Return state needed for resume."""
        return {
            "counter": self._internal_counter,
            "results": self._accumulated_results.copy(),
        }
    
    def restore_from_checkpoint(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self._internal_counter = state.get("counter", 0)
        self._accumulated_results = state.get("results", [])
        self._is_resumed = True
```

---

## 4. Backward Compatibility Guarantees

### 4.1 What Will NOT Change

| Aspect | Guarantee |
|--------|-----------|
| `Flow(nodes=...)` signature | Unchanged |
| `flow.run_sync(input_data)` | Unchanged |
| `workflow.run(input_data)` | Unchanged |
| Default behavior | No checkpointing unless enabled |
| Error handling | Same exceptions raised |
| Tracing | Continues to work |

### 4.2 What's Added (Non-Breaking)

| Addition | Description |
|----------|-------------|
| `CheckpointConfig` | New optional parameter |
| `resume_from` parameter | Optional in run methods |
| `list_checkpoints()` | New convenience method |
| `workflow.resume()` | New convenience method |

### 4.3 Deprecation Policy

No deprecations in this RFC. All changes are additive.

---

## 5. Implementation Timeline

### Phase 1: Core (Week 1)
- [ ] `dynamiq/checkpoint/` module structure
- [ ] `CheckpointConfig`, `FlowCheckpoint` models
- [ ] `CheckpointBackend` ABC
- [ ] `FileCheckpointBackend`
- [ ] Basic `Flow.run_sync` modifications
- [ ] Unit tests for models and file backend

### Phase 2: Node Implementations (Week 2)
- [ ] `CheckpointMixin` integration with `Node`
- [ ] `Agent.get_checkpoint_state()` / `restore_from_checkpoint()`
- [ ] `GraphOrchestrator` checkpoint support
- [ ] `LinearOrchestrator` checkpoint support
- [ ] `Map` checkpoint support
- [ ] Unit tests for node checkpointing

### Phase 3: Additional Backends (Week 3)
- [ ] `SQLiteCheckpointBackend`
- [ ] `RedisCheckpointBackend`
- [ ] `PostgresCheckpointBackend`
- [ ] Backend tests

### Phase 4: External Resources (Week 3-4)
- [ ] `E2BInterpreterTool` checkpoint support
- [ ] `HumanFeedbackTool` checkpoint support
- [ ] `MCPServer` checkpoint support
- [ ] Integration tests

### Phase 5: Runtime Integration (Week 4)
- [ ] Database migrations
- [ ] `execute_run.py` modifications
- [ ] New API endpoints
- [ ] End-to-end tests

### Phase 6: Documentation & Release (Week 4)
- [ ] API documentation
- [ ] Usage examples
- [ ] Migration guide
- [ ] Performance benchmarks
- [ ] Release notes

---

## 6. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Test coverage | > 90% | pytest-cov |
| Backward compatibility | 100% | Existing test suite passes |
| Save latency (SQLite) | < 10ms | Performance tests |
| Resume accuracy | 100% | Integration tests |
| HITL resume rate | 100% | E2E tests |

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Checkpoint data too large | Medium | Performance | Compress, truncate large fields |
| External resource unavailable on resume | Medium | Functional | Graceful recreation with logging |
| Backward compatibility break | Low | High | Extensive testing, gradual rollout |
| Performance regression | Low | Medium | Benchmarks, async operations |

---

## 8. Edge Cases & Production Considerations (Critical)

This section documents edge cases, potential issues, and how we address them.

### 8.1 Checkpoint Integrity & Corruption

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Process crash during save** | Partial/corrupt checkpoint | Use atomic writes (temp file + rename for File; transaction for DB) |
| **Corrupt JSON in checkpoint** | Restore fails | Validate with Pydantic before restore; fall back to previous checkpoint |
| **Disk full during save** | Save fails silently | Catch IOError, log, continue without checkpoint |
| **Checksum mismatch** | Data integrity issue | Add optional `checksum` field in FlowCheckpoint; verify on load |

```python
# Atomic save pattern for FileBackend
def save(self, checkpoint: FlowCheckpoint) -> str:
    temp_path = f"{self.base_path}/.tmp_{checkpoint.id}.json"
    final_path = f"{self.base_path}/{checkpoint.id}.json"
    
    with open(temp_path, 'w') as f:
        f.write(json.dumps(checkpoint.model_dump()))
    
    os.rename(temp_path, final_path)  # Atomic on most filesystems
    return checkpoint.id
```

### 8.2 Concurrency & Race Conditions

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Two workers resume same checkpoint** | Duplicate execution | Use atomic `claimed_at` + `claimed_by` columns in PostgreSQL |
| **Concurrent checkpoint saves** | Lost updates | Use `updated_at` optimistic locking or last-write-wins |
| **HITL input arrives during checkpoint** | Input lost | Process input queue before checkpoint; include queue state |

```python
# Atomic claim pattern for PostgreSQL
def claim_checkpoint(checkpoint_id: str, worker_id: str) -> bool:
    result = await db.execute("""
        UPDATE checkpoints 
        SET claimed_at = NOW(), claimed_by = %s
        WHERE id = %s AND claimed_at IS NULL
        RETURNING id
    """, (worker_id, checkpoint_id))
    return result.rowcount > 0  # True if we won the claim
```

### 8.3 Large Data & File Handling

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **VisionMessage with large images** | Checkpoint too big | Store images separately; save reference only |
| **Agent with 100+ message history** | Memory pressure | Truncate history beyond `max_history_messages` in checkpoint |
| **Map node with 10K iterations** | Checkpoint explosion | Compress completed iterations; use pagination |
| **Tool output with binary data** | JSON serialization fails | Base64 encode or store externally with reference |
| **Agent input files (BytesIO)** | Large checkpoint, JSON incompatible | Save metadata only; require re-upload on resume |
| **E2B sandbox files** | Files lost if sandbox expires | Save file paths; re-upload if sandbox recreated |
| **FileStore (S3/GCS/Azure)** | Duplicating external storage | DON'T checkpoint; files persist independently |
| **FileStore (InMemoryFileStore)** | ⚠️ Files LOST on crash! | Save file list as warning; recommend persistent backend |

**See 04-NODE-ANALYSIS.md Section 7b for detailed file handling strategy.**

```python
# Large history handling in Agent
def get_checkpoint_state(self) -> dict:
    messages = self._prompt.messages
    
    # Truncate if too large (keep system + last N messages)
    if len(messages) > self.checkpoint_max_messages:
        messages = [messages[0]] + messages[-self.checkpoint_max_messages:]
    
    return {
        "messages": [_serialize_message(m) for m in messages],
        "history_truncated": len(self._prompt.messages) > self.checkpoint_max_messages,
        # ...
    }
```

### 8.4 External Resource State

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **E2B sandbox expired** | Can't reconnect | Graceful fallback: create new sandbox, reinstall packages |
| **MCP server restarted** | Tool list changed | Validate discovered tools match checkpoint; warn if different |
| **Browser session expired** | Navigation state lost | Log warning; navigate to last known URL |
| **Database connection pooled** | Connection stale | Verify connections before use; reconnect if needed |

### 8.5 Time & Clock Issues

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Clock skew between pods** | Checkpoint ordering wrong | Use logical sequence numbers alongside timestamps |
| **Resume after days** | Tokens/credentials expired | Refresh connections on resume; re-authenticate if needed |
| **Timezone inconsistency** | Comparison fails | Always use UTC (`datetime.utcnow()`) |

### 8.6 Security Considerations

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Sensitive data in checkpoint** | Data exposure | Document as security consideration; encrypt at rest (v2) |
| **API keys in tool inputs** | Keys in checkpoint | Redact sensitive fields; store reference not value |
| **Cross-tenant access** | Data leak | Always filter by `flow_id` + tenant context |
| **Checkpoint ID enumeration** | Unauthorized access | Use UUIDs; validate ownership before load |

```python
# Sensitive field redaction
SENSITIVE_FIELDS = {"api_key", "password", "secret", "token"}

def _redact_sensitive(data: dict) -> dict:
    """Remove sensitive fields from checkpoint data."""
    result = {}
    for key, value in data.items():
        if any(s in key.lower() for s in SENSITIVE_FIELDS):
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = _redact_sensitive(value)
        else:
            result[key] = value
    return result
```

### 8.7 Observability & Monitoring

| Metric | Why Important | Implementation |
|--------|---------------|----------------|
| `checkpoint_save_duration_ms` | Detect slow saves | Timer around save() |
| `checkpoint_save_errors_total` | Detect backend issues | Counter on exception |
| `checkpoint_size_bytes` | Detect bloat | Log serialized size |
| `checkpoint_restore_duration_ms` | Detect slow resumes | Timer around restore() |
| `checkpoints_per_flow` | Detect retention issues | Gauge from list_by_flow count |
| `orphaned_checkpoints_total` | Detect cleanup issues | Periodic audit |

```python
# Metrics integration (example with prometheus_client)
from prometheus_client import Histogram, Counter

CHECKPOINT_SAVE_DURATION = Histogram(
    'checkpoint_save_duration_seconds',
    'Time to save a checkpoint',
    ['backend', 'status']
)

CHECKPOINT_ERRORS = Counter(
    'checkpoint_errors_total',
    'Total checkpoint errors',
    ['operation', 'error_type']
)
```

### 8.8 Memory Pressure

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Loading large checkpoint** | OOM | Stream checkpoint data; lazy load large fields |
| **Many checkpoints in memory** | Memory leak | LRU cache with max size |
| **Restoring many nodes** | Spike in memory | Restore nodes lazily on first access |

### 8.9 Network & Infrastructure

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Database unavailable** | Can't save/load | Retry with backoff; graceful degradation |
| **Redis cluster failover** | Temporary unavailability | Use circuit breaker; fall back to local |
| **Network partition** | Split-brain | Use consistent database as source of truth |
| **High latency** | Slow saves blocking execution | Always async save; don't block workflow |

### 8.10 Version Compatibility

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Library version mismatch** | Deserialization fails | Store `dynamiq_version` in checkpoint |
| **Schema migration** | Old checkpoints unreadable | Version field + migration functions |
| **Node class changed** | State incompatible | Graceful fallback; re-execute node if can't restore |

```python
# Version-aware restore
def restore_from_checkpoint(self, state: dict) -> None:
    schema_version = state.get("_schema_version", 1)
    
    if schema_version < CURRENT_SCHEMA_VERSION:
        state = self._migrate_state(state, schema_version)
    
    # Now restore
    self._prompt.messages = state.get("messages", [])
    ...
```

### 8.11 Streaming Integration

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Checkpoint during token streaming** | Partial token in state | Only checkpoint at complete boundaries |
| **Resume with stale SSE connection** | Client doesn't see events | Client reconnect with checkpoint_id; replay from checkpoint |
| **Multiple clients streaming** | Duplicate events | Use event sequence numbers; client dedup |

### 8.12 Tracing Continuity

| Scenario | Risk | Mitigation |
|----------|------|------------|
| **Resume creates new trace** | Trace fragmentation | Link traces: `resumed_from_trace_id` |
| **Original trace expired** | Can't correlate | Store trace context in checkpoint |
| **Different trace backend** | Correlation broken | Use standard W3C trace context |

```python
# Tracing continuity
class FlowCheckpoint:
    # ...
    trace_context: dict | None = None  # W3C trace context
    original_trace_id: str | None = None
    
# On resume
if checkpoint.original_trace_id:
    span.set_attribute("resumed_from_trace", checkpoint.original_trace_id)
```

---

## 9. Open Questions & Future Work

### 9.1 Deferred to v2

| Feature | Reason | Priority |
|---------|--------|----------|
| **Checkpoint encryption** | Security team review needed | High |
| **Binary serialization (msgpack)** | Performance optimization | Medium |
| **Cross-cluster resume** | Requires distributed state | Low |
| **Checkpoint export/import** | User-facing feature | Medium |

### 9.2 Requires Further Design

1. **Distributed locking strategy** - PostgreSQL advisory locks vs Redis SETNX?
   - *Recommendation:* Start with PostgreSQL `SELECT FOR UPDATE`

2. **Checkpoint TTL management** - How long to keep checkpoints?
   - *Recommendation:* Configurable per-flow, default 7 days

3. **Nested workflow checkpoints** - Workflow A calls Workflow B?
   - *Recommendation:* Each workflow has independent checkpoint; link via parent_checkpoint_id

4. **Partial Map resumption** - Resume specific iterations only?
   - *Recommendation:* Implement in v1 with `failed_iterations` list

---

## 10. Production Readiness Checklist

Before production deployment, verify:

| Category | Item | Status |
|----------|------|--------|
| **Testing** | All unit tests pass | ☐ |
| | All integration tests pass | ☐ |
| | Load test with 100 concurrent workflows | ☐ |
| | Chaos test (kill worker mid-execution) | ☐ |
| **Security** | Security team review | ☐ |
| | Sensitive data handling reviewed | ☐ |
| | Access control verified | ☐ |
| **Operations** | Monitoring dashboards created | ☐ |
| | Alerting rules configured | ☐ |
| | Runbook for checkpoint issues | ☐ |
| **Documentation** | User documentation complete | ☐ |
| | API documentation complete | ☐ |
| | Troubleshooting guide | ☐ |

---

## 11. Conclusion

This RFC provides a complete checkpoint/resume implementation that:

1. ✅ **Is backward compatible** - No changes required for existing users
2. ✅ **Handles all node types** - From simple to complex (Agent, Orchestrators, Map)
3. ✅ **Integrates with runtime** - Works with existing HITL, streaming, WebSocket
4. ✅ **Provides multiple backends** - File → SQLite → Redis → PostgreSQL
5. ✅ **Is well-tested** - Comprehensive test strategy
6. ✅ **Has clear migration path** - Step-by-step guide for all stakeholders

**Recommended Next Steps:**

1. Review this RFC at Architecture Review Board
2. Approve implementation plan
3. Begin Phase 1 implementation
4. Weekly progress reviews

---

**Previous:** [07-FLOW-INTEGRATION.md](./07-FLOW-INTEGRATION.md)  
**Next:** [09-UI-CHAT-INTEGRATION.md](./09-UI-CHAT-INTEGRATION.md)
