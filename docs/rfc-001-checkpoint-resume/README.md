# RFC-001: Checkpoint/Resume for Dynamiq Workflows

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Target Review:** Architecture Review Board

---

## Summary

This RFC proposes adding checkpoint/resume capabilities to Dynamiq workflows, enabling:

- **Fault tolerance** - Resume from failures without re-executing completed nodes
- **Human-in-the-Loop persistence** - HITL workflows survive browser closes, device switches
- **Long-running workflow support** - Complex multi-agent tasks can be interrupted and resumed

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Opt-in via `CheckpointConfig`** | Backward compatible, no changes for existing users |
| **Checkpoint at node boundaries** | Matches DAG execution model (like LangGraph) |
| **Protocol-based node support** | Each node defines its own checkpoint logic |
| **Pluggable backends** | File → SQLite → Redis → PostgreSQL |
| **Integrates with existing HITL** | Complements runtime's database polling mechanism |

## Document Structure

| Part | Document | Description |
|------|----------|-------------|
| 0 | [**Review Checklist**](./00-REVIEW-CHECKLIST.md) | **Executive briefing for Architecture Board** |
| 1 | [Executive Summary](./01-EXECUTIVE-SUMMARY.md) | Problem, solution, stakeholders |
| 2 | [Industry Research](./02-INDUSTRY-RESEARCH.md) | **12 frameworks**: LangGraph, Temporal, AutoGen, Prefect, Haystack, Bedrock, etc. |
| 3 | [Runtime Integration](./03-RUNTIME-INTEGRATION.md) | How checkpoints work with HITL, streaming, WebSockets |
| 4 | [Node Analysis](./04-NODE-ANALYSIS.md) | State requirements for each node type |
| 5 | [Data Models](./05-DATA-MODELS.md) | Pydantic models, protocols, serialization |
| 6 | [Storage Backends](./06-STORAGE-BACKENDS.md) | File, SQLite, Redis, PostgreSQL implementations |
| 7 | [Flow Integration](./07-FLOW-INTEGRATION.md) | Modifications to Flow.run_sync |
| 8 | [Testing & Migration](./08-TESTING-MIGRATION.md) | Test strategy, backward compatibility, timeline |
| 9 | [UI & Chat Integration](./09-UI-CHAT-INTEGRATION.md) | Frontend, streaming, **alternative approaches** |

## Key Insight: Checkpoints vs Memory

For **90% of chat use cases**, no new endpoints are needed:
- **Memory** handles conversation continuity across runs
- **HITL** during a run uses existing `run_input_events` mechanism
- **Checkpoints** provide safety net for crashes and long pauses

See [Section 10 of UI & Chat Integration](./09-UI-CHAT-INTEGRATION.md#10-alternative-continue-via-new-message-no-new-endpoints) for the alternative "Continue via New Message" pattern.

## Quick Start Example

```python
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.flows.flow import CheckpointConfig
from dynamiq.checkpoint.backends import FileCheckpointBackend

# Enable checkpointing
flow = Flow(
    nodes=[agent1, agent2],
    checkpoint_config=CheckpointConfig(
        enabled=True,
        backend=FileCheckpointBackend(".checkpoints"),
    ),
)

workflow = Workflow(flow=flow)

# Run with automatic checkpointing
result = workflow.run(input_data={"query": "Research AI trends"})

# Resume from failure
checkpoint = workflow.get_latest_checkpoint()
result = workflow.resume(checkpoint_id=checkpoint.id)
```

## Implementation Timeline

| Phase | Week | Deliverables |
|-------|------|--------------|
| Core | 1 | Models, file backend, basic flow integration |
| Nodes | 2 | Agent, orchestrators, Map checkpoint support |
| Backends | 3 | SQLite, Redis, PostgreSQL |
| Runtime | 4 | API endpoints, database migrations |
| Release | 4 | Documentation, tests, benchmarks |

## Backward Compatibility

✅ **100% backward compatible** - Existing code works unchanged  
✅ **Opt-in only** - Checkpointing disabled by default  
✅ **No breaking changes** - All new parameters are optional

## Review Checklist

- [ ] Architecture Review Board approval
- [ ] Security review (checkpoint data handling)
- [ ] Performance benchmarks acceptable
- [ ] Runtime team sign-off on integration approach
- [ ] Documentation complete

---

*For questions or feedback, contact the AI Architecture Team.*
