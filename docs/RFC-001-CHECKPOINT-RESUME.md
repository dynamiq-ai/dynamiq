# RFC-001: Checkpoint/Resume for Dynamiq Workflows

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Author:** AI Architecture Team  
**Target Review:** Architecture Review Board

---

## Overview

This RFC proposes adding checkpoint/resume capabilities to Dynamiq workflows. The full RFC is broken into 8 sub-documents for detailed coverage.

**📁 Full RFC Location:** [`docs/rfc-001-checkpoint-resume/`](./rfc-001-checkpoint-resume/)

---

## Problem Statement

Dynamiq workflows are currently **stateless**. When a workflow:
- Fails mid-execution
- Requires human input across sessions
- Is interrupted by infrastructure issues

...there is no way to resume from where it left off.

**Impact:**
- Wasted computation (re-executing completed LLM calls)
- Poor UX for human-in-the-loop workflows
- No fault tolerance for long-running agent tasks

---

## Proposed Solution

Add an **opt-in checkpoint/resume capability** that:

1. **Persists workflow state** after each node execution
2. **Supports resumption** from any checkpoint
3. **Integrates with existing runtime** (WebSocket, SSE, HITL)
4. **Handles complex nodes** (Agent loops, Map parallelism, orchestrators)
5. **Maintains 100% backward compatibility**

---

## Key Design Decisions

| Decision | Source | Rationale |
|----------|--------|-----------|
| Checkpoint at node boundaries | LangGraph | Matches DAG execution model |
| `PENDING_INPUT` status for HITL | CrewAI | Explicit handling of human feedback |
| Clone-based resume | Metaflow | Skip completed nodes efficiently |
| Protocol-based node support | All frameworks | Each node defines its checkpoint logic |
| Pydantic models | Dynamiq pattern | Type safety, consistent serialization |

---

## Document Structure

| Part | Document | Description |
|------|----------|-------------|
| **0** | [**Review Checklist**](./rfc-001-checkpoint-resume/00-REVIEW-CHECKLIST.md) | **Executive briefing for Architecture Board** |
| 1 | [Executive Summary](./rfc-001-checkpoint-resume/01-EXECUTIVE-SUMMARY.md) | Problem, solution, stakeholders, success criteria |
| 2 | [Industry Research](./rfc-001-checkpoint-resume/02-INDUSTRY-RESEARCH.md) | **12 frameworks analyzed**: LangGraph, Temporal, AutoGen, Prefect, Haystack, Bedrock, etc. |
| 3 | [Runtime Integration](./rfc-001-checkpoint-resume/03-RUNTIME-INTEGRATION.md) | How checkpoints work with HITL, streaming, WebSockets |
| 4 | [Node Analysis](./rfc-001-checkpoint-resume/04-NODE-ANALYSIS.md) | State requirements for every node type |
| 5 | [Data Models](./rfc-001-checkpoint-resume/05-DATA-MODELS.md) | Pydantic models, protocols, serialization |
| 6 | [Storage Backends](./rfc-001-checkpoint-resume/06-STORAGE-BACKENDS.md) | File, SQLite, Redis, PostgreSQL implementations |
| 7 | [Flow Integration](./rfc-001-checkpoint-resume/07-FLOW-INTEGRATION.md) | Modifications to `Flow.run_sync()` |
| 8 | [Testing & Migration](./rfc-001-checkpoint-resume/08-TESTING-MIGRATION.md) | Test strategy, backward compatibility, timeline |
| 9 | [UI & Chat Integration](./rfc-001-checkpoint-resume/09-UI-CHAT-INTEGRATION.md) | Frontend, streaming, **alternative approaches** |

---

## Alternative: Continue via New Message (Cursor Pattern)

**Important:** For **90% of chat use cases**, explicit checkpoints aren't needed:

| Scenario | Solution | New Endpoint? |
|----------|----------|---------------|
| Normal follow-up message | Memory (existing) | ❌ No |
| HITL input (workflow running) | `run_input_events` (existing) | ❌ No |
| HITL resume (workflow paused) | Extend existing `POST /threads/{id}/runs` | ❌ No |
| Crash recovery | Checkpoint resume | ✅ Optional |

**The Cursor/ChatGPT Pattern:** When you send a new message in a chat, the agent continues naturally because Memory provides context from previous runs. Checkpoints are only needed for edge cases like crash recovery and long pauses.

See [Section 10 of UI & Chat Integration](./rfc-001-checkpoint-resume/09-UI-CHAT-INTEGRATION.md) for full details on avoiding new endpoints.

---

## Quick Start Example

```python
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.flows.flow import CheckpointConfig
from dynamiq.checkpoint.backends import FileCheckpointBackend

# Enable checkpointing (opt-in)
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

# Resume from failure if needed
checkpoint = workflow.get_latest_checkpoint()
result = workflow.resume(checkpoint_id=checkpoint.id)
```

---

## HITL + Checkpointing: How They Work Together

**Key Insight:** Checkpointing and HITL are **complementary**, not conflicting.

```
┌─────────────────────────────────────────────────────────────┐
│                    Current HITL Flow                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Agent tool emits approval event                          │
│ 2. Event persisted to stream_chunks                         │
│ 3. Client receives via WebSocket/SSE                        │
│ 4. User provides input                                      │
│ 5. Input saved to run_input_events                          │
│ 6. hitl_input_pump forwards to workflow                     │
│ 7. Workflow continues                                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              With Checkpointing (Enhancement)               │
├─────────────────────────────────────────────────────────────┤
│ • Checkpoint captures "waiting for input at tool X"         │
│ • If runtime crashes, restore from checkpoint               │
│ • Re-emit approval event via streaming                      │
│ • User provides input through SAME mechanism                │
│ • Workflow continues                                        │
└─────────────────────────────────────────────────────────────┘
```

**Scenario: User closes browser, returns later**
- Without checkpointing: Workflow times out, fails ❌
- With checkpointing: Resume from checkpoint, re-emit prompt ✅

---

## Implementation Timeline

| Phase | Week | Deliverables |
|-------|------|--------------|
| **Core** | 1 | Models, file backend, basic Flow integration |
| **Nodes** | 2 | Agent, Orchestrators, Map checkpoint support |
| **Backends** | 3 | SQLite, Redis, PostgreSQL implementations |
| **Runtime** | 4 | API endpoints, database migrations |
| **Release** | 4 | Documentation, tests, benchmarks |

---

## Backward Compatibility

| Guarantee | Status |
|-----------|--------|
| Existing code works unchanged | ✅ |
| No required parameters added | ✅ |
| No breaking API changes | ✅ |
| Checkpointing is opt-in | ✅ |

---

## Industry Research Summary

We analyzed **12 frameworks** across AI agents, workflow orchestration, and enterprise platforms:

### Primary Analysis (Code Review)
| Framework | Checkpoint Model | HITL Support | Key Learning |
|-----------|------------------|--------------|--------------|
| **LangGraph** | Channel-based, super-steps | `interrupt()` function | Thread isolation via `thread_id` |
| **CrewAI** | Decorator-based | `HumanFeedbackPending` exception | Separate pending_feedback table |
| **Google ADK** | Session events | Event append | Multi-tenant isolation |
| **Metaflow** | Clone-based resume | N/A | Selective re-execution |
| **Letta** | Agent serialization | N/A | ID remapping for clean export |
| **Manus AI** | Context engineering | N/A | TiDB for high write throughput |

### Extended Analysis (Documentation Review)
| Framework | Checkpoint Model | Key Learning |
|-----------|------------------|--------------|
| **Temporal** | Event-sourced replay | Industry gold standard for durable execution |
| **Microsoft AutoGen** | Superstep boundaries | Automatic checkpointing, state isolation |
| **Prefect** | `persist_result=True` | Per-task opt-in (validates our approach) |
| **Haystack** | Breakpoints + snapshots | **Very similar to our design!** |
| **Amazon Bedrock** | Session Management APIs | KMS encryption, enterprise patterns |
| **Semantic Kernel** | Stateful steps | Process framework checkpointing |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Backward compatibility | 100% |
| Resume accuracy | Completed nodes never re-executed |
| Checkpoint overhead | < 100ms (file backend) |
| HITL resume reliability | 100% |
| Test coverage | > 90% |

---

## Review Checklist

- [ ] Architecture Review Board approval
- [ ] Security review (checkpoint data handling)
- [ ] Performance benchmarks acceptable
- [ ] Runtime team sign-off
- [ ] Documentation complete

---

## Next Steps

1. **Review** the detailed sub-documents in [`docs/rfc-001-checkpoint-resume/`](./rfc-001-checkpoint-resume/)
2. **Provide feedback** on specific sections
3. **Approve** for implementation
4. **Begin Phase 1** (Core infrastructure)

---

*Document version: 5.0 | Last updated: January 6, 2026*
