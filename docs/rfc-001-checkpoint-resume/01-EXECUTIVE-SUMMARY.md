# RFC-001-01: Executive Summary

**Status:** Final Draft v7.0  
**Created:** January 6, 2026  
**Part:** 1 of 9

---

## 1. Problem Statement

Dynamiq workflows are currently **stateless**. When a workflow execution fails, is interrupted, or requires human input across sessions, there is no mechanism to resume from where it left off. This creates significant problems:

| Problem | Impact |
|---------|--------|
| **Wasted computation** | Re-executing completed nodes on failure wastes expensive LLM API calls |
| **Poor UX for HITL** | Human-in-the-loop workflows cannot persist across browser sessions or device switches |
| **No fault tolerance** | Transient failures (network, rate limits) require full restart |
| **Long-running workflows fail** | Complex multi-agent tasks spanning hours have no recovery mechanism |

## 2. Proposed Solution

Add an **opt-in checkpoint/resume capability** that:

1. **Persists workflow state** after each node execution to a configurable backend
2. **Supports resumption** from any checkpoint (time travel capability)
3. **Integrates with existing runtime** - Works seamlessly with WebSocket, SSE, and HTTP streaming
4. **Handles complex nodes** - Agent loops, Map parallelism, orchestrators, HITL
5. **Maintains backward compatibility** - All existing code works unchanged
6. **Provides pluggable storage** - File → SQLite → Redis → PostgreSQL

## 3. Key Stakeholders

| Role | Concern |
|------|---------|
| **Library Users** | Simple API, minimal code changes to enable checkpointing |
| **Runtime Team** | Integration with existing streaming/HITL infrastructure |
| **DevOps** | Production backends, monitoring, cleanup strategies |
| **Product** | Reliable human-in-the-loop experiences |

## 4. Design Principles (Informed by Industry Research)

After analyzing LangGraph, CrewAI, Google ADK, Metaflow, and Letta, we adopt these principles:

| Principle | Source | Application |
|-----------|--------|-------------|
| **Checkpoint at boundaries** | LangGraph | Checkpoint after each node completes |
| **Thread isolation** | LangGraph, ADK | Use `run_id` + `flow_id` for isolation |
| **Pending input state** | CrewAI | Special `PENDING_INPUT` status for HITL |
| **Clone-based resume** | Metaflow | Skip completed nodes, only re-execute pending |
| **Protocol-based** | All frameworks | Each node defines its own checkpoint logic |
| **ID remapping** | Letta | Clean serialization with referential integrity |

## 5. Scope

### In Scope
- Checkpoint creation after node execution
- Resume from any checkpoint
- All node types (Agent, Orchestrators, Map, HITL tools, etc.)
- File, SQLite, Redis, PostgreSQL backends
- Integration with existing runtime HITL mechanism
- Pydantic-based serialization

### Out of Scope (Future Work)
- Real-time checkpoint streaming to external systems
- Multi-region checkpoint replication
- Checkpoint-based workflow debugging UI
- Automatic retry with exponential backoff (handled by runtime)

## 6. Success Criteria

| Metric | Target |
|--------|--------|
| **Backward compatibility** | 100% - existing code unchanged |
| **Resume accuracy** | Completed nodes never re-executed |
| **Checkpoint overhead** | < 100ms per checkpoint (file backend) |
| **HITL reliability** | Resume after browser close, device switch |
| **Test coverage** | > 90% for checkpoint code |

## 7. Document Structure

This RFC is organized into 9 sub-documents:

| Part | Document | Description |
|------|----------|-------------|
| 1 | Executive Summary | This document |
| 2 | Industry Research | Deep analysis of LangGraph, CrewAI, ADK, Metaflow, Letta |
| 3 | Runtime Integration | How checkpoints work with HITL, streaming, WebSockets |
| 4 | Node Analysis | State requirements for each node type |
| 5 | Data Models | Pydantic models, protocols, serialization |
| 6 | Storage Backends | File, SQLite, Redis, PostgreSQL implementations |
| 7 | Flow Integration | Modifications to Flow.run_sync |
| 8 | Testing & Migration | Test strategy, backward compatibility, migration guide |
| 9 | UI & Chat Integration | Frontend, streaming events, multi-device scenarios |

---

**Next:** [02-INDUSTRY-RESEARCH.md](./02-INDUSTRY-RESEARCH.md)
