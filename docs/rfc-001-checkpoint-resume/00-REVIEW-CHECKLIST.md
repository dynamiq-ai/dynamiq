# RFC-001: Pre-Review Checklist & Executive Briefing

**For:** Architecture Review Board  
**Date:** January 6, 2026  
**Status:** Ready for Review

---

## 1. Document Completeness Checklist

| Section | Document | Status | Key Content |
|---------|----------|--------|-------------|
| ✅ | README | Complete | Index, quick start |
| ✅ | 01-Executive Summary | Complete | Problem, solution, stakeholders |
| ✅ | 02-Industry Research | **COMPREHENSIVE** | 12 frameworks analyzed (LangGraph, Letta, ADK, Temporal, AutoGen, Prefect, Haystack, Bedrock, etc.) |
| ✅ | 03-Runtime Integration | Complete | HITL, WebSocket, SSE, database polling |
| ✅ | 04-Node Analysis | Complete | All node types categorized (A-D complexity) |
| ✅ | 05-Data Models | Complete | Pydantic models, protocols |
| ✅ | 06-Storage Backends | Complete | 4 backends with full code |
| ✅ | 07-Flow Integration | Complete | `Flow.run_sync()` modifications |
| ✅ | 08-Testing & Migration | Complete | Test strategy, timeline, backward compat |
| ✅ | 09-UI & Chat Integration | Complete | Frontend, streaming, multi-device |

---

## 2. Anticipated Review Questions & Answers

### Q1: Why not just use LangGraph's checkpointing directly?
**A:** LangGraph's checkpoint system is tightly coupled to their channel-based state model. Dynamiq uses a DAG-based execution model with different node types. We've extracted the **best practices** (checkpoint at boundaries, thread isolation, pending input state) while adapting to our architecture.

**Evidence from deep dive (see 02-INDUSTRY-RESEARCH.md):**
- LangGraph's `Checkpoint` TypedDict stores `channel_values` and `versions_seen` - not applicable to our node-based model
- Their `interrupt()` function pattern **IS** directly applicable and we adopt it
- Their `JsonPlusSerializer` handles Pydantic, dataclasses, numpy - we can use similar patterns
- Their PostgreSQL backend with migrations is our reference implementation

### Q1b: What about Manus AI? They seem to have the best agents.
**A:** We investigated Manus AI thoroughly:
- **OpenManus repository:** Empty placeholder (only README.md) - no code available
- **Source code:** Proprietary/closed-source, not accessible
- **What we learned:** From TiDB case study and observed behavior, we inferred their approach:
  - Frequent checkpoints with distributed database (TiDB)
  - Event-based state updates
  - Client-agnostic checkpoint IDs (multi-device support)
- **What we adopted:** Philosophy of "checkpoint often", multi-device design, graceful degradation
- **See:** Section 5 of 02-INDUSTRY-RESEARCH.md for full analysis

### Q2: What's the performance impact?
**A:** 
- **Checkpoint save:** 5-20ms (PostgreSQL) 
- **Checkpoint load:** 5-20ms
- **Non-blocking:** Saves are async, don't block streaming
- **Opt-in:** Zero overhead if disabled (default)

### Q3: How does this affect existing deployments?
**A:** 
- **100% backward compatible** - No code changes required
- **Opt-in** - `CheckpointConfig(enabled=True)` to activate
- **Graceful degradation** - Checkpoint failures don't crash workflows

### Q4: What about data security? Checkpoints contain sensitive data.
**A:**
- Checkpoints stored in same database/storage as run data
- Same access controls apply
- **Future:** Encryption at rest (marked as future work in RFC)
- **Recommendation:** Review with Security team before production

### Q5: How do we handle checkpoint storage growth?
**A:**
- **`max_checkpoints`** config limits per-flow (default: 10)
- **`retention_days`** for automatic cleanup
- **Cleanup on completion** - Old checkpoints pruned after success

### Q6: How do we compare to the competition?
**A:** See Section 10 of 02-INDUSTRY-RESEARCH.md for detailed comparison. Summary:

| Area | Dynamiq Advantage |
|------|-------------------|
| **Agent granularity** | Mid-loop checkpoint (loop 5 of 15). LangGraph only checkpoints at node boundaries. |
| **Map parallelism** | Iteration-level tracking. Metaflow requires full restart. |
| **HITL integration** | Leverage existing runtime infrastructure. Others need new setup. |
| **Simplicity** | DAG model vs LangGraph's complex channels. |

**Unique Capabilities:**
1. Mid-agent loop checkpointing (no competitor has this)
2. Map iteration-level resume
3. Nested HITL support (Agent → Tool → HumanFeedback)
4. YAML + Checkpoint synergy

**Known Gaps:**
- No ormsgpack binary optimization (JSON is portable)
- No distributed storage native (PostgreSQL scales adequately)

---

## 3. Key Design Decisions Requiring Approval

### Decision 1: Checkpoint at Node Boundaries (Not Token Level)
**Rationale:** Token-level would be too expensive (1000+ checkpoints per response). Node boundaries match DAG model and industry standard (LangGraph).

### Decision 2: Opt-in via `CheckpointConfig`
**Rationale:** Backward compatibility is paramount. No existing code should break.

### Decision 3: Protocol-based Node Support
**Rationale:** Each node knows its own state better than a generic solution. Nodes implement `get_checkpoint_state()` and `restore_from_checkpoint()`.

### Decision 4: PostgreSQL as Primary Production Backend
**Rationale:** Already in runtime stack, ACID guarantees, JSONB for flexible schema.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Checkpoint data too large | Medium | Performance | Compress, truncate large fields, set limits |
| External resources unavailable | Medium | Functional | Graceful recreation with logging |
| Backward compatibility break | Low | High | Extensive testing, existing tests must pass |
| Performance regression | Low | Medium | Benchmarks, async operations |
| Security concerns | Medium | High | Security review before production |
| Concurrent resume attempts | Medium | Data corruption | Atomic claim pattern with DB transactions |
| Checkpoint corruption | Low | High | Atomic writes, validation on load |
| Clock skew across pods | Low | Ordering issues | Use logical sequence + UTC timestamps |

---

## 4b. Edge Cases We've Explicitly Addressed

**See 08-TESTING-MIGRATION.md Section 8 for full details.** Summary:

| Category | # Cases Documented | Key Mitigations |
|----------|-------------------|-----------------|
| **Integrity/Corruption** | 4 | Atomic writes, checksums, Pydantic validation |
| **Concurrency** | 3 | DB transactions, optimistic locking |
| **Large Data** | 4 | Truncation, compression, external storage |
| **External Resources** | 4 | Graceful reconnect, fallback creation |
| **Security** | 4 | Redaction, tenant isolation, ownership checks |
| **Observability** | 6 | Prometheus metrics, structured logging |
| **Network/Infra** | 4 | Retry, circuit breaker, async saves |
| **Version Compat** | 3 | Schema versioning, migration functions |

**Key Message:** We've thought through 32+ production edge cases and documented mitigations.

---

## 5. Implementation Effort Estimate

| Phase | Duration | Effort | Dependencies |
|-------|----------|--------|--------------|
| **Phase 1: Core** | 1 week | 1 engineer | None |
| **Phase 2: Nodes** | 1 week | 1 engineer | Phase 1 |
| **Phase 3: Backends** | 1 week | 1 engineer | Phase 1 |
| **Phase 4: Runtime** | 1 week | 1 engineer + Runtime team | Phase 1-3 |
| **Phase 5: Testing** | 1 week | 1 engineer | Phase 1-4 |

**Total:** ~4-5 weeks with 1 engineer, parallelizable to 3 weeks with 2 engineers.

---

## 6. What We Explicitly Did NOT Include (Out of Scope)

| Feature | Reason | Future Work? |
|---------|--------|--------------|
| Multi-region replication | Complexity, low priority | Yes (v2) |
| Checkpoint debugging UI | UX effort, not core | Yes (v2) |
| Automatic retry | Already handled by runtime | No |
| Real-time sync to external systems | Scope creep | Maybe (v3) |
| Checkpoint encryption | Needs security review | Yes (v1.1) |

---

## 7. Success Metrics (Post-Implementation)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backward compat | 100% | All existing tests pass |
| Resume accuracy | 100% | Completed nodes never re-execute |
| HITL resume success | > 99% | Production monitoring |
| Checkpoint overhead | < 100ms | Performance benchmarks |
| Adoption | 50% of HITL workflows | Usage analytics |

---

## 8. Approval Required From

- [ ] **Architecture Review Board** - Design approval
- [ ] **Runtime Team** - Integration approach
- [ ] **Security Team** - Data handling review
- [ ] **DevOps** - Production backend requirements

---

## 9. Next Steps After Approval

1. Create `dynamiq/checkpoint/` module structure
2. Implement core models and file backend
3. Add checkpoint support to Agent node (most critical)
4. Implement PostgreSQL backend
5. Integrate with runtime `execute_run.py`
6. Write tests
7. Documentation and examples

---

*This document serves as the executive briefing for the Architecture Review Board.*
