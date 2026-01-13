# Performance Review

Analyze the code for performance issues specific to the Dynamiq AI orchestration framework.

## What to Analyze

Review the code against these performance areas:

### 1. DAG Execution

- Sequential bottlenecks where parallelizable nodes are chained unnecessarily
- Deep `node.depends_on()` chains limiting parallelism
- Nodes with `is_postponed_component_init=True` that could be pre-initialized
- Excessive `reset_run_state()` calls recreating TopologicalSorter

### 2. Executor Configuration

- Default `MAX_WORKERS_THREAD_POOL_EXECUTOR = 8` may be suboptimal for I/O workloads
- ProcessExecutor serialization overhead with `jsonpickle`
- Missing `executor.shutdown(wait=True)` causing resource leaks
- Thread pool exhaustion from nested agent tool execution

### 3. Connection Management

- Nodes creating clients in `execute()` instead of `init_components()`
- Missing `_connection_manager` storage preventing reconnection
- Connection objects recreated per-request instead of reused
- Missing `ensure_client()` override for custom clients

### 4. Caching

- Nodes with `caching.enabled = False` that have deterministic outputs
- Large output data causing cache serialization overhead
- Missing cache invalidation for nodes with mutable external state

### 5. LLM and Agent Performance

- Agent `max_loops` too high causing unnecessary iterations
- Missing `stop_sequences` causing over-generation
- `parallel_tool_calls_enabled=True` creating ThreadPoolExecutor per loop
- Token counting overhead with `prompt.count_tokens()` on every loop

### 6. Async Patterns

- `asyncio.to_thread()` wrapping already async-capable operations
- Blocking `time.sleep()` instead of `asyncio.sleep()` in async paths
- `requests.*` in async context instead of httpx/aiohttp
- File I/O without `aiofiles` in async node execution

### 7. Memory

- Node `clone()` creating deep copies of large objects unnecessarily
- `_intermediate_steps` growing unboundedly in long agent runs
- Prompt message history accumulation without `summarization_config`

### 8. Callbacks and Tracing

- `TracingCallbackHandler` serializing large node state on every event
- `to_dict(for_tracing=True)` called multiple times per node execution
- Streaming callbacks processing every chunk individually

### 9. Serialization

- `WorkflowYAMLDumper.dump()` serializing entire workflow graph
- Nested node serialization creating deep recursion
- `jsonpickle.encode()` overhead for ProcessExecutor

### 10. Vector Store Operations

- Missing batch operations for document indexing
- Embedder called per-document instead of batch embedding
- Retriever `top_k` too high returning unused results
- Vector store client reconnection on every query

## Output Format

Provide findings as:

```markdown
## Performance Analysis Report

### Critical Issues
- [ ] Issue description with file:line reference

### Optimization Opportunities
- [ ] Improvement that could yield measurable gains

### Metrics to Monitor
- Specific measurements to track improvement
```

## Key Files to Check

- `dynamiq/flows/flow.py` - Core execution engine
- `dynamiq/nodes/agents/agent.py` - Agent loop performance
- `dynamiq/executors/pool.py` - Thread/process pool management
- `dynamiq/connections/managers.py` - Connection lifecycle
- `dynamiq/cache/managers/workflow.py` - Caching implementation
