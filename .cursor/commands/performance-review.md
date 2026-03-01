Analyze the current code for performance issues specific to the Dynamiq framework.

Focus on these areas:

1. **DAG Execution**: Look for sequential bottlenecks where nodes could run in parallel. Check for unnecessary `node.depends_on()` chains and excessive `reset_run_state()` calls.

2. **Executor Configuration**: Review thread pool settings. Check if `ProcessExecutor` is used where `ThreadExecutor` would be better for I/O workloads. Look for missing `executor.shutdown()` calls.

3. **Connection Management**: Find nodes creating clients in `execute()` instead of `init_components()`. Check for missing `_connection_manager` storage and connection objects being recreated per-request.

4. **Caching**: Identify nodes with deterministic outputs that have `caching.enabled = False`. Look for large objects being cached causing serialization overhead.

5. **Agent Performance**: Check for high `max_loops` values, missing `stop_sequences`, and `parallel_tool_calls_enabled=True` creating new ThreadPoolExecutor per loop.

6. **Async Patterns**: Find blocking calls in async context - `time.sleep()` instead of `asyncio.sleep()`, `requests.*` instead of httpx, sync file I/O in async methods.

7. **Memory**: Look for unbounded growth in `_intermediate_steps`, prompt message accumulation without summarization, and unnecessary deep copies in `clone()`.

For each issue found, provide:
- File path and line number
- Description of the problem
- Suggested fix

Prioritize critical issues that significantly impact performance.
