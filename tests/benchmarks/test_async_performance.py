"""
Synthetic performance benchmarks for async execution optimization.

Run: pytest tests/benchmarks/test_async_performance.py -v -s
"""
import asyncio
import time

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableConfig, RunnableStatus

from .conftest import (
    BenchmarkMetrics,
    MockAsyncLLMNode,
    MockPassthroughNode,
    MockSyncCPUNode,
    MockSyncLLMNode,
    compute_node_gaps,
    print_comparison,
)


def build_workflow_before():
    """Build test workflow using sync-only nodes (baseline)."""
    llm_a = MockSyncLLMNode(id="llm_a", name="LLM-A", latency=0.15)
    llm_b = MockSyncLLMNode(id="llm_b", name="LLM-B", latency=0.15)
    llm_c = MockSyncLLMNode(id="llm_c", name="LLM-C", latency=0.10)
    aggregator = MockPassthroughNode(id="agg", name="Aggregator")
    cpu_node = MockSyncCPUNode(id="cpu", name="CPU-Work")
    final = MockPassthroughNode(id="final", name="Final")

    # DAG: [A, B] -> Aggregator -> Final
    #       C -> CPU -> Final
    aggregator.depends_on([llm_a, llm_b])
    cpu_node.depends_on(llm_c)
    final.depends_on([aggregator, cpu_node])

    return Workflow(flow=Flow(nodes=[llm_a, llm_b, llm_c, aggregator, cpu_node, final]))


def build_workflow_after():
    """Build test workflow using native async nodes (optimized)."""
    llm_a = MockAsyncLLMNode(id="llm_a", name="LLM-A", latency=0.15)
    llm_b = MockAsyncLLMNode(id="llm_b", name="LLM-B", latency=0.15)
    llm_c = MockAsyncLLMNode(id="llm_c", name="LLM-C", latency=0.10)
    aggregator = MockPassthroughNode(id="agg", name="Aggregator")
    cpu_node = MockSyncCPUNode(id="cpu", name="CPU-Work")
    final = MockPassthroughNode(id="final", name="Final")

    aggregator.depends_on([llm_a, llm_b])
    cpu_node.depends_on(llm_c)
    final.depends_on([aggregator, cpu_node])

    return Workflow(flow=Flow(nodes=[llm_a, llm_b, llm_c, aggregator, cpu_node, final]))


async def run_concurrent_workflows(build_fn, concurrency: int) -> BenchmarkMetrics:
    """Run multiple workflows concurrently and collect metrics."""
    metrics = BenchmarkMetrics(num_workflows=concurrency)

    async def run_single():
        wf = build_fn()
        tracing = TracingCallbackHandler()
        config = RunnableConfig(callbacks=[tracing])

        t0 = time.perf_counter()
        result = await wf.run(input_data={}, config=config)
        elapsed = time.perf_counter() - t0

        metrics.latencies.append(elapsed)
        metrics.node_gaps.extend(compute_node_gaps(tracing))
        return result

    wall_start = time.perf_counter()
    results = await asyncio.gather(*[run_single() for _ in range(concurrency)])
    metrics.total_wall_time = time.perf_counter() - wall_start

    for r in results:
        assert r.status == RunnableStatus.SUCCESS

    return metrics


class TestSyntheticBenchmarks:
    """Before/after benchmarks with synthetic mock nodes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 10, 25, 50])
    async def test_concurrent_workflows(self, concurrency):
        """Compare before vs after at different concurrency levels."""
        before = await run_concurrent_workflows(build_workflow_before, concurrency)
        after = await run_concurrent_workflows(build_workflow_after, concurrency)

        print_comparison(f"{concurrency} concurrent workflows", before, after)

        # After should be no worse than before
        if concurrency >= 10:
            assert after.p50_latency <= before.p50_latency * 1.1, (
                f"p50 regression at concurrency={concurrency}: "
                f"before={before.p50_latency:.3f}s, after={after.p50_latency:.3f}s"
            )
