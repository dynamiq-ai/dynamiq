import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig


class MockAsyncLLMNode(Node):
    """Simulates I/O-bound LLM node with native async support."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "MockAsyncLLM"
    latency: float = 0.15

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(self.latency)
        return {"content": "sync response"}

    async def execute_async(self, input_data, config=None, **kwargs):
        await asyncio.sleep(self.latency)
        return {"content": "async response"}


class MockSyncLLMNode(Node):
    """Simulates I/O-bound LLM node WITHOUT async support (baseline)."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "MockSyncLLM"
    latency: float = 0.15

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(self.latency)
        return {"content": "sync response"}


class MockSyncCPUNode(Node):
    """Simulates CPU-bound node without async support."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "MockCPU"

    def execute(self, input_data, config=None, **kwargs):
        total = sum(i * i for i in range(100_000))
        return {"result": total}


class MockPassthroughNode(Node):
    """Lightweight passthrough node."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "Passthrough"

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "pass"}


@dataclass
class BenchmarkMetrics:
    """Collected metrics from a benchmark run."""
    latencies: list[float] = field(default_factory=list)
    node_gaps: list[float] = field(default_factory=list)
    total_wall_time: float = 0.0
    num_workflows: int = 0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def avg_gap(self) -> float:
        if not self.node_gaps:
            return 0.0
        return statistics.mean(self.node_gaps)

    @property
    def throughput(self) -> float:
        if self.total_wall_time == 0:
            return 0.0
        return self.num_workflows / self.total_wall_time


def compute_node_gaps(tracing: TracingCallbackHandler) -> list[float]:
    """Extract inter-node gaps from tracing runs."""
    gaps = []
    runs = list(tracing.runs.values())
    node_runs = [r for r in runs if r.type.value == "node"]
    node_runs.sort(key=lambda r: r.start_time)

    for i in range(1, len(node_runs)):
        prev_end = node_runs[i - 1].end_time
        curr_start = node_runs[i].start_time
        if prev_end and curr_start:
            gap = (curr_start - prev_end).total_seconds()
            if gap > 0:
                gaps.append(gap)

    return gaps


def print_comparison(scenario: str, before: BenchmarkMetrics, after: BenchmarkMetrics):
    """Print a before/after comparison table."""
    def improvement(before_val, after_val):
        if before_val == 0:
            return "N/A"
        pct = ((before_val - after_val) / before_val) * 100
        return f"{pct:+.1f}%"

    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario}")
    print(f"{'=' * 70}")
    print(f"{'Metric':<25} | {'Before':>12} | {'After':>12} | {'Change':>10}")
    print(f"{'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
    print(
        f"{'e2e p50 (ms)':<25} | {before.p50_latency*1000:>12.1f} | "
        f"{after.p50_latency*1000:>12.1f} | {improvement(before.p50_latency, after.p50_latency):>10}"
    )
    print(
        f"{'e2e p95 (ms)':<25} | {before.p95_latency*1000:>12.1f} | "
        f"{after.p95_latency*1000:>12.1f} | {improvement(before.p95_latency, after.p95_latency):>10}"
    )
    print(
        f"{'avg node gap (ms)':<25} | {before.avg_gap*1000:>12.1f} | "
        f"{after.avg_gap*1000:>12.1f} | {improvement(before.avg_gap, after.avg_gap):>10}"
    )
    print(
        f"{'throughput (wf/sec)':<25} | {before.throughput:>12.1f} | "
        f"{after.throughput:>12.1f} | {improvement(-before.throughput, -after.throughput):>10}"
    )
    print(f"{'=' * 70}\n")
