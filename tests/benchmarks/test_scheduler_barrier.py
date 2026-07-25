"""Micro-benchmark: as-soon-as-ready dispatch in Flow.run_async.

Proves that once a node's deps finish, downstream nodes start immediately
rather than waiting for slower peers in the current scheduling batch.
"""
import asyncio
import time

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableStatus


class TimingAsyncNode(Node):
    """Async mock node that records when it starts and ends."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "TimingAsync"
    latency: float = 0.01
    started_at: float = 0.0
    ended_at: float = 0.0

    def execute(self, input_data, config=None, **kwargs):
        self.started_at = time.perf_counter()
        time.sleep(self.latency)
        self.ended_at = time.perf_counter()
        return {"content": "ok"}

    async def execute_async(self, input_data, config=None, **kwargs):
        self.started_at = time.perf_counter()
        await asyncio.sleep(self.latency)
        self.ended_at = time.perf_counter()
        return {"content": "ok"}


def build_barrier_dag() -> tuple[Workflow, dict[str, TimingAsyncNode]]:
    """Build a DAG that exposes the batch-barrier if present."""
    root = TimingAsyncNode(id="root", name="root", latency=0.01)
    slow = TimingAsyncNode(id="slow", name="slow", latency=0.50)
    fast = TimingAsyncNode(id="fast", name="fast", latency=0.01)
    after_slow = TimingAsyncNode(id="after_slow", name="after_slow", latency=0.01)
    after_fast = TimingAsyncNode(id="after_fast", name="after_fast", latency=0.01)

    slow.depends_on(root)
    fast.depends_on(root)
    after_slow.depends_on(slow)
    after_fast.depends_on(fast)

    wf = Workflow(flow=Flow(nodes=[root, slow, fast, after_slow, after_fast]))
    nodes = {n.id: n for n in (root, slow, fast, after_slow, after_fast)}
    return wf, nodes


@pytest.mark.asyncio
async def test_fast_branch_not_blocked_by_slow_peer():
    """after_fast must start soon after fast ends, not after slow ends."""
    wf, nodes = build_barrier_dag()

    result = await wf.run(input_data={})
    assert result.status == RunnableStatus.SUCCESS

    gap_fast = nodes["after_fast"].started_at - nodes["fast"].ended_at
    gap_slow = nodes["after_slow"].started_at - nodes["slow"].ended_at

    # Tight threshold: under the new scheduler the gap should be dominated
    # by scheduler overhead (tens of ms max), never by peer latencies.
    assert gap_fast < 0.05, (
        f"after_fast started {gap_fast*1000:.1f}ms after fast finished; "
        f"batch barrier appears to still be present (slow latency was 500ms)"
    )
    assert gap_slow < 0.05, (
        f"after_slow started {gap_slow*1000:.1f}ms after slow finished"
    )
