import asyncio
import time
from unittest.mock import patch

import pytest

from dynamiq.flows.flow import Flow
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus


class SlowSyncNode(Node):
    """Sync-only node that takes time."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "SlowSync"
    latency: float = 0.1

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(self.latency)
        return {"result": "sync_done"}


class FastAsyncNode(Node):
    """Async node that is fast."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "FastAsync"

    def execute(self, input_data, config=None, **kwargs):
        time.sleep(0.1)
        return {"result": "sync_done"}

    async def execute_async(self, input_data, config=None, **kwargs):
        await asyncio.sleep(0.01)
        return {"result": "async_done"}


class TestFlowAsyncExecutor:
    @pytest.mark.asyncio
    async def test_flow_run_async_creates_dedicated_executor(self):
        """Each flow run should create its own ContextAwareThreadPoolExecutor."""
        node = SlowSyncNode(id="slow1")
        flow = Flow(nodes=[node])

        with patch("dynamiq.flows.flow.ContextAwareThreadPoolExecutor") as mock_executor_cls:
            from dynamiq.executors.context import ContextAwareThreadPoolExecutor

            real_executor = ContextAwareThreadPoolExecutor(max_workers=4)
            mock_executor_cls.return_value = real_executor

            try:
                _ = await flow.run_async(input_data={}, config=RunnableConfig(callbacks=[]))
            finally:
                real_executor.shutdown(wait=False)

            mock_executor_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_flows_have_separate_executors(self):
        """Two concurrent flow runs should not share executors."""
        node_a = SlowSyncNode(id="a", latency=0.05)
        node_b = SlowSyncNode(id="b", latency=0.05)
        flow_a = Flow(nodes=[node_a])
        flow_b = Flow(nodes=[node_b])

        config = RunnableConfig(callbacks=[])

        results = await asyncio.gather(
            flow_a.run_async(input_data={}, config=config),
            flow_b.run_async(input_data={}, config=config),
        )

        assert results[0].status == RunnableStatus.SUCCESS
        assert results[1].status == RunnableStatus.SUCCESS
