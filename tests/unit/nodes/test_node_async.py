import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from dynamiq.nodes.node import Node, ErrorHandling
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus


class SyncOnlyNode(Node):
    """Test node with only sync execute."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "SyncOnly"

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "sync"}


class NativeAsyncNode(Node):
    """Test node with both sync and async execute."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "NativeAsync"

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "sync"}

    async def execute_async(self, input_data, config=None, **kwargs):
        await asyncio.sleep(0.01)
        return {"result": "async"}


class TestNodeAsyncProtocol:
    def test_sync_only_node_has_no_native_async(self):
        node = SyncOnlyNode()
        assert node.has_native_async is False

    def test_native_async_node_has_native_async(self):
        node = NativeAsyncNode()
        assert node.has_native_async is True

    @pytest.mark.asyncio
    async def test_base_execute_async_returns_not_implemented(self):
        node = SyncOnlyNode()
        result = await node.execute_async(input_data={})
        assert result is NotImplemented


class FailThenSucceedAsyncNode(Node):
    """Test node that fails N times then succeeds."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "FailThenSucceed"
    attempt_count: int = 0
    fail_times: int = 2
    error_handling: ErrorHandling = ErrorHandling(
        max_retries=3, retry_interval_seconds=0.01, backoff_rate=1
    )

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "sync"}

    async def execute_async(self, input_data, config=None, **kwargs):
        self.attempt_count += 1
        if self.attempt_count <= self.fail_times:
            raise ValueError(f"Attempt {self.attempt_count} failed")
        return {"result": "success", "attempts": self.attempt_count}


class TimeoutAsyncNode(Node):
    """Test node that takes too long."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "TimeoutAsync"
    error_handling: ErrorHandling = ErrorHandling(timeout_seconds=0.05)

    def execute(self, input_data, config=None, **kwargs):
        return {"result": "sync"}

    async def execute_async(self, input_data, config=None, **kwargs):
        await asyncio.sleep(10)  # Way longer than timeout
        return {"result": "should not reach"}


class TestExecuteAsyncWithRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        node = FailThenSucceedAsyncNode()
        config = RunnableConfig(callbacks=[])
        result = await node.execute_async_with_retry(input_data={}, config=config)
        assert result == {"result": "success", "attempts": 3}
        assert node.attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        node = FailThenSucceedAsyncNode(fail_times=10)
        config = RunnableConfig(callbacks=[])
        with pytest.raises(ValueError, match="Attempt .* failed"):
            await node.execute_async_with_retry(input_data={}, config=config)

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        node = TimeoutAsyncNode()
        config = RunnableConfig(callbacks=[])
        with pytest.raises(asyncio.TimeoutError):
            await node.execute_async_with_retry(input_data={}, config=config)


class TestRunAsyncRouting:
    @pytest.mark.asyncio
    async def test_sync_node_uses_executor(self):
        """Sync-only node should offload to the provided executor."""
        node = SyncOnlyNode()
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            result = await node.run_async(
                input_data={"input": "test"}, config=RunnableConfig(callbacks=[]), executor=executor
            )
            assert result.status == RunnableStatus.SUCCESS
            assert result.output == {"result": "sync"}
        finally:
            executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_async_node_runs_on_event_loop(self):
        """Async-native node should NOT use executor — runs directly on event loop."""
        node = NativeAsyncNode()
        result = await node.run_async(
            input_data={"input": "test"}, config=RunnableConfig(callbacks=[]), executor=None
        )
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"result": "async"}

    @pytest.mark.asyncio
    async def test_sync_node_without_executor_falls_back_to_default(self):
        """Sync-only node with executor=None should use default executor (backward compat)."""
        node = SyncOnlyNode()
        result = await node.run_async(
            input_data={"input": "test"}, config=RunnableConfig(callbacks=[])
        )
        assert result.status == RunnableStatus.SUCCESS
        assert result.output == {"result": "sync"}
