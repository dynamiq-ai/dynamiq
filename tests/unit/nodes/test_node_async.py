import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest

from dynamiq.executors.context import ContextAwareThreadPoolExecutor
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


class CachingAsyncNode(Node):
    """Test node that tracks whether sync or async execute was called."""
    group: NodeGroup = NodeGroup.UTILS
    name: str = "CachingAsync"
    sync_called: bool = False
    async_called: bool = False

    def execute(self, input_data, config=None, **kwargs):
        self.sync_called = True
        return {"result": "sync"}

    async def execute_async(self, input_data, config=None, **kwargs):
        self.async_called = True
        return {"result": "async"}


class TestRunAsyncContextPropagation:
    @pytest.mark.asyncio
    async def test_context_aware_executor_does_not_double_copy(self):
        """When executor is ContextAwareThreadPoolExecutor, run_async should not
        wrap with ctx.run since the executor handles context propagation."""
        node = SyncOnlyNode()
        executor = ContextAwareThreadPoolExecutor(max_workers=2)
        try:
            with patch("dynamiq.nodes.node.contextvars") as mock_contextvars:
                result = await node.run_async(
                    input_data={"input": "test"},
                    config=RunnableConfig(callbacks=[]),
                    executor=executor,
                )
                mock_contextvars.copy_context.assert_not_called()
                assert result.status == RunnableStatus.SUCCESS
        finally:
            executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_regular_executor_still_copies_context(self):
        """When executor is a regular ThreadPoolExecutor, run_async should
        still copy context explicitly."""
        node = SyncOnlyNode()
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            with patch("dynamiq.nodes.node.contextvars") as mock_contextvars:
                mock_ctx = MagicMock()
                mock_ctx.run = lambda fn, *a, **kw: fn(*a, **kw)
                mock_contextvars.copy_context.return_value = mock_ctx
                result = await node.run_async(
                    input_data={"input": "test"},
                    config=RunnableConfig(callbacks=[]),
                    executor=executor,
                )
                mock_contextvars.copy_context.assert_called_once()
                assert result.status == RunnableStatus.SUCCESS
        finally:
            executor.shutdown(wait=False)


class TestAsyncCachingPath:
    @pytest.mark.asyncio
    async def test_cached_async_path_uses_execute_async(self):
        """When caching is enabled in _run_async_native, it should still use
        execute_async_with_retry (async path), not execute_with_retry (sync path)."""
        from dynamiq.nodes.node import CachingConfig

        node = CachingAsyncNode(
            caching=CachingConfig(enabled=True),
        )
        # Run without actual cache config so cache decorator is a passthrough
        result = await node.run_async(
            input_data={"input": "test"},
            config=RunnableConfig(callbacks=[]),
        )
        assert result.status == RunnableStatus.SUCCESS
        assert node.async_called is True
        assert node.sync_called is False
