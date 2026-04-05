import asyncio

import pytest

from dynamiq.nodes.node import Node, ErrorHandling
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


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

    def test_base_execute_async_returns_not_implemented(self):
        node = SyncOnlyNode()
        result = asyncio.get_event_loop().run_until_complete(
            node.execute_async(input_data={})
        )
        assert result is NotImplemented
