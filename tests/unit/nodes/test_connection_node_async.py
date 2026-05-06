import httpx
import pytest

from dynamiq.connections.connections import HTTPMethod
from dynamiq.connections.connections import Http as HttpConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode


class _AsyncTestNode(ConnectionNode):
    """Minimal ConnectionNode subclass for exercising async client wiring."""

    group: str = NodeGroup.TOOLS
    name: str = "async-test"

    def execute(self, input_data, config=None, **kwargs):
        return {}


@pytest.mark.asyncio
async def test_get_async_client_returns_httpx_async_client():
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")
    node = _AsyncTestNode(connection=connection)
    node._connection_manager = cm

    client = await node.get_async_client()
    try:
        assert isinstance(client, httpx.AsyncClient)
    finally:
        await cm.aclose()


@pytest.mark.asyncio
async def test_get_async_client_caches_via_manager():
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")
    node = _AsyncTestNode(connection=connection)
    node._connection_manager = cm

    try:
        c1 = await node.get_async_client()
        c2 = await node.get_async_client()
        assert c1 is c2
    finally:
        await cm.aclose()


@pytest.mark.asyncio
async def test_get_async_client_lazily_creates_manager_when_missing():
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")
    node = _AsyncTestNode(connection=connection)
    node._connection_manager = None

    try:
        client = await node.get_async_client()
        assert isinstance(client, httpx.AsyncClient)
        assert node._connection_manager is not None
    finally:
        await node._connection_manager.aclose()


@pytest.mark.asyncio
async def test_get_async_client_without_connection_raises():
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")
    node = _AsyncTestNode(connection=connection, client=object())
    node._connection_manager = cm
    node.connection = None

    with pytest.raises(ValueError):
        await node.get_async_client()
