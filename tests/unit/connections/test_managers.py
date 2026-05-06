import asyncio

import httpx
import pytest

from dynamiq.connections.connections import HTTPMethod
from dynamiq.connections.connections import Http as HttpConnection
from dynamiq.connections.connections import Whisper
from dynamiq.connections.managers import (
    ConnectionClientInitType,
    ConnectionManager,
    ConnectionManagerException,
    get_async_connection_manager,
)


def test_connection_manager_caches_client_on_first_connect_and_reuses():
    cm = ConnectionManager()
    connection = Whisper(id="test", api_key="test_key_123")
    connection_another_instance = Whisper(id="test", api_key="test_key_123")

    assert connection.headers == {"Authorization": f"Bearer {connection.api_key}"}
    assert connection_another_instance.headers == {"Authorization": f"Bearer {connection_another_instance.api_key}"}

    client1 = cm.get_connection_client(connection)
    assert client1 is not None
    assert len(cm.connection_clients) == 1

    client2 = cm.get_connection_client(connection)
    assert client1 is client2
    assert len(cm.connection_clients) == 1

    client3 = cm.get_connection_client(connection_another_instance)
    assert client1 is client3
    assert len(cm.connection_clients) == 1

    client4 = cm.get_connection_client(connection_another_instance)
    assert client1 is client4
    assert len(cm.connection_clients) == 1

    cm.close()
    assert len(cm.connection_clients) == 0


def test_get_connection_client_rejects_async_init_type():
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")
    with pytest.raises(ConnectionManagerException):
        cm.get_connection_client(connection, init_type=ConnectionClientInitType.ASYNC)


@pytest.mark.asyncio
async def test_async_client_is_cached_within_loop():
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")

    client1 = await cm.get_async_connection_client(connection)
    client2 = await cm.get_async_connection_client(connection)
    try:
        assert isinstance(client1, httpx.AsyncClient)
        assert client1 is client2
        assert len(cm.connection_clients) == 1
    finally:
        await cm.aclose()
    assert len(cm.connection_clients) == 0


def test_async_client_loop_id_isolation():
    """Sequential ``asyncio.run`` invocations must never share a cached client.

    Loop ids are returned by ``id()`` on the loop object and can be recycled when one
    loop is garbage-collected. The manager uses a weakref to the originating loop on
    each cached client to detect this and rebuild rather than returning a stale client.
    """
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")

    async def build():
        return await cm.get_async_connection_client(connection)

    clients = [asyncio.run(build()) for _ in range(10)]

    assert len({id(c) for c in clients}) == 10, "stale client returned across asyncio.run boundaries"

    asyncio.run(cm.aclose())
    assert len(cm.connection_clients) == 0


@pytest.mark.asyncio
async def test_async_client_eviction_when_underlying_closed():
    """A closed httpx.AsyncClient in the cache must not be returned."""
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")

    client1 = await cm.get_async_connection_client(connection)
    await client1.aclose()
    assert client1.is_closed

    client2 = await cm.get_async_connection_client(connection)
    try:
        assert client2 is not client1
        assert not client2.is_closed
    finally:
        await cm.aclose()


def test_is_client_alive_recognizes_httpx_is_closed():
    """``_is_client_alive`` must inspect ``is_closed`` (httpx) as well as ``closed``."""
    cm = ConnectionManager()

    class _FakeHttpx:
        def __init__(self, closed):
            self.is_closed = closed

    assert cm._is_client_alive(_FakeHttpx(False)) is True
    assert cm._is_client_alive(_FakeHttpx(True)) is False


@pytest.mark.asyncio
async def test_async_connection_manager_context_closes_clients():
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")

    async with get_async_connection_manager() as cm:
        client = await cm.get_async_connection_client(connection)
        assert isinstance(client, httpx.AsyncClient)
        assert len(cm.connection_clients) == 1

    assert len(cm.connection_clients) == 0


@pytest.mark.asyncio
async def test_get_async_connection_client_unsupported_connection():
    cm = ConnectionManager()

    class FakeConnection:
        id = "fake"
        type = "fake.Type"

        def model_dump_json(self):
            return "{}"

    with pytest.raises(ConnectionManagerException):
        await cm.get_async_connection_client(FakeConnection())
