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
    cm = ConnectionManager()
    connection = HttpConnection(method=HTTPMethod.GET, url="https://example.com")

    async def build():
        return await cm.get_async_connection_client(connection)

    client_loop_1 = asyncio.run(build())
    client_loop_2 = asyncio.run(build())

    assert client_loop_1 is not client_loop_2
    assert len(cm.connection_clients) == 2

    asyncio.run(cm.aclose())
    assert len(cm.connection_clients) == 0


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
