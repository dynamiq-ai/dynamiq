"""Async unit tests for pipedream."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq.connections.connections import PipedreamOAuth2
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.pipedream import Pipedream


def _mock_response(status_code=200, json_payload=None, text="ok"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    return resp


def _build_node():
    connection = PipedreamOAuth2(access_token="t", project_id="p")
    return Pipedream(
        connection=connection,
        external_user_id="u",
        action_id="a",
        input_props={},
        configurable_props={},
    )


def test_pipedream_has_native_async():
    assert Pipedream.execute_async is not Node.execute_async


@pytest.mark.asyncio
async def test_pipedream_execute_async_success():
    node = _build_node()
    payload = {"exports": {"debug": {"status": 200, "data": "ok"}}}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(Pipedream, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_pipedream_execute_async_failed_status():
    node = _build_node()
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(status_code=500, text="boom"))

    with patch.object(Pipedream, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_pipedream_connection_supports_connect_async():
    import httpx

    connection = PipedreamOAuth2(access_token="t", project_id="p")
    client = await connection.connect_async()
    try:
        assert isinstance(client, httpx.AsyncClient)
    finally:
        await client.aclose()
