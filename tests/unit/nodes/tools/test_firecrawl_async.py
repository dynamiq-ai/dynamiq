"""Async unit tests for firecrawl and firecrawl_search."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchTool


def _mock_response(status_code=200, json_payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    resp.raise_for_status = MagicMock()
    return resp


def test_firecrawl_has_native_async():
    assert FirecrawlTool.execute_async is not Node.execute_async


def test_firecrawl_search_has_native_async():
    assert FirecrawlSearchTool.execute_async is not Node.execute_async


@pytest.mark.asyncio
async def test_firecrawl_execute_async():
    node = FirecrawlTool(connection=connections.Firecrawl(api_key="k"))
    payload = {"success": True, "data": {"markdown": "# hello"}}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(FirecrawlTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_firecrawl_execute_async_request_failure():
    node = FirecrawlTool(connection=connections.Firecrawl(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=RuntimeError("net"))

    with patch.object(FirecrawlTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_firecrawl_execute_async_missing_url():
    node = FirecrawlTool(connection=connections.Firecrawl(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock()

    with patch.object(FirecrawlTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"
    mock_client.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_firecrawl_search_execute_async():
    node = FirecrawlSearchTool(connection=connections.Firecrawl(api_key="k"))
    payload = {
        "success": True,
        "data": {"web": [{"title": "T", "url": "https://u", "description": "d"}]},
    }
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(FirecrawlSearchTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "foo"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_firecrawl_search_execute_async_missing_query():
    node = FirecrawlSearchTool(connection=connections.Firecrawl(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock()

    with patch.object(FirecrawlSearchTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"
    mock_client.request.assert_not_awaited()
