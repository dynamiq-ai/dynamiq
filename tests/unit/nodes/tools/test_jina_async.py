"""Async unit tests for jina scrape and search tools."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.jina import JinaScrapeTool, JinaSearchTool


def _mock_response(status_code=200, json_payload=None, content=b""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def test_jina_scrape_has_native_async():
    assert JinaScrapeTool.execute_async is not Node.execute_async


def test_jina_search_has_native_async():
    assert JinaSearchTool.execute_async is not Node.execute_async


@pytest.mark.asyncio
async def test_jina_scrape_execute_async():
    node = JinaScrapeTool(connection=connections.Jina(api_key="k"))
    payload = {"data": {"content": "scraped markdown", "links": {}, "images": {}}}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(JinaScrapeTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_jina_scrape_execute_async_failure():
    node = JinaScrapeTool(connection=connections.Jina(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(JinaScrapeTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_jina_search_execute_async():
    node = JinaSearchTool(connection=connections.Jina(api_key="k"))
    payload = {"data": [{"title": "T", "url": "https://u", "snippet": "s"}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(JinaSearchTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "foo"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_jina_search_execute_async_failure():
    node = JinaSearchTool(connection=connections.Jina(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(JinaSearchTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "foo"})

    assert result.status.value == "failure"
