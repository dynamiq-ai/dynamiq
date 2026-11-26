"""Unit tests for SpecStreamSearchTool and SpecStreamExtractTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.specstream_extract import SpecStreamExtractTool
from dynamiq.nodes.tools.specstream_search import SpecStreamSearchTool


def _mock_response(status_code=200, json_payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    resp.raise_for_status = MagicMock()
    return resp


def test_specstream_tools_have_native_async():
    assert SpecStreamSearchTool.execute_async is not Node.execute_async
    assert SpecStreamExtractTool.execute_async is not Node.execute_async


def test_specstream_connection_headers():
    conn = connections.SpecStream(api_key="test-key")
    assert conn.headers["Authorization"] == "Bearer test-key"
    assert conn.headers["Content-Type"] == "application/json"


def test_specstream_search_execute_sync():
    mock_client = MagicMock()
    mock_client.is_closed = False
    mock_client.closed = False
    node = SpecStreamSearchTool(connection=connections.SpecStream(api_key="k"), client=mock_client)
    payload = {
        "sources": [
            {
                "title": "Python Docs",
                "url": "https://docs.python.org",
                "snippet": "Short snippet",
                "excerpt": "Detailed passage",
            }
        ]
    }
    mock_client.request = MagicMock(return_value=_mock_response(json_payload=payload))

    result = node.run(input_data={"query": "python 3.13", "num": 5, "speed": "fast"})
    assert result.status.value == "success"
    mock_client.request.assert_called_once()
    req_kwargs = mock_client.request.call_args.kwargs
    assert req_kwargs["url"] == "https://api.specstream.ai/v1/web-search"
    assert req_kwargs["json"] == {"q": "python 3.13", "num": 5, "speed": "fast", "passages": True}
    assert "Detailed passage" in result.output["content"]["result"]


@pytest.mark.asyncio
async def test_specstream_search_execute_async():
    node = SpecStreamSearchTool(connection=connections.SpecStream(api_key="k"), is_optimized_for_agents=True)
    payload = {
        "sources": [
            {"title": "Example", "url": "https://example.com", "snippet": "Snippet text"}
        ]
    }
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(SpecStreamSearchTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "example"})

    assert result.status.value == "success"
    assert result.output["urls"] == ["https://example.com"]
    assert "Snippet text" in result.output["content"]


@pytest.mark.asyncio
async def test_specstream_extract_execute_async():
    node = SpecStreamExtractTool(connection=connections.SpecStream(api_key="k"), is_optimized_for_agents=True)
    payload = {
        "results": [
            {
                "url": "https://example.com/doc",
                "ok": True,
                "title": "Doc Title",
                "text": "Clean extracted body.",
            }
        ]
    }
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(SpecStreamExtractTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com/doc"})

    assert result.status.value == "success"
    req_kwargs = mock_client.request.call_args.kwargs
    assert req_kwargs["url"] == "https://api.specstream.ai/v1/extract"
    assert req_kwargs["json"] == {"urls": ["https://example.com/doc"]}
    assert "Clean extracted body." in result.output["content"]


@pytest.mark.asyncio
async def test_specstream_extract_missing_url_fails():
    node = SpecStreamExtractTool(connection=connections.SpecStream(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock()

    with patch.object(SpecStreamExtractTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"
    mock_client.request.assert_not_awaited()
