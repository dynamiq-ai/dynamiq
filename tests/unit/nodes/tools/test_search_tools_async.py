"""Async unit tests for tavily, exa_search, scale_serp, zenrows."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool


def _mock_response(status_code=200, json_payload=None, text="", content=b"", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.json = MagicMock(return_value=json_payload if json_payload is not None else {})
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.parametrize("tool_cls", [TavilyTool, ExaTool, ScaleSerpTool, ZenRowsTool])
def test_tool_has_native_async(tool_cls):
    assert tool_cls.execute_async is not Node.execute_async


@pytest.mark.asyncio
async def test_tavily_execute_async():
    node = TavilyTool(connection=connections.Tavily(api_key="k"))
    payload = {"results": [{"title": "T", "url": "https://u", "content": "c", "score": 0.9}], "query": "q"}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(TavilyTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_tavily_execute_async_request_failure_wraps_exception():
    node = TavilyTool(connection=connections.Tavily(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(TavilyTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_exa_search_execute_async():
    node = ExaTool(connection=connections.Exa(api_key="k"))
    payload = {"results": [{"title": "T", "url": "https://u", "snippet": "s", "score": 0.9}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(ExaTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_scale_serp_execute_async():
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    payload = {"organic_results": [{"title": "T", "link": "https://u", "snippet": "s"}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_scale_serp_execute_async_failed_status():
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(status_code=400, json_payload={"request_info": {"message": "bad"}})
    )

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_scale_serp_execute_async_requires_query_or_url():
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock()

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"
    mock_client.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_zenrows_execute_async():
    node = ZenRowsTool(connection=connections.ZenRows(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(text="<html>scraped</html>"))

    with patch.object(ZenRowsTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "success"
    assert "scraped" in str(result.output["content"])
    mock_client.request.assert_awaited_once()


@pytest.mark.asyncio
async def test_zenrows_execute_async_failed_status():
    node = ZenRowsTool(connection=connections.ZenRows(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(status_code=403, json_payload={"detail": "blocked"})
    )

    with patch.object(ZenRowsTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "failure"
