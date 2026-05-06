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
async def test_scale_serp_input_search_type_drives_formatting():
    """When input_data overrides the node's default search_type, formatted markdown
    and structured content must both come from the input-resolved result key.

    Pre-existing bug: ``_format_search_results`` used ``self.search_type.result_key``
    while the structured ``content_results`` used the input-resolved type, so a
    user requesting ``news`` on a ``web``-default node would see organic_results in
    the markdown and news_results in the structured payload.
    """
    from dynamiq.nodes.tools.scale_serp import SearchType

    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    assert node.search_type == SearchType.WEB

    # Server returns BOTH news_results and organic_results so we can detect which
    # the formatter actually pulled from.
    payload = {
        "news_results": [{"title": "NewsTitle", "link": "https://news", "snippet": "news snippet"}],
        "organic_results": [{"title": "OrganicTitle", "link": "https://web", "snippet": "web snippet"}],
    }
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q", "search_type": "news"})

    assert result.status.value == "success"
    # Inspect the formatted markdown and structured sources, not raw_response
    # (which contains both result keys verbatim).
    formatted = result.output["content"]["result"]
    sources = result.output["content"]["sources_with_url"]
    assert "NewsTitle" in formatted
    assert "OrganicTitle" not in formatted
    assert any("NewsTitle" in s for s in sources)
    assert not any("OrganicTitle" in s for s in sources)


@pytest.mark.parametrize("include_html, expected", [(True, "True"), (False, "False")])
def test_scale_serp_get_params_stringifies_include_html(include_html, expected):
    """``include_html`` must reach the wire as Python bool repr ("True"/"False").

    Regression: ``requests`` calls ``str(True)`` -> ``"True"`` while ``httpx``
    encodes bools as ``"true"``. Pinning the string at ``get_params`` time keeps
    sync and async query strings byte-identical for case-sensitive upstreams.
    """
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    params = node.get_params(query="q", include_html=include_html)
    assert params["include_html"] == expected
    assert isinstance(params["include_html"], str)


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
