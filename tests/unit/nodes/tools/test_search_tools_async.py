"""Async unit tests for tavily, exa_search, linkup_search, scale_serp, serply, zenrows, fxmacrodata."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchTool
from dynamiq.nodes.tools.fxmacrodata import FXMacroDataTool
from dynamiq.nodes.tools.jina import JinaSearchTool
from dynamiq.nodes.tools.linkup_search import LinkupTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.serply import SerplyTool
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


@pytest.mark.parametrize(
  "tool_cls", [TavilyTool, ExaTool, LinkupTool, ScaleSerpTool, SerplyTool, ZenRowsTool, FXMacroDataTool]
)
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
async def test_linkup_search_execute_async():
    node = LinkupTool(connection=connections.Linkup(api_key="k"))
    payload = {"results": [{"type": "text", "name": "T", "url": "https://u", "content": "c"}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(LinkupTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q", "max_results": 3})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["url"] == "https://api.linkup.so/v1/search"
    assert call_kwargs["headers"]["Authorization"] == "Bearer k"
    assert call_kwargs["json"]["q"] == "q"
    assert call_kwargs["json"]["maxResults"] == 3
    assert result.output["content"]["urls"] == ["https://u"]


@pytest.mark.asyncio
async def test_linkup_search_execute_async_failed_status():
    node = LinkupTool(connection=connections.Linkup(api_key="k"))
    response = _mock_response(status_code=401)
    response.raise_for_status.side_effect = RuntimeError("401 Client Error: Unauthorized")
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=response)

    with patch.object(LinkupTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "failure"
    assert "Unauthorized" in result.error.message


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


@pytest.mark.asyncio
async def test_serply_execute_async():
    node = SerplyTool(connection=connections.Serply(api_key="k"))
    payload = {"results": [{"title": "T", "link": "https://u", "description": "d"}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(SerplyTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q", "limit": 3})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["headers"]["X-Api-Key"] == "k"
    assert call_kwargs["params"] == {"q": "q", "num": 3}
    assert result.output["content"]["urls"] == ["https://u"]


@pytest.mark.asyncio
async def test_serply_execute_async_failed_status():
    node = SerplyTool(connection=connections.Serply(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(status_code=401, json_payload={"detail": "Invalid API key"})
    )

    with patch.object(SerplyTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "failure"
    assert "Invalid API key" in result.error.message


@pytest.mark.asyncio
async def test_serply_execute_async_requires_query():
    node = SerplyTool(connection=connections.Serply(api_key="k"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock()

    with patch.object(SerplyTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"
    mock_client.request.assert_not_awaited()


@pytest.mark.asyncio
async def test_fxmacrodata_execute_async():
    node = FXMacroDataTool(connection=connections.FXMacroData(api_key="test-key"))
    payload = {"currency": "USD", "name": "Policy Rate", "source": "FRB", "data": [{"date": "2026-09-17", "val": 3.75}]}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(FXMacroDataTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"currency": "usd", "indicator": "policy_rate", "limit": 1})

    assert result.status.value == "success"
    mock_client.request.assert_awaited_once()
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["url"] == "https://api.fxmacrodata.com/v1/announcements/usd/policy_rate"
    assert call_kwargs["headers"]["X-API-Key"] == "test-key"
    assert call_kwargs["params"] == {"limit": 1}
    assert call_kwargs["follow_redirects"] is False
    assert "2026-09-17: 3.75" in result.output["content"]["result"]


@pytest.mark.asyncio
async def test_fxmacrodata_execute_async_failed_status():
    node = FXMacroDataTool(connection=connections.FXMacroData(api_key="test-key"))
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(status_code=404, json_payload={"detail": "Unknown indicator"})
    )

    with patch.object(FXMacroDataTool, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={"currency": "usd", "indicator": "nope"})

    assert result.status.value == "failure"
    assert "Unknown indicator" in result.error.message


def _recording_client(requests_seen: list[httpx.Request], response: httpx.Response) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        return response

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_scale_serp_execute_async_sends_search_type_value():
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"))
    requests_seen = []
    client = _recording_client(requests_seen, httpx.Response(200, json={"news_results": []}))

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=client)):
        result = await node.run_async(input_data={"query": "q", "search_type": "news"})

    assert result.status.value == "success"
    assert requests_seen[0].url.params["search_type"] == "news"


@pytest.mark.asyncio
@pytest.mark.parametrize(("markdown_response", "response_type"), [(True, "markdown"), (False, None)])
async def test_zenrows_requests_markdown_with_response_type(markdown_response, response_type):
    node = ZenRowsTool(connection=connections.ZenRows(api_key="k"), markdown_response=markdown_response)
    requests_seen = []
    client = _recording_client(requests_seen, httpx.Response(200, text="# scraped"))

    with patch.object(ZenRowsTool, "get_async_client", AsyncMock(return_value=client)):
        result = await node.run_async(input_data={"url": "https://example.com"})

    assert result.status.value == "success"
    params = requests_seen[0].url.params
    assert params.get("response_type") == response_type
    assert "markdown_response" not in params


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["html", "csv"])
async def test_scale_serp_returns_html_and_csv_output_as_text(output):
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"), output=output)
    client = _recording_client([], httpx.Response(200, text="<html>results</html>"))

    with patch.object(ScaleSerpTool, "get_async_client", AsyncMock(return_value=client)):
        result = await node.run_async(input_data={"query": "q"})

    assert result.status.value == "success"
    assert result.output["content"]["result"] == "<html>results</html>"


def test_scale_serp_returns_html_output_to_agents_as_text():
    node = ScaleSerpTool(connection=connections.ScaleSerp(api_key="k"), output="html", is_optimized_for_agents=True)
    response = _mock_response(text="<html>results</html>")
    response.json.side_effect = ValueError("not JSON")

    with patch("requests.request", return_value=response):
        result = node.run(input_data={"query": "q"})

    assert result.status.value == "success"
    assert result.output["content"] == "<html>results</html>"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("node_factory", "input_data"),
    [
        (lambda: TavilyTool(connection=connections.Tavily(api_key="k")), {"query": "q"}),
        (lambda: ExaTool(connection=connections.Exa(api_key="k")), {"query": "q"}),
        (lambda: LinkupTool(connection=connections.Linkup(api_key="k")), {"query": "q"}),
        (lambda: JinaSearchTool(connection=connections.Jina(api_key="k")), {"query": "q"}),
        (lambda: FirecrawlSearchTool(connection=connections.Firecrawl(api_key="k")), {"query": "q"}),
        (lambda: FirecrawlTool(connection=connections.Firecrawl(api_key="k")), {"url": "https://example.com"}),
    ],
    ids=["tavily", "exa", "linkup", "jina", "firecrawl-search", "firecrawl"],
)
async def test_search_tools_quote_why_the_provider_refused_the_request(node_factory, input_data):
    node = node_factory()
    refusal = {"error": "The company category does not support excludeDomains."}
    client = _recording_client([], httpx.Response(400, json=refusal))

    with patch.object(type(node), "get_async_client", AsyncMock(return_value=client)):
        result = await node.run_async(input_data=input_data)

    assert result.status.value == "failure"
    assert "The company category does not support excludeDomains." in result.error.message


@pytest.mark.asyncio
async def test_zenrows_scrapes_the_url_set_on_the_node():
    node = ZenRowsTool(connection=connections.ZenRows(api_key="k"), url="https://example.com/pricing")
    requests_seen = []
    client = _recording_client(requests_seen, httpx.Response(200, text="# Pricing"))

    with patch.object(ZenRowsTool, "get_async_client", AsyncMock(return_value=client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "success"
    assert requests_seen[0].url.params["url"] == "https://example.com/pricing"
    assert result.output["content"]["url"] == "https://example.com/pricing"
