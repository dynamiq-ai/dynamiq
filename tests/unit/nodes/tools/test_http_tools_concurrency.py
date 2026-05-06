"""Concurrency-correctness assertion across every HTTP tool with execute_async.

Verifies that gather'd run_async calls overlap rather than serialize. Catches
"accidentally synchronous" regressions in any tool's execute_async.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.connections.connections import PipedreamOAuth2
from dynamiq.nodes.tools import HttpApiCall
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchTool
from dynamiq.nodes.tools.jina import JinaScrapeTool, JinaSearchTool
from dynamiq.nodes.tools.pipedream import Pipedream
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from tests.helpers.async_node import assert_concurrent_execution


SLEEP_S = 0.05
N_CALLS = 10


def _make_mock_response(json_payload):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "ok"
    resp.content = b"ok"
    resp.headers = {"content-type": "application/json"}
    resp.json = MagicMock(return_value=json_payload)
    resp.raise_for_status = MagicMock()
    return resp


def _make_slow_async_client(json_payload):
    async def slow_request(**kwargs):
        await asyncio.sleep(SLEEP_S)
        return _make_mock_response(json_payload)

    client = MagicMock()
    client.request = AsyncMock(side_effect=slow_request)
    return client


@pytest.mark.parametrize(
    ("tool_factory", "tool_cls", "input_payload", "json_payload"),
    [
        (
            lambda: HttpApiCall(
                connection=connections.Http(method=connections.HTTPMethod.GET, url="https://x.io"),
                response_type="json",
            ),
            HttpApiCall,
            {},
            {"a": 1},
        ),
        (
            lambda: TavilyTool(connection=connections.Tavily(api_key="k")),
            TavilyTool,
            {"query": "q"},
            {"results": [], "query": "q"},
        ),
        (
            lambda: ExaTool(connection=connections.Exa(api_key="k")),
            ExaTool,
            {"query": "q"},
            {"results": []},
        ),
        (
            lambda: ScaleSerpTool(connection=connections.ScaleSerp(api_key="k")),
            ScaleSerpTool,
            {"query": "q"},
            {"organic_results": []},
        ),
        (
            lambda: ZenRowsTool(connection=connections.ZenRows(api_key="k")),
            ZenRowsTool,
            {"url": "https://x.io"},
            None,
        ),
        (
            lambda: FirecrawlTool(connection=connections.Firecrawl(api_key="k")),
            FirecrawlTool,
            {"url": "https://x.io"},
            {"success": True, "data": {}},
        ),
        (
            lambda: FirecrawlSearchTool(connection=connections.Firecrawl(api_key="k")),
            FirecrawlSearchTool,
            {"query": "q"},
            {"success": True, "data": {"web": []}},
        ),
        (
            lambda: JinaScrapeTool(connection=connections.Jina(api_key="k")),
            JinaScrapeTool,
            {"url": "https://x.io"},
            {"data": {"content": "c", "links": {}, "images": {}}},
        ),
        (
            lambda: JinaSearchTool(connection=connections.Jina(api_key="k")),
            JinaSearchTool,
            {"query": "q"},
            {"data": []},
        ),
        (
            lambda: Pipedream(
                connection=PipedreamOAuth2(access_token="t", project_id="p"),
                external_user_id="u",
                action_id="a",
                input_props={},
                configurable_props={},
            ),
            Pipedream,
            {},
            {"exports": {"debug": {"status": 200, "data": "ok"}}},
        ),
    ],
    ids=[
        "http_api_call",
        "tavily",
        "exa_search",
        "scale_serp",
        "zenrows",
        "firecrawl",
        "firecrawl_search",
        "jina_scrape",
        "jina_search",
        "pipedream",
    ],
)
@pytest.mark.asyncio
async def test_http_tool_concurrency(tool_factory, tool_cls, input_payload, json_payload):
    node = tool_factory()
    mock_client = _make_slow_async_client(json_payload)

    with patch.object(tool_cls, "get_async_client", AsyncMock(return_value=mock_client)):
        results = await assert_concurrent_execution(
            node,
            payloads=[input_payload for _ in range(N_CALLS)],
            expected_single_call_s=SLEEP_S,
        )

    assert len(results) == N_CALLS
    assert mock_client.request.await_count == N_CALLS
