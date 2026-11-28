"""A web tool must keep the path of the base URL it was configured with.

``Http.url`` is a plain string, so a gateway or a self-hosted deployment is configured as
``https://gw.example.com/firecrawl/v2`` -- a base whose last segment belongs to the path
rather than being an endpoint directory. Firecrawl's scrape tool appended ``"scrape"`` to
that string, so it posted to ``.../v2scrape``; the search tools passed the relative
``"search"`` to ``urljoin``, which replaces the last segment, so they posted to
``.../firecrawl/search``. Both are 404s that stay hidden while the base URL ends with ``/``,
as every default does.
"""

import pytest

from dynamiq.connections import Firecrawl, Linkup, ScaleSerp, Tavily
from dynamiq.nodes.tools.firecrawl import FirecrawlInputSchema, FirecrawlTool
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchInputSchema, FirecrawlSearchTool
from dynamiq.nodes.tools.linkup_search import LinkupInputSchema, LinkupTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpInputSchema, ScaleSerpTool
from dynamiq.nodes.tools.tavily import TavilyInputSchema, TavilyTool

GATEWAY = "https://gateway.example.com"
PAGE = "https://example.com/article"


def _both_forms(base: str) -> list[str]:
    """The base URL as an operator writes it, with and without the trailing slash."""
    return [base, base + "/"]


@pytest.mark.parametrize("base_url", _both_forms(f"{GATEWAY}/firecrawl/v2"))
def test_firecrawl_scrape_keeps_the_base_url_path(base_url):
    tool = FirecrawlTool(connection=Firecrawl(api_key="test", url=base_url))

    _, _, request_url = tool._resolve_scrape_request(FirecrawlInputSchema(url=PAGE))

    assert request_url == f"{GATEWAY}/firecrawl/v2/scrape"


@pytest.mark.parametrize("base_url", _both_forms(f"{GATEWAY}/firecrawl/v2"))
def test_firecrawl_search_keeps_the_base_url_path(base_url):
    tool = FirecrawlSearchTool(connection=Firecrawl(api_key="test", url=base_url))

    _, _, request_url = tool._resolve_search_request(FirecrawlSearchInputSchema(query="dynamiq"))

    assert request_url == f"{GATEWAY}/firecrawl/v2/search"


@pytest.mark.parametrize("base_url", _both_forms(f"{GATEWAY}/linkup/v1"))
def test_linkup_search_keeps_the_base_url_path(base_url):
    tool = LinkupTool(connection=Linkup(api_key="test", url=base_url))

    _, request_url = tool._build_search_payload(LinkupInputSchema(query="dynamiq"))

    assert request_url == f"{GATEWAY}/linkup/v1/search"


@pytest.mark.parametrize("base_url", _both_forms(f"{GATEWAY}/tavily"))
def test_tavily_search_keeps_the_base_url_path(base_url):
    tool = TavilyTool(connection=Tavily(api_key="test", url=base_url))

    _, request_url = tool._build_search_payload(TavilyInputSchema(query="dynamiq"))

    assert request_url == f"{GATEWAY}/tavily/search"


@pytest.mark.parametrize("base_url", _both_forms(f"{GATEWAY}/serp"))
def test_scale_serp_search_keeps_the_base_url_path(base_url):
    tool = ScaleSerpTool(connection=ScaleSerp(api_key="test", url=base_url))

    request_kwargs, _, _, _ = tool._build_request_kwargs(ScaleSerpInputSchema(query="dynamiq"))

    assert request_kwargs["url"] == f"{GATEWAY}/serp/search"


def test_firecrawl_scrape_posts_to_the_resolved_url(mocker):
    """The URL the resolver returns is the one the tool actually posts to."""
    mock_request = mocker.patch(
        "requests.request",
        return_value=mocker.Mock(status_code=200, json=lambda: {"success": True, "data": {"markdown": "# Page"}}),
    )
    tool = FirecrawlTool(connection=Firecrawl(api_key="test", url=f"{GATEWAY}/firecrawl/v2"))

    tool.run({"url": PAGE}, None)

    assert mock_request.call_args.kwargs["url"] == f"{GATEWAY}/firecrawl/v2/scrape"
