import pytest

from dynamiq.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl_search import FirecrawlSearchTool, SourceImages, SourceNews, SourceWeb
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_firecrawl_search_response():
    return {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://www.firecrawl.dev/",
                    "title": "Firecrawl - The Web Data API for AI",
                    "description": "The web crawling, scraping, and search API for AI.",
                    "position": 1,
                }
            ],
            "news": [
                {
                    "title": "Firecrawl raises funding",
                    "url": "https://news.example.com/firecrawl",
                    "snippet": "Firecrawl announced new funding.",
                    "position": 1,
                }
            ],
            "images": [
                {
                    "title": "Firecrawl logo",
                    "imageUrl": "https://cdn.example.com/firecrawl.png",
                    "url": "https://docs.firecrawl.dev/",
                    "position": 1,
                }
            ],
        },
    }


@pytest.fixture
def mock_firecrawl_scrape_search_response():
    return {
        "success": True,
        "data": [
            {
                "title": "Firecrawl - The Web Data API",
                "description": "Firecrawl overview page.",
                "url": "https://www.firecrawl.dev/",
                "markdown": "# Firecrawl\n\nAPI for AI agents.",
                "links": ["https://www.firecrawl.dev/docs"],
                "metadata": {"title": "Firecrawl - The Web Data API"},
            }
        ],
    }


@pytest.fixture
def mock_firecrawl_search_requests(mocker, mock_firecrawl_search_response):
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_firecrawl_search_response
    mock_response.raise_for_status.return_value = None
    return mocker.patch("requests.request", return_value=mock_response)


@pytest.fixture
def mock_firecrawl_search_requests_scrape(mocker, mock_firecrawl_scrape_search_response):
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_firecrawl_scrape_search_response
    mock_response.raise_for_status.return_value = None
    return mocker.patch("requests.request", return_value=mock_response)


def test_firecrawl_search_basic(mock_firecrawl_search_requests, mock_firecrawl_search_response):
    connection = Firecrawl(api_key="test_key")
    tool = FirecrawlSearchTool(connection=connection)

    input_data = {"query": "firecrawl"}
    result = tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert content["success"] is True
    assert content["query"] == input_data["query"]
    assert content["data"] == mock_firecrawl_search_response["data"]

    mock_firecrawl_search_requests.assert_called_once()
    call_args = mock_firecrawl_search_requests.call_args
    assert call_args[1]["url"] == "https://api.firecrawl.dev/v2/search"

    payload = call_args[1]["json"]
    assert payload["query"] == input_data["query"]
    assert payload["limit"] == 5
    assert payload["sources"] == [{"type": "web"}]
    assert payload["country"] == "US"
    assert payload["ignoreInvalidURLs"] is False
    assert payload["timeout"] == 60000


def test_firecrawl_search_with_overrides(mock_firecrawl_search_requests_scrape, mock_firecrawl_scrape_search_response):
    connection = Firecrawl(api_key="test_key")
    tool = FirecrawlSearchTool(connection=connection, limit=10, timeout=20000)

    input_data = {
        "query": "firecrawl web scraping",
        "limit": 3,
        "sources": [SourceWeb(tbs="qdr:w"), SourceNews(), SourceImages()],
        "categories": ["github"],
        "location": "Germany",
        "tbs": "qdr:d",
        "timeout": 45000,
        "ignoreInvalidURLs": True,
        "country": "DE",
    }

    result = tool.run(input_data, None)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert content["data"] == mock_firecrawl_scrape_search_response["data"]

    mock_firecrawl_search_requests_scrape.assert_called_once()
    call_args = mock_firecrawl_search_requests_scrape.call_args
    payload = call_args[1]["json"]

    assert payload["query"] == input_data["query"]
    assert payload["limit"] == input_data["limit"]
    assert payload["sources"] == [
        {"type": "web", "tbs": "qdr:w"},
        {"type": "news"},
        {"type": "images"},
    ]
    assert payload["categories"] == ["github"]
    assert payload["location"] == input_data["location"]
    assert payload["tbs"] == input_data["tbs"]
    assert payload["timeout"] == input_data["timeout"]
    assert payload["ignoreInvalidURLs"] is True
    assert payload["country"] == "DE"


def test_firecrawl_search_agent_output(mock_firecrawl_search_requests, mock_firecrawl_search_response):
    connection = Firecrawl(api_key="test_key")
    tool = FirecrawlSearchTool(connection=connection, is_optimized_for_agents=True)

    input_data = {"query": "firecrawl"}
    result = tool.run(input_data, None)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert "Firecrawl Search Results" in content
    assert "Web Results" in content
    assert "https://www.firecrawl.dev/" in content
    assert "News Results" in content
    assert "Image Results" in content
