import pytest
from pydantic import ConfigDict

from dynamiq.connections import Jina
from dynamiq.nodes.tools.jina import JinaScrapeTool, JinaSearchTool
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_jina_search_response(mocker):
    """Mock response from Jina API."""
    return {
        "data": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1",
                "description": "Test Article 1 description",
                "content": "This is content of Test Article 1",
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/article2",
                "description": "Test Article 2 description",
                "content": "This is content of Test Article 2",
            },
        ],
    }


@pytest.fixture
def mock_search_requests(mocker, mock_jina_search_response):
    """Mock requests library for searching with Jina API."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_jina_search_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


@pytest.fixture
def mock_scrape_requests(mocker):
    """Mock requests library for scraping with Jina API."""
    mock_response = mocker.Mock()
    mock_response.text = "Mock response"
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_jina_basic_search(mock_search_requests, mock_jina_search_response):
    """Test basic search functionality."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaSearchTool(connection=jina_connection, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence", "max_results": 2}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]
    assert input_dump["max_results"] == input_data["max_results"]

    mock_search_requests.assert_called_once()
    call_args = mock_search_requests.call_args
    assert call_args[1]["url"] == "https://s.jina.ai/artificial intelligence"
    assert call_args[1]["params"]["count"] == 2


def test_jina_search_agent_optimized(mock_search_requests, mock_jina_search_response):
    """Test search with agent-optimized output format."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaSearchTool(connection=jina_connection, is_optimized_for_agents=True, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "<Sources with URLs>" in content
    assert f"<Search results for query {input_data['query']}>" in content
    assert all(f"[{r['title']}]({r['url']})" in content for r in mock_jina_search_response["data"])

    for result in mock_jina_search_response["data"]:
        if "content" in result:
            assert any(highlight in content for highlight in result["content"])


def test_jina_basic_scraping(mock_scrape_requests):
    """Test basic scrape functionality"""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaScrapeTool(connection=jina_connection, model_config=ConfigDict())

    input_data = {"url": "https://your-url"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["url"] == input_data["url"]

    content = result.output["content"]
    assert content["content"] == "Mock response"
    assert content["url"] == input_data["url"]

    mock_scrape_requests.assert_called_once()
    call_args = mock_scrape_requests.call_args
    assert call_args[1]["url"] == "https://r.jina.ai/https://your-url"
    assert call_args[1]["headers"]["X-Timeout"] == "60000"


def test_jina_scrape_agent_optimized(mock_scrape_requests):
    """Test scraping with agent-optimized output format."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaScrapeTool(connection=jina_connection, is_optimized_for_agents=True, model_config=ConfigDict())

    input_data = {"url": "https://your-url"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "<Source URL>" in content
    assert "<Scraped result>" in content
