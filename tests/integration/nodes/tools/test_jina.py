import pytest
from pydantic import ConfigDict

from dynamiq.connections import Jina
from dynamiq.nodes.tools.jina import JinaScrapeTool, JinaSearchTool
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_jina_search_response(mocker):
    """Mock response from Jina Search API."""
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
def mock_jina_scrape_response(mocker):
    """Mock response from Jina Reader API."""
    return {
        "code": 200,
        "status": 20000,
        "data": {
            "title": "Test Page Title",
            "description": "Test page description",
            "url": "https://your-url",
            "content": "# Test Page Title\n\nThis is the scraped content from the test page.",
            "links": {"Example Link": "https://example.com", "Another Link": "https://another.com"},
            "images": {"Test Image": "https://example.com/image.jpg"},
        },
    }


@pytest.fixture
def mock_search_requests(mocker, mock_jina_search_response):
    """Mock requests library for searching with Jina API."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_jina_search_response
    mock_response.raise_for_status.return_value = None

    # Patch at the requests level since that's what the connection uses
    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


@pytest.fixture
def mock_scrape_requests(mocker, mock_jina_scrape_response):
    """Mock requests library for scraping with Jina API."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_jina_scrape_response
    mock_response.text = "Mock response text"  # Fallback for non-JSON responses
    mock_response.content = b"Mock response content"  # For screenshot responses
    mock_response.raise_for_status.return_value = None

    # Patch at the requests level since that's what the connection uses
    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_jina_basic_search(mock_search_requests, mock_jina_search_response):
    """Test basic search functionality."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaSearchTool(connection=jina_connection, model_config=ConfigDict(), include_full_content=True)

    input_data = {"query": "artificial intelligence", "max_results": 2}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]
    assert input_dump["max_results"] == input_data["max_results"]

    # Verify the correct API call was made
    mock_search_requests.assert_called_once()
    call_args = mock_search_requests.call_args

    # Check method and URL (accessing kwargs)
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["url"] == "https://s.jina.ai/"

    # Check request body
    request_body = call_args.kwargs["json"]
    assert request_body["q"] == "artificial intelligence"
    assert request_body["num"] == 2

    # Check headers
    headers = call_args.kwargs["headers"]
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"


def test_jina_search_agent_optimized(mock_search_requests, mock_jina_search_response):
    """Test search with agent-optimized output format."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaSearchTool(
        connection=jina_connection, is_optimized_for_agents=True, model_config=ConfigDict(), include_full_content=True
    )

    input_data = {"query": "artificial intelligence"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "## Sources with URLs" in content
    assert f"## Search results for query '{input_data['query']}'\n" in content
    assert all(f"[{r['title']}]({r['url']})" in content for r in mock_jina_search_response["data"])

    for result_data in mock_jina_search_response["data"]:
        if "content" in result_data:
            assert result_data["content"] in content
        if "description" in result_data:
            assert result_data["description"] in content


def test_jina_basic_scraping(mock_scrape_requests, mock_jina_scrape_response):
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
    assert content["url"] == input_data["url"]
    assert content["content"] == mock_jina_scrape_response["data"]["content"]
    assert content["links"] == mock_jina_scrape_response["data"]["links"]
    assert content["images"] == mock_jina_scrape_response["data"]["images"]

    # Verify the correct API call was made
    mock_scrape_requests.assert_called_once()
    call_args = mock_scrape_requests.call_args

    # Check method and URL
    assert call_args.kwargs["method"] == "POST"
    assert call_args.kwargs["url"] == "https://r.jina.ai/"

    # Check request body
    request_body = call_args.kwargs["json"]
    assert request_body["url"] == "https://your-url"

    # Check headers
    headers = call_args.kwargs["headers"]
    assert headers["X-Timeout"] == "60"
    assert "X-Return-Format" in headers


def test_jina_scrape_agent_optimized(mock_scrape_requests, mock_jina_scrape_response):
    """Test scraping with agent-optimized output format."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaScrapeTool(connection=jina_connection, is_optimized_for_agents=True, model_config=ConfigDict())

    input_data = {"url": "https://your-url"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "## Source URL" in content
    assert "## Scraped Content" in content
    assert "## Links Found" in content
    assert "## Images Found" in content
    assert "https://your-url" in content


def test_jina_search_with_additional_parameters(mock_search_requests, mock_jina_search_response):
    """Test search with additional parameters."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaSearchTool(connection=jina_connection, model_config=ConfigDict())

    input_data = {
        "query": "machine learning",
        "max_results": 10,
        "country": "US",
        "language": "en",
        "include_links": True,
    }

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    # Verify API call parameters
    call_args = mock_search_requests.call_args
    request_body = call_args.kwargs["json"]
    headers = call_args.kwargs["headers"]

    assert request_body["q"] == "machine learning"
    assert request_body["num"] == 10
    assert request_body["gl"] == "US"
    assert request_body["hl"] == "en"
    assert headers["X-With-Links-Summary"] == "true"


def test_jina_scrape_with_selectors(mock_scrape_requests, mock_jina_scrape_response):
    """Test scraping with CSS selectors."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaScrapeTool(connection=jina_connection, model_config=ConfigDict())

    input_data = {
        "url": "https://example.com",
        "target_selector": "main .content",
        "remove_selector": "header, footer, .ads",
        "include_links": True,
        "include_images": True,
        "engine": "browser",
    }

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    # Verify headers contain selector information
    call_args = mock_scrape_requests.call_args
    headers = call_args.kwargs["headers"]

    assert headers["X-Target-Selector"] == "main .content"
    assert headers["X-Remove-Selector"] == "header, footer, .ads"
    assert headers["X-With-Links-Summary"] == "true"
    assert headers["X-With-Images-Summary"] == "true"
    assert headers["X-Engine"] == "browser"


def test_jina_scrape_screenshot_format(mock_scrape_requests):
    """Test scraping with screenshot format."""
    jina_connection = Jina(api_key="test_key")
    jina_tool = JinaScrapeTool(connection=jina_connection, model_config=ConfigDict(), response_format="screenshot")

    input_data = {"url": "https://example.com"}

    result = jina_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    # Should return binary content for screenshots
    content = result.output["content"]
    assert content["content"] == b"Mock response content"

    call_args = mock_scrape_requests.call_args
    headers = call_args.kwargs["headers"]
    assert headers["X-Return-Format"] == "screenshot"
