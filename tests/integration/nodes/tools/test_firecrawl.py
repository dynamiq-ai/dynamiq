import pytest

from dynamiq.connections import Firecrawl
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_firecrawl_response(mocker):
    """Mock response from Firecrawl API."""
    return {
        "success": True,
        "data": {
            "markdown": "# Test Page\n\nThis is a test page with some content.",
            "html": "<html><body><h1>Test Page</h1><p>This is a test page with some content.</p></body></html>",
            "links": [
                {"url": "https://example.com/page1", "text": "Page 1"},
                {"url": "https://example.com/page2", "text": "Page 2"},
            ],
            "metadata": {"title": "Test Page", "description": "Test page description"},
        },
    }


@pytest.fixture
def mock_firecrawl_requests(mocker, mock_firecrawl_response):
    """Mock requests library for Firecrawl API."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_firecrawl_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_firecrawl_basic_scrape(mock_firecrawl_requests, mock_firecrawl_response):
    """Test basic scrape functionality."""
    firecrawl_connection = Firecrawl(api_key="test_key")
    firecrawl_tool = FirecrawlTool(connection=firecrawl_connection)

    input_data = {"url": "https://example.com"}

    result = firecrawl_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["url"] == input_data["url"]

    content = result.output["content"]
    assert content["success"] is True
    assert content["url"] == input_data["url"]
    assert content["markdown"] == mock_firecrawl_response["data"]["markdown"]
    assert content["html"] == mock_firecrawl_response["data"]["html"]

    mock_firecrawl_requests.assert_called_once()
    call_args = mock_firecrawl_requests.call_args
    assert call_args[1]["url"] == "https://api.firecrawl.dev/v1/scrape"

    request_body = call_args[1]["json"]
    assert request_body["url"] == input_data["url"]
    assert request_body["formats"] == ["markdown"]
    assert request_body["onlyMainContent"] is True


def test_firecrawl_agent_optimized(mock_firecrawl_requests, mock_firecrawl_response):
    """Test scraping with agent-optimized output format."""
    firecrawl_connection = Firecrawl(api_key="test_key")
    firecrawl_tool = FirecrawlTool(
        connection=firecrawl_connection,
        is_optimized_for_agents=True,
        formats=["markdown", "html"],
    )

    input_data = {"url": "https://example.com"}

    result = firecrawl_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "## Source URL" in content
    assert input_data["url"] in content
    assert "Markdown Content" in content
    assert mock_firecrawl_response["data"]["markdown"] in content
    assert "HTML" in content
    assert mock_firecrawl_response["data"]["html"] in content

    mock_firecrawl_requests.assert_called_once()
    call_args = mock_firecrawl_requests.call_args
    request_body = call_args[1]["json"]
    assert request_body["formats"] == ["markdown", "html"]


def test_firecrawl_with_custom_options(mock_firecrawl_requests):
    """Test scraping with custom configuration options."""
    firecrawl_connection = Firecrawl(api_key="test_key")
    firecrawl_tool = FirecrawlTool(
        connection=firecrawl_connection,
        formats=["markdown", "screenshot"],
        only_main_content=False,
        exclude_tags=["nav", "footer"],
        wait_for=2000,
        mobile=True,
        remove_base64_images=True,
        timeout=45000,
    )

    input_data = {"url": "https://example.com"}

    firecrawl_tool.run(input_data, None)

    mock_firecrawl_requests.assert_called_once()
    call_args = mock_firecrawl_requests.call_args
    request_body = call_args[1]["json"]

    assert request_body["formats"] == ["markdown", "screenshot"]
    assert request_body["onlyMainContent"] is False
    assert request_body["excludeTags"] == ["nav", "footer"]
    assert request_body["waitFor"] == 2000
    assert request_body["mobile"] is True
    assert request_body["removeBase64Images"] is True
    assert request_body["timeout"] == 45000
