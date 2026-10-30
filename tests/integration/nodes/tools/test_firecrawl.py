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
                "https://example.com/page1",
                "https://example.com/page2",
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
    assert call_args[1]["url"] == "https://api.firecrawl.dev/v2/scrape"

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
    assert "## Web Scraping Result" in content
    assert input_data["url"] in content
    assert "## Markdown" in content
    assert mock_firecrawl_response["data"]["markdown"] in content
    assert "## HTML" in content
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


def test_firecrawl_leaves_tls_verification_to_firecrawl(mock_firecrawl_requests):
    # Firecrawl verifies TLS for a scrape that sends headers or actions, unless the request turns it off.
    firecrawl_tool = FirecrawlTool(connection=Firecrawl(api_key="test_key"), headers={"Cookie": "session=secret"})

    result = firecrawl_tool.run({"url": "https://example.com"})

    assert result.status == RunnableStatus.SUCCESS
    assert "skipTlsVerification" not in mock_firecrawl_requests.call_args.kwargs["json"]


def test_firecrawl_sends_explicit_tls_setting(mock_firecrawl_requests):
    firecrawl_tool = FirecrawlTool(connection=Firecrawl(api_key="test_key"), skip_tls_verification=True)

    result = firecrawl_tool.run({"url": "https://example.com"})

    assert result.status == RunnableStatus.SUCCESS
    assert mock_firecrawl_requests.call_args.kwargs["json"]["skipTlsVerification"] is True


def test_firecrawl_requests_a_full_page_screenshot_as_v2_expects(mock_firecrawl_requests):
    firecrawl_tool = FirecrawlTool(
        connection=Firecrawl(api_key="test_key"), formats=["markdown", "screenshot@fullPage"]
    )

    result = firecrawl_tool.run({"url": "https://example.com"})

    assert result.status == RunnableStatus.SUCCESS
    assert mock_firecrawl_requests.call_args.kwargs["json"]["formats"] == [
        "markdown",
        {"type": "screenshot", "fullPage": True},
    ]


def test_firecrawl_sends_parser_options_with_the_keys_firecrawl_accepts(mock_firecrawl_requests):
    # The workflow builder stores {"type": "pdf", "maxPages": 5} with its keys in snake_case.
    firecrawl_tool = FirecrawlTool(
        connection=Firecrawl(api_key="test_key"), parsers=[{"type": "pdf", "max_pages": 5}, "image"]
    )

    result = firecrawl_tool.run({"url": "https://example.com/report.pdf"})

    assert result.status == RunnableStatus.SUCCESS
    assert mock_firecrawl_requests.call_args.kwargs["json"]["parsers"] == [{"type": "pdf", "maxPages": 5}, "image"]


def test_firecrawl_gives_agents_the_extracted_json(mocker):
    response = mocker.Mock()
    response.json.return_value = {"success": True, "data": {"json": {"price": "$10", "currency": "USD"}}}
    mocker.patch("requests.request", return_value=response)
    firecrawl_tool = FirecrawlTool(
        connection=Firecrawl(api_key="test_key"), formats=["json"], is_optimized_for_agents=True
    )

    result = firecrawl_tool.run({"url": "https://shop.example.com/item"})

    assert result.status == RunnableStatus.SUCCESS
    assert '## JSON\n```json\n{\n  "price": "$10",\n  "currency": "USD"\n}\n```' in result.output["content"]
