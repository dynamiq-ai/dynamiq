import pytest

from dynamiq.connections import Serply
from dynamiq.nodes.tools.serply import SerplyTool


@pytest.fixture
def mock_serply_response():
    """Mock response from Serply Search API."""
    return {
        "results": [
            {
                "title": "Test Result 1",
                "link": "https://example.com/result1",
                "description": "This is a snippet from result 1",
            },
            {
                "title": "Test Result 2",
                "link": "https://example.com/result2",
                "description": "This is a snippet from result 2",
            },
        ]
    }


@pytest.fixture
def mock_requests(mocker, mock_serply_response):
    """Mock requests library."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_serply_response
    mock_response.raise_for_status.return_value = None
    mock_response.status_code = 200

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_basic_search(mock_requests):
    """Test basic search functionality."""
    # Setup
    serply_connection = Serply(api_key="test_key")
    search_tool = SerplyTool(connection=serply_connection)

    # Execute
    input_data = {"query": "test query", "limit": 5}
    result = search_tool.run(input_data)

    # Verify API call
    mock_requests.assert_called_once()
    call_kwargs = mock_requests.call_args[1]
    assert call_kwargs["url"] == "https://api.serply.io/v1/search"
    assert call_kwargs["headers"]["X-Api-Key"] == "test_key"
    assert call_kwargs["params"]["q"] == "test query"
    assert call_kwargs["params"]["num"] == 5

    # Verify response formatting
    assert "result" in result.output["content"]
    assert "Test Result 1" in str(result.output["content"])
    assert "This is a snippet from result 2" in str(result.output["content"])


def test_search_with_custom_params(mock_requests):
    """Test search with localization parameters and the agent-optimized format."""
    # Setup
    serply_connection = Serply(api_key="test_key")
    search_tool = SerplyTool(connection=serply_connection, gl="de", hl="de", is_optimized_for_agents=True)

    # Execute
    input_data = {"query": "test query", "limit": 1}
    result = search_tool.run(input_data)

    # Verify API call
    mock_requests.assert_called_once()
    params = mock_requests.call_args[1]["params"]
    assert params["gl"] == "de"
    assert params["hl"] == "de"

    # Verify response contains agent-optimized format, trimmed to the requested limit
    content = result.output["content"]
    assert "## Sources with URLs" in content
    assert "## Search results for" in content
    assert "Test Result 2" not in content
