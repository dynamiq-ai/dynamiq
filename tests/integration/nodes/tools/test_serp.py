import pytest
from pydantic import ConfigDict

from dynamiq.connections import ScaleSerp
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool, SearchType


@pytest.fixture
def mock_scale_response(mocker):
    """Mock response from Scale SERP API."""
    return {
        "organic_results": [
            {
                "title": "Test Result 1",
                "link": "https://example.com/result1",
                "snippet": "This is a snippet from result 1",
            },
            {
                "title": "Test Result 2",
                "link": "https://example.com/result2",
                "snippet": "This is a snippet from result 2",
            },
        ]
    }


@pytest.fixture
def mock_requests(mocker, mock_scale_response):
    """Mock requests library."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_scale_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_basic_search(mock_requests):
    """Test basic search functionality."""
    # Setup
    scale_connection = ScaleSerp(api_key="test_key")
    search_tool = ScaleSerpTool(connection=scale_connection, model_config=ConfigDict())

    # Execute
    input_data = {"query": "test query", "limit": 5}
    result = search_tool.run(input_data)

    # Verify API call
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args[1]["params"]
    assert call_args["q"] == "test query"
    assert call_args["num"] == 5

    # Verify response formatting
    assert "result" in result.output["content"]
    assert "Test Result 1" in str(result.output["content"])
    assert "Test Result 2" in str(result.output["content"])


def test_search_with_custom_params(mock_requests):
    """Test search with custom parameters."""
    # Setup
    scale_connection = ScaleSerp(api_key="test_key")
    search_tool = ScaleSerpTool(connection=scale_connection, model_config=ConfigDict(), is_optimized_for_agents=True)

    # Execute
    input_data = {"query": "test query", "search_type": SearchType.NEWS, "limit": 2}
    result = search_tool.run(input_data)

    # Verify API call
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args[1]["params"]
    assert call_args["q"] == "test query"
    assert call_args["num"] == 2
    assert call_args["search_type"] == "news"

    # Verify response contains agent-optimized format
    content = result.output["content"]
    assert "<Sources with URLs>" in content
    assert "<Search results>" in content
