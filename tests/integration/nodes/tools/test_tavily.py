import pytest

from dynamiq.connections import Tavily
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_tavily_response(mocker):
    """Mock response from Tavily API."""
    return {
        "results": [
            {
                "title": "Quantum Computing Article 1",
                "url": "https://example.com/quantum1",
                "content": "Content from article 1",
                "score": 0.95,
                "raw_content": "Full content from article 1",
            },
            {
                "title": "Quantum Computing Article 2",
                "url": "https://example.com/quantum2",
                "content": "Content from article 2",
                "score": 0.85,
                "raw_content": "Full content from article 2",
            },
        ],
        "answer": "This is a summarized answer about quantum computing.",
        "query": "Latest developments in quantum computing",
        "response_time": 1.5,
    }


@pytest.fixture
def mock_requests(mocker, mock_tavily_response):
    """Mock requests library."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_tavily_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_tavily_node_parameters(mock_requests):
    """Test TavilyTool initialization with node parameters and basic search."""
    tavily_connection = Tavily(api_key="test_key")
    tavily_tool = TavilyTool(
        connection=tavily_connection,
        search_depth="advanced",
        max_results=3,
        include_answer=True,
        include_domains=["example.com"],
        exclude_domains=["wikipedia.org"],
    )

    # Verify node parameters were set correctly
    assert tavily_tool.search_depth == "advanced"
    assert tavily_tool.max_results == 3
    assert tavily_tool.include_answer is True
    assert tavily_tool.include_domains == ["example.com"]
    assert tavily_tool.exclude_domains == ["wikipedia.org"]

    # Test with basic query
    input_data = {"query": "Latest developments in quantum computing"}
    result = tavily_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    # Verify node parameters were used in the API call
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["query"] == "Latest developments in quantum computing"
    assert call_args[1]["json"]["search_depth"] == "advanced"
    assert call_args[1]["json"]["max_results"] == 3
    assert call_args[1]["json"]["include_answer"] is True
    assert call_args[1]["json"]["include_domains"] == ["example.com"]
    assert call_args[1]["json"]["exclude_domains"] == ["wikipedia.org"]


def test_tavily_parameter_override(mock_requests):
    """Test overriding node parameters during execution."""
    tavily_connection = Tavily(api_key="test_key")
    tavily_tool = TavilyTool(
        connection=tavily_connection,
        search_depth="basic",
        max_results=5,
        include_answer=False,
        include_domains=[],
        exclude_domains=[],
    )

    # Override parameters in input
    input_data = {
        "query": "Latest developments in quantum computing",
        "search_depth": "advanced",
        "max_results": 3,
        "include_answer": True,
        "exclude_domains": ["wikipedia.org"],
    }

    result = tavily_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    # Verify input parameters override node parameters
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["query"] == "Latest developments in quantum computing"
    assert call_args[1]["json"]["search_depth"] == "advanced"
    assert call_args[1]["json"]["max_results"] == 3
    assert call_args[1]["json"]["include_answer"] is True
    assert call_args[1]["json"]["exclude_domains"] == ["wikipedia.org"]

    # Check that the response is properly formatted
    output = result.output["content"]
    if not tavily_tool.is_optimized_for_agents:
        assert isinstance(output, dict)
        assert "result" in output
        assert "sources_with_url" in output
        assert "raw_response" in output
    else:
        assert isinstance(output, str)
        assert "<Sources with URLs>" in output
        assert "<Search results for query" in output
