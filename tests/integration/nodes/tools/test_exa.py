import pytest
from pydantic import ConfigDict

from dynamiq.connections import Exa
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.runnables import RunnableResult, RunnableStatus


@pytest.fixture
def mock_exa_response(mocker):
    """Mock response from Exa API."""
    return {
        "results": [
            {
                "title": "Test Article 1",
                "url": "https://example.com/article1",
                "publishedDate": "2024-01-01",
                "author": "John Doe",
                "score": 0.95,
                "highlights": ["This is a highlight from article 1"],
                "summary": "Summary of article 1",
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/article2",
                "publishedDate": "2024-01-02",
                "author": "Jane Smith",
                "score": 0.85,
                "highlights": ["This is a highlight from article 2"],
                "summary": "Summary of article 2",
            },
        ],
        "autopromptString": "Enhanced search query",
    }


@pytest.fixture
def mock_requests(mocker, mock_exa_response):
    """Mock requests library."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_exa_response
    mock_response.raise_for_status.return_value = None

    mock_requests = mocker.patch("requests.request", return_value=mock_response)
    return mock_requests


def test_exa_basic_search(mock_requests, mock_exa_response):
    """Test basic search functionality without content retrieval."""
    exa_connection = Exa(api_key="test_key")
    exa_tool = ExaTool(connection=exa_connection, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence", "limit": 2, "query_type": "neural", "include_full_content": False}

    result = exa_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input.model_dump()
    assert input_dump["query"] == input_data["query"]
    assert input_dump["limit"] == input_data["limit"]
    assert input_dump["query_type"] == input_data["query_type"]
    assert input_dump["include_full_content"] == input_data["include_full_content"]

    assert input_dump["use_autoprompt"] is False
    assert input_dump["category"] is None
    assert input_dump["include_domains"] is None
    assert input_dump["exclude_domains"] is None
    assert input_dump["include_text"] is None
    assert input_dump["exclude_text"] is None

    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["query"] == "artificial intelligence"
    assert call_args[1]["json"]["numResults"] == 2
    assert "contents" not in call_args[1]["json"]


def test_exa_search_agent_optimized(mock_requests, mock_exa_response):
    """Test search with agent-optimized output format."""
    exa_connection = Exa(api_key="test_key")
    exa_tool = ExaTool(connection=exa_connection, is_optimized_for_agents=True, model_config=ConfigDict())

    input_data = {"query": "artificial intelligence", "include_full_content": True}

    result = exa_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "<Sources with URLs>" in content
    assert "<Search results>" in content
    assert all(f"{r['title']}: ({r['url']})" in content for r in mock_exa_response["results"])

    for result in mock_exa_response["results"]:
        if "highlights" in result:
            assert any(highlight in content for highlight in result["highlights"])
        if "summary" in result:
            assert result["summary"] in content
