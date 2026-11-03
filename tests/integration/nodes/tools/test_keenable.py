import pytest

from dynamiq.connections import Keenable
from dynamiq.nodes.tools.keenable_search import KeenableTool
from dynamiq.runnables import RunnableStatus


@pytest.fixture
def mock_keenable_response():
    return {
        "query": "ai",
        "results": [
            {"title": "A", "url": "https://example.com/a", "description": "", "snippet": "x" * 50},
            {"title": "B", "url": "https://example.com/b", "description": "Desc B", "snippet": "long"},
            {"title": "no url", "url": ""},
        ],
    }


@pytest.fixture
def mock_requests(mocker, mock_keenable_response):
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_keenable_response
    mock_response.raise_for_status.return_value = None
    return mocker.patch("requests.request", return_value=mock_response)


def test_keenable_keyless_uses_public_endpoint(mock_requests):
    tool = KeenableTool(connection=Keenable(api_key=""), max_content_chars=10, is_optimized_for_agents=True)
    result = tool.run(input_data={"query": "ai", "max_results": 3})

    assert result.status == RunnableStatus.SUCCESS
    kwargs = mock_requests.call_args.kwargs
    assert kwargs["url"] == "https://api.keenable.ai/v1/search/public"
    assert kwargs["json"] == {"query": "ai", "max_results": 3}
    assert kwargs["headers"]["X-Keenable-Title"] == "Dynamiq"
    assert "X-API-Key" not in kwargs["headers"]

    sources = result.output["sources"]
    assert [s["url"] for s in sources] == ["https://example.com/a", "https://example.com/b"]
    assert sources[0]["content"] == "x" * 10 + "..."
    assert sources[1]["content"] == "Desc B"


def test_keenable_keyed_uses_keyed_endpoint(mock_requests):
    tool = KeenableTool(connection=Keenable(api_key="k-1"))
    tool.run(input_data={"query": "ai"})

    kwargs = mock_requests.call_args.kwargs
    assert kwargs["url"] == "https://api.keenable.ai/v1/search"
    assert kwargs["headers"]["X-API-Key"] == "k-1"
    assert kwargs["json"]["max_results"] == 5
