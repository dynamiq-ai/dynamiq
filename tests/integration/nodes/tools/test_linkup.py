import json

import pytest
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.connections import Linkup
from dynamiq.flows import Flow
from dynamiq.nodes.tools.linkup_search import LinkupTool
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import JsonWorkflowEncoder


@pytest.fixture
def mock_linkup_response():
    """Mock searchResults response from Linkup API."""
    return {
        "results": [
            {
                "type": "text",
                "name": "Test Article 1",
                "url": "https://example.com/article1",
                "content": "Content of article 1",
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "type": "text",
                "name": "Test Article 2",
                "url": "https://example.com/article2",
                "content": "Content of article 2",
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "type": "image",
                "name": "Test Image",
                "url": "https://example.com/image.png",
            },
        ]
    }


@pytest.fixture
def mock_linkup_answer_response():
    """Mock sourcedAnswer response from Linkup API."""
    return {
        "answer": "Artificial intelligence is the simulation of human intelligence by machines.",
        "sources": [
            {
                "name": "Test Article 1",
                "url": "https://example.com/article1",
                "snippet": "Snippet of article 1",
            },
        ],
    }


def _mock_requests(mocker, payload):
    mock_response = mocker.Mock()
    mock_response.json.return_value = payload
    mock_response.raise_for_status.return_value = None
    return mocker.patch("requests.request", return_value=mock_response)


@pytest.fixture
def mock_requests(mocker, mock_linkup_response):
    """Mock requests library."""
    return _mock_requests(mocker, mock_linkup_response)


def test_linkup_node_parameters(mock_requests):
    """Test LinkupTool initialization with node parameters."""
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(
        connection=linkup_connection,
        depth="deep",
        max_results=5,
        include_domains=["example.com"],
    )

    assert linkup_tool.depth == "deep"
    assert linkup_tool.max_results == 5
    assert linkup_tool.include_domains == ["example.com"]

    input_data = {"query": "artificial intelligence"}
    result = linkup_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["url"] == "https://api.linkup.so/v1/search"
    assert call_args[1]["headers"]["Authorization"] == "Bearer test_key"
    assert call_args[1]["json"]["q"] == "artificial intelligence"
    assert call_args[1]["json"]["depth"] == "deep"
    assert call_args[1]["json"]["outputType"] == "searchResults"
    assert call_args[1]["json"]["maxResults"] == 5
    assert call_args[1]["json"]["includeDomains"] == ["example.com"]
    assert "excludeDomains" not in call_args[1]["json"]
    assert "includeInlineCitations" not in call_args[1]["json"]


def test_linkup_parameter_override(mock_requests):
    """Test overriding node parameters during execution."""
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(connection=linkup_connection, depth="deep", max_results=5)

    input_data = {
        "query": "artificial intelligence",
        "depth": "fast",
        "max_results": 2,
        "exclude_domains": ["example.org"],
        "from_date": "2026-01-01",
        "to_date": "2026-01-31",
    }

    result = linkup_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["depth"] == "fast"  # Overridden from deep
    assert call_args[1]["json"]["maxResults"] == 2  # Overridden from 5
    assert call_args[1]["json"]["excludeDomains"] == ["example.org"]
    assert call_args[1]["json"]["fromDate"] == "2026-01-01"
    assert call_args[1]["json"]["toDate"] == "2026-01-31"


def test_linkup_basic_search(mock_requests, mock_linkup_response):
    """Test basic search functionality with raw output."""
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(connection=linkup_connection)

    input_data = {"query": "artificial intelligence", "max_results": 2}

    result = linkup_tool.run(input_data, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    input_dump = result.input
    assert input_dump["query"] == input_data["query"]
    assert input_dump["max_results"] == input_data["max_results"]

    content = result.output["content"]
    assert content["urls"] == ["https://example.com/article1", "https://example.com/article2"]
    assert content["raw_response"] == mock_linkup_response
    assert content["answer"] is None
    assert "Content of article 1" in content["result"]
    assert "Test Image" not in content["result"]


def test_linkup_search_agent_optimized(mock_requests, mock_linkup_response):
    """Test search with agent-optimized output format."""
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(connection=linkup_connection, is_optimized_for_agents=True)

    result = linkup_tool.run({"query": "artificial intelligence"}, None)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS

    content = result.output["content"]
    assert "## Sources" in content
    assert "## Search Results" in content

    text_results = [r for r in mock_linkup_response["results"] if r["type"] == "text"]
    for idx, result_data in enumerate(text_results, start=1):
        assert f"- [{result_data['name']}]({result_data['url']})" in content
        assert f"### Result {idx}: {result_data['name']}" in content
        assert result_data["content"] in content


def test_linkup_sourced_answer(mocker, mock_linkup_answer_response):
    """Test sourcedAnswer output type."""
    mock_requests = _mock_requests(mocker, mock_linkup_answer_response)
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(
        connection=linkup_connection,
        output_type="sourcedAnswer",
        include_inline_citations=True,
        is_optimized_for_agents=True,
    )

    result = linkup_tool.run({"query": "what is artificial intelligence"}, None)

    assert result.status == RunnableStatus.SUCCESS
    call_args = mock_requests.call_args
    assert call_args[1]["json"]["outputType"] == "sourcedAnswer"
    assert call_args[1]["json"]["includeInlineCitations"] is True

    content = result.output["content"]
    assert "## Answer" in content
    assert mock_linkup_answer_response["answer"] in content
    assert "- [Test Article 1](https://example.com/article1)" in content
    assert "Snippet of article 1" in content
    assert result.output["urls"] == ["https://example.com/article1"]


def test_linkup_request_failure(mocker):
    """Test that HTTP errors are surfaced as a failed run."""
    mocker.patch("requests.request", side_effect=RuntimeError("401 Unauthorized"))
    linkup_tool = LinkupTool(connection=Linkup(api_key="test_key"))

    result = linkup_tool.run({"query": "artificial intelligence"}, None)

    assert result.status == RunnableStatus.FAILURE
    assert "401 Unauthorized" in result.error.message


def test_linkup_with_invalid_input_schema(mock_requests):
    """Test behavior with invalid input schema."""
    linkup_connection = Linkup(api_key="test_key")
    linkup_tool = LinkupTool(connection=linkup_connection)

    wf = Workflow(flow=Flow(nodes=[linkup_tool]))
    input_data = {}
    tracing = TracingCallbackHandler()
    result = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    result_linkup = result.output[linkup_tool.id]
    assert result.status == RunnableStatus.FAILURE
    assert result.input == input_data
    assert result_linkup["status"] == RunnableStatus.FAILURE.value
    assert result_linkup["input"] == input_data
    assert result_linkup["output"] is None
    assert result_linkup["error"]["type"] == ValidationError.__name__

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 3
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.output is None
    assert wf_run.status == RunStatus.FAILED
    assert "failed_nodes" in wf_run.metadata
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output is None
    assert flow_run.status == RunStatus.FAILED
    assert "failed_nodes" in flow_run.metadata
    linkup_tool_run = tracing_runs[2]
    assert linkup_tool_run.metadata["node"]["id"] == linkup_tool.id
    assert linkup_tool_run.parent_run_id == flow_run.id
    assert linkup_tool_run.input == input_data
    assert linkup_tool_run.output is None
    assert linkup_tool_run.error
    assert linkup_tool_run.status == RunStatus.FAILED
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)
