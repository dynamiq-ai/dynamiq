from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.connections import Dynamiq
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.knowledgebases.knowledgebase_graph import (
    DynamiqKnowledgebaseGraphSearch,
    DynamiqKnowledgebaseGraphSearchInputSchema,
)
from dynamiq.runnables import RunnableConfig


@pytest.fixture
def connection():
    return Dynamiq(url="https://api.example.ai/", api_key="secret-token")


@pytest.fixture
def retriever(connection):
    node = DynamiqKnowledgebaseGraphSearch(connection=connection, knowledgebase_id="kb-123", limit=5)
    node.client = MagicMock()  # bypass real connection initialization
    return node


def _mock_response(payload, status_code=200):
    # The graph-search API wraps its result in a `data` envelope; `_parse_response` unwraps it, so `payload`
    # here is the INNER object the node returns.
    body = {"data": payload}
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.text = str(body)
    return response


def test_type_resolves_to_module_path(retriever):
    assert retriever.type == "dynamiq.nodes.knowledgebases.DynamiqKnowledgebaseGraphSearch"


def test_query_is_required():
    with pytest.raises(ValidationError):
        DynamiqKnowledgebaseGraphSearchInputSchema()


def test_build_url(retriever):
    assert retriever._build_url() == "https://api.example.ai/v1/knowledgebases/kb-123/graph-search"


def test_execute_builds_request_and_forwards_body(retriever):
    payload = {"content": "Acme -[USES]-> Helios", "documents": [{"id": "e1"}]}
    retriever.client.request.return_value = _mock_response(payload)

    input_data = DynamiqKnowledgebaseGraphSearchInputSchema(
        query="what systems does Acme use", filters={"k": "v"}, limit=3
    )
    result = retriever.execute(input_data, RunnableConfig(callbacks=[]))

    # Request built correctly (graph-search endpoint, graph-specific params)
    _, kwargs = retriever.client.request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/graph-search"
    assert kwargs["headers"] == {"Authorization": "Bearer secret-token"}
    assert kwargs["json"] == {
        "query": "what systems does Acme use",
        "limit": 3,
        "filters": {"k": "v"},
    }

    # Pass-through: the API body is forwarded verbatim (not reshaped into documents/content).
    assert result == payload


def test_execute_forwards_variable_shape(retriever):
    """Whatever keys the server returns (grounding/summarize on) are relayed untouched."""
    payload = {
        "content": "verbatim source text",
        "facts": "- Acme -[USES]-> Helios",
        "documents": [{"id": "e1"}],
        "source_documents": [{"id": "d1", "content": "source chunk"}],
        "context": "raw retrieval",
    }
    retriever.client.request.return_value = _mock_response(payload)

    result = retriever.execute(
        DynamiqKnowledgebaseGraphSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert result == payload  # every key relayed, identical


def test_execute_uses_node_defaults_for_limit(retriever):
    retriever.client.request.return_value = _mock_response({"content": ""})

    retriever.execute(DynamiqKnowledgebaseGraphSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = retriever.client.request.call_args
    assert kwargs["json"] == {"query": "q", "limit": 5}  # node-level limit default, no empty filters


def test_execute_raises_on_error_status(retriever):
    retriever.client.request.return_value = _mock_response({"error": "forbidden"}, status_code=403)

    with pytest.raises(ToolExecutionException):
        retriever.execute(DynamiqKnowledgebaseGraphSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))


def test_execute_raises_recoverable_on_invalid_json(retriever):
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("invalid json")
    retriever.client.request.return_value = response

    with pytest.raises(ToolExecutionException):
        retriever.execute(DynamiqKnowledgebaseGraphSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))


@pytest.mark.asyncio
async def test_execute_async_builds_request_and_forwards_body(retriever):
    payload = {"content": "fact", "documents": [{"id": "e1"}]}
    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=_mock_response(payload))

    with patch.object(
        DynamiqKnowledgebaseGraphSearch, "get_async_client", AsyncMock(return_value=async_client)
    ):
        result = await retriever.execute_async(
            DynamiqKnowledgebaseGraphSearchInputSchema(query="hi"), RunnableConfig(callbacks=[])
        )

    _, kwargs = async_client.request.call_args
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/graph-search"
    assert result == payload


def test_yaml_roundtrip(tmp_path):
    connection = Dynamiq(id="dynamiq-conn", url="https://api.example.ai/", api_key="secret-token")
    node = DynamiqKnowledgebaseGraphSearch(
        id="kb-graph-node",
        connection=connection,
        knowledgebase_id="kb-123",
        limit=7,
        filters={"allowed_principals": {"$intersects": ["group:a"]}},
    )
    workflow = Workflow(id="kb-workflow", flow=Flow(id="kb-flow", nodes=[node]))

    yaml_path = tmp_path / "kb_workflow.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    loaded_node = loaded.flow.nodes[0]
    assert isinstance(loaded_node, DynamiqKnowledgebaseGraphSearch)
    assert loaded_node.client is not None  # init_components built the HTTP client

    roundtrip_path = tmp_path / "kb_workflow_roundtrip.yaml"
    loaded.to_yaml_file(roundtrip_path)
    roundtrip_node = Workflow.from_yaml_file(str(roundtrip_path), init_components=True).flow.nodes[0]

    assert roundtrip_node.knowledgebase_id == "kb-123"
    assert roundtrip_node.limit == 7
    assert roundtrip_node.filters == {"allowed_principals": {"$intersects": ["group:a"]}}
    assert roundtrip_node.connection.id == "dynamiq-conn"
    assert roundtrip_node.connection.url == "https://api.example.ai/"

    # The deserialized node still executes end-to-end with the config it was loaded with.
    roundtrip_node.client = MagicMock()
    roundtrip_node.client.request.return_value = _mock_response({"content": "hit", "documents": [{"id": "e1"}]})

    result = roundtrip_node.execute(
        DynamiqKnowledgebaseGraphSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    _, kwargs = roundtrip_node.client.request.call_args
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/graph-search"
    assert kwargs["json"] == {
        "query": "q",
        "limit": 7,  # node-level limit survived the roundtrip
        "filters": {"allowed_principals": {"$intersects": ["group:a"]}},  # locked filters survived
    }
    assert result == {"content": "hit", "documents": [{"id": "e1"}]}
