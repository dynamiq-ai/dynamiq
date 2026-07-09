from typing import ClassVar, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from dynamiq import Workflow
from dynamiq.connections import Cohere, Dynamiq
from dynamiq.flows import Flow
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.knowledgebases.knowledgebase_graph import DynamiqKnowledgebaseGraphSearch
from dynamiq.nodes.knowledgebases.knowledgebase_hybrid import (
    DynamiqKnowledgebaseHybridSearch,
    DynamiqKnowledgebaseHybridSearchInputSchema,
)
from dynamiq.nodes.knowledgebases.knowledgebase_vector import DynamiqKnowledgebaseVectorSearch
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.nodes.rankers import CohereReranker
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


@pytest.fixture
def connection():
    return Dynamiq(url="https://api.example.ai/", api_key="secret-token")


def _mock_client():
    # ``.run()`` calls ensure_client(), which rebuilds the client when is_client_closed() is truthy.
    # A bare MagicMock's ``.closed`` is truthy, so pin it False to keep our mocked client in place.
    client = MagicMock()
    client.closed = False
    return client


@pytest.fixture
def vector_node(connection):
    node = DynamiqKnowledgebaseVectorSearch(connection=connection, knowledgebase_id="kb-123")
    node.client = _mock_client()
    return node


@pytest.fixture
def graph_node(connection):
    node = DynamiqKnowledgebaseGraphSearch(connection=connection, knowledgebase_id="kb-123")
    node.client = _mock_client()
    return node


class _StubRerankerInput(BaseModel):
    query: str
    documents: list[Document] = Field(default_factory=list)


class _StubReranker(Node):
    """Reranker that reverses the document order so tests can prove reranking was applied."""

    group: Literal[NodeGroup.RANKERS] = NodeGroup.RANKERS
    name: str = "stub-reranker"
    input_schema: ClassVar[type[_StubRerankerInput]] = _StubRerankerInput

    def execute(self, input_data: _StubRerankerInput, config: RunnableConfig = None, **kwargs):
        self.run_on_node_execute_run(config.callbacks if config else [], **kwargs)
        return {"documents": list(reversed(input_data.documents))}


def _vector_response(documents, status_code=200):
    # Vector API returns a flat `{"data": [...documents...]}` envelope.
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"data": documents}
    response.text = str(documents)
    return response


def _graph_response(payload, status_code=200):
    # Graph API wraps its inner object in a `{"data": {...}}` envelope.
    body = {"data": payload}
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.text = str(body)
    return response


def _hybrid(vector_node, graph_node, **kwargs):
    return DynamiqKnowledgebaseHybridSearch(
        vector_search=vector_node, graph_search=graph_node, **kwargs
    )


def test_type_resolves_to_module_path(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node)
    assert node.type == "dynamiq.nodes.knowledgebases.DynamiqKnowledgebaseHybridSearch"


def test_query_is_required():
    with pytest.raises(ValidationError):
        DynamiqKnowledgebaseHybridSearchInputSchema()


def test_execute_merges_dedupes_and_appends_graph_facts(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node)  # no reranker -> all merged docs returned
    vector_node.client.request.return_value = _vector_response(
        [
            {"id": "1", "content": "alpha", "metadata": {"title": "A"}, "score": 0.9},
            {"id": "2", "content": "beta"},
        ]
    )
    # Realistic graph shape when source docs are present: `content` is the source-doc text concatenated
    # (a duplicate of `source_documents`), while `facts` holds the unique relationship triples.
    graph_node.client.request.return_value = _graph_response(
        {
            "content": "beta\n\ngamma",
            "facts": "Acme -[USES]-> Helios",
            "source_documents": [{"id": "2", "content": "beta"}, {"id": "3", "content": "gamma"}],
        }
    )

    result = node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    # Deduped by id (vector "2" wins the tie), graph-only "3" appended.
    assert [doc.id for doc in result["documents"]] == ["1", "2", "3"]
    # Provenance tagged on every document.
    assert [doc.metadata["retrieval_source"] for doc in result["documents"]] == ["vector", "vector", "graph"]
    # The `facts` triples are appended -- NOT the `content`, which would duplicate the source-doc text.
    assert "--- Graph Facts ---\nAcme -[USES]-> Helios" in result["content"]
    assert result["content"].count("gamma") == 1  # source-doc text appears once (from the pool), not duplicated


def test_execute_reranks_and_applies_top_k(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node, reranker=_StubReranker(), top_k=2)
    vector_node.client.request.return_value = _vector_response(
        [
            {"id": "1", "content": "a"},
            {"id": "2", "content": "b"},
            {"id": "3", "content": "c"},
        ]
    )
    graph_node.client.request.return_value = _graph_response({"content": ""})

    result = node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    # Stub reranker reverses order -> [3, 2, 1], then top_k=2 -> [3, 2].
    assert [doc.id for doc in result["documents"]] == ["3", "2"]


def test_execute_applies_top_k_without_reranker(vector_node, graph_node):
    """top_k caps the merged pool even when no reranker is configured (order preserved)."""
    node = _hybrid(vector_node, graph_node, top_k=2)  # no reranker
    vector_node.client.request.return_value = _vector_response(
        [{"id": "1", "content": "a"}, {"id": "2", "content": "b"}, {"id": "3", "content": "c"}]
    )
    graph_node.client.request.return_value = _graph_response(
        {"content": "", "source_documents": [{"id": "4", "content": "d"}]}
    )

    result = node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    # Merged pool is [1, 2, 3, 4]; top_k=2 caps to the first two in merge order.
    assert [doc.id for doc in result["documents"]] == ["1", "2"]


def test_execute_graph_facts_only(vector_node, graph_node):
    """Graph returns facts but no source docs: docs come from vector, facts appended to content."""
    node = _hybrid(vector_node, graph_node)
    vector_node.client.request.return_value = _vector_response([{"id": "1", "content": "alpha"}])
    graph_node.client.request.return_value = _graph_response(
        {"content": "only facts here", "facts": "only facts here", "source_documents": []}
    )

    result = node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    assert [doc.id for doc in result["documents"]] == ["1"]
    assert "--- Graph Facts ---\nonly facts here" in result["content"]


def test_execute_ignores_graph_content_without_facts(vector_node, graph_node):
    """A graph result carrying only `content` (no `facts`) contributes no appended text -- `content` is
    the source-doc text, which is already in the pool via `source_documents`; appending it would duplicate."""
    node = _hybrid(vector_node, graph_node)
    vector_node.client.request.return_value = _vector_response([{"id": "1", "content": "alpha"}])
    graph_node.client.request.return_value = _graph_response(
        {"content": "alpha", "source_documents": [{"id": "1", "content": "alpha"}]}
    )

    result = node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )

    assert "--- Graph Facts ---" not in result["content"]
    assert result["content"].count("alpha") == 1


def test_execute_drops_vector_only_params_from_graph_request(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node)
    vector_node.client.request.return_value = _vector_response([])
    graph_node.client.request.return_value = _graph_response({"content": ""})

    node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q", limit=4, similarity_threshold=0.5, alpha=0.3),
        RunnableConfig(callbacks=[]),
    )

    _, vector_kwargs = vector_node.client.request.call_args
    _, graph_kwargs = graph_node.client.request.call_args
    # Vector receives the vector-only params...
    assert vector_kwargs["json"]["alpha"] == 0.3
    assert vector_kwargs["json"]["similarity_threshold"] == 0.5
    # ...the graph request never carries alpha/similarity_threshold (its schema rejects them).
    assert "alpha" not in graph_kwargs["json"]
    assert "similarity_threshold" not in graph_kwargs["json"]
    assert graph_kwargs["json"]["limit"] == 4


def test_execute_propagates_sub_node_failure(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node)
    vector_node.client.request.return_value = _vector_response([], status_code=403)
    graph_node.client.request.return_value = _graph_response({"content": ""})

    with pytest.raises(ToolExecutionException):
        node.execute(
            DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
        )


@pytest.mark.asyncio
async def test_execute_async_fans_out_and_merges(vector_node, graph_node):
    node = _hybrid(vector_node, graph_node)

    vector_client = MagicMock()
    vector_client.request = AsyncMock(return_value=_vector_response([{"id": "1", "content": "alpha"}]))
    graph_client = MagicMock()
    graph_client.request = AsyncMock(
        return_value=_graph_response(
            {"content": "beta", "facts": "fact", "source_documents": [{"id": "2", "content": "beta"}]}
        )
    )

    with patch.object(
        DynamiqKnowledgebaseVectorSearch, "get_async_client", AsyncMock(return_value=vector_client)
    ), patch.object(DynamiqKnowledgebaseGraphSearch, "get_async_client", AsyncMock(return_value=graph_client)):
        result = await node.execute_async(
            DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
        )

    assert [doc.id for doc in result["documents"]] == ["1", "2"]
    assert "fact" in result["content"]


def test_yaml_roundtrip(tmp_path):
    dynamiq_connection = Dynamiq(id="dynamiq-conn", url="https://api.example.ai/", api_key="secret-token")
    cohere_connection = Cohere(id="cohere-conn", api_key="cohere-key")
    node = DynamiqKnowledgebaseHybridSearch(
        id="kb-hybrid",
        top_k=3,
        vector_search=DynamiqKnowledgebaseVectorSearch(
            id="kb-vector", connection=dynamiq_connection, knowledgebase_id="kb-123", limit=7
        ),
        graph_search=DynamiqKnowledgebaseGraphSearch(
            id="kb-graph", connection=dynamiq_connection, knowledgebase_id="kb-123"
        ),
        reranker=CohereReranker(id="kb-reranker", connection=cohere_connection, top_k=5),
    )
    workflow = Workflow(id="kb-workflow", flow=Flow(id="kb-flow", nodes=[node]))

    yaml_path = tmp_path / "kb_hybrid_workflow.yaml"
    workflow.to_yaml_file(yaml_path)

    loaded = Workflow.from_yaml_file(str(yaml_path), init_components=True)
    loaded_node = loaded.flow.nodes[0]

    assert isinstance(loaded_node, DynamiqKnowledgebaseHybridSearch)
    assert loaded_node.top_k == 3
    # Composed sub-nodes survive the roundtrip and are initialized by init_components.
    assert isinstance(loaded_node.vector_search, DynamiqKnowledgebaseVectorSearch)
    assert loaded_node.vector_search.knowledgebase_id == "kb-123"
    assert loaded_node.vector_search.limit == 7
    assert loaded_node.vector_search.client is not None
    assert isinstance(loaded_node.graph_search, DynamiqKnowledgebaseGraphSearch)
    assert isinstance(loaded_node.reranker, CohereReranker)
    assert loaded_node.reranker.top_k == 5
    # Connections survive the roundtrip with their fields intact (id, url, api_key), and the shared
    # Dynamiq connection resolves back to ONE instance for both sub-searches -- not two copies.
    loaded_dynamiq_connection = loaded_node.vector_search.connection
    assert loaded_dynamiq_connection.id == dynamiq_connection.id == "dynamiq-conn"
    assert loaded_dynamiq_connection.url == dynamiq_connection.url == "https://api.example.ai/"
    assert loaded_dynamiq_connection.api_key == dynamiq_connection.api_key == "secret-token"
    assert loaded_node.vector_search.connection is loaded_node.graph_search.connection
    # The reranker keeps its own distinct Cohere connection.
    assert loaded_node.reranker.connection.id == cohere_connection.id == "cohere-conn"
    assert loaded_node.reranker.connection.api_key == cohere_connection.api_key == "cohere-key"

    # The deserialized node still executes end-to-end (sub-node HTTP clients mocked, reranker unused here).
    loaded_node.reranker = None
    loaded_node.vector_search.client = _mock_client()
    loaded_node.vector_search.client.request.return_value = _vector_response([{"id": "1", "content": "hit"}])
    loaded_node.graph_search.client = _mock_client()
    loaded_node.graph_search.client.request.return_value = _graph_response({"content": "fact", "facts": "fact"})

    result = loaded_node.execute(
        DynamiqKnowledgebaseHybridSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert [doc.content for doc in result["documents"]] == ["hit"]
    assert "fact" in result["content"]
