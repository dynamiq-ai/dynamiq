from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.connections import Dynamiq
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.knowledgebases.knowledgebase import (
    DynamiqKnowledgebaseVectorSearch,
    DynamiqKnowledgebaseVectorSearchInputSchema,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


@pytest.fixture
def connection():
    return Dynamiq(url="https://api.example.ai/", api_key="secret-token")


@pytest.fixture
def retriever(connection):
    node = DynamiqKnowledgebaseVectorSearch(connection=connection, knowledgebase_id="kb-123", limit=5)
    node.client = MagicMock()  # bypass  real connection initialization
    return node


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = str(payload)
    return response


def test_type_resolves_to_module_path(retriever):
    assert retriever.type == "dynamiq.nodes.knowledgebases.DynamiqKnowledgebaseVectorSearch"


def test_query_is_required():
    with pytest.raises(ValidationError):
        DynamiqKnowledgebaseVectorSearchInputSchema()


def test_build_url(retriever):
    assert retriever._build_url() == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"


def test_execute_builds_request_and_parses_documents(retriever):
    payload = {
        "data": [
            # `embedding` is an internal field and must be filtered out of the formatted content.
            {"id": "1", "content": "Doc one", "metadata": {"title": "A", "embedding": [0.1, 0.2]}, "score": 0.9},
            {"id": "2", "content": "Doc two", "metadata": {"title": "B"}, "score": 0.7},
        ],
    }
    retriever.client.request.return_value = _mock_response(payload)

    input_data = DynamiqKnowledgebaseVectorSearchInputSchema(query="hello", filters={"k": "v"}, limit=3)
    result = retriever.execute(input_data, RunnableConfig(callbacks=[]))

    # Request built correctly
    _, kwargs = retriever.client.request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"
    assert kwargs["headers"] == {"Authorization": "Bearer secret-token"}
    assert kwargs["json"] == {"query": "hello", "limit": 3, "filters": {"k": "v"}}

    # Response forwarded as retriever-shaped output
    assert all(isinstance(doc, Document) for doc in result["documents"])
    assert [doc.content for doc in result["documents"]] == ["Doc one", "Doc two"]
    assert result["documents"][0].score == 0.9
    # Content is formatted into numbered, source-labelled blocks
    assert result["content"] == (
        "--- Retrieved Source 1 ---\n"
        "Metadata:\nScore: 0.9\ntitle: A\n\n"
        "Content:\nDoc one\n"
        "--- End Source 1 ---\n\n"
        "--- Retrieved Source 2 ---\n"
        "Metadata:\nScore: 0.7\ntitle: B\n\n"
        "Content:\nDoc two\n"
        "--- End Source 2 ---"
    )


def test_metadata_fields_none_emits_all_metadata(retriever):
    retriever.metadata_fields = None
    payload = {"data": [{"content": "doc", "metadata": {"title": "A", "embedding": [0.1]}, "score": 0.5}]}
    retriever.client.request.return_value = _mock_response(payload)

    result = retriever.execute(DynamiqKnowledgebaseVectorSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    assert "embedding: [0.1]" in result["content"]
    assert "title: A" in result["content"]


def test_execute_uses_node_defaults_for_limit(retriever):
    retriever.client.request.return_value = _mock_response({"data": []})

    retriever.execute(DynamiqKnowledgebaseVectorSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = retriever.client.request.call_args
    assert kwargs["json"] == {"query": "q", "limit": 5}  # node-level limit default, no empty filters


def test_execute_handles_empty_data(retriever):
    retriever.client.request.return_value = _mock_response({"data": []})

    result = retriever.execute(
        DynamiqKnowledgebaseVectorSearchInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert result["documents"] == []
    assert result["content"] == ""


def test_execute_raises_on_error_status(retriever):
    retriever.client.request.return_value = _mock_response({"error": "forbidden"}, status_code=403)

    with pytest.raises(ToolExecutionException):
        retriever.execute(DynamiqKnowledgebaseVectorSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))


def test_execute_raises_recoverable_on_invalid_json(retriever):
    response = MagicMock()
    response.status_code = 200
    response.json.side_effect = ValueError("invalid json")
    retriever.client.request.return_value = response

    with pytest.raises(ToolExecutionException):
        retriever.execute(DynamiqKnowledgebaseVectorSearchInputSchema(query="q"), RunnableConfig(callbacks=[]))


@pytest.mark.asyncio
async def test_execute_async_builds_request_and_parses_documents(retriever):
    payload = {"data": [{"id": "1", "content": "Doc"}]}
    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=_mock_response(payload))

    with patch.object(
        DynamiqKnowledgebaseVectorSearch, "get_async_client", AsyncMock(return_value=async_client)
    ):
        result = await retriever.execute_async(
            DynamiqKnowledgebaseVectorSearchInputSchema(query="hi"), RunnableConfig(callbacks=[])
        )

    _, kwargs = async_client.request.call_args
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"
    assert result["documents"][0].content == "Doc"
