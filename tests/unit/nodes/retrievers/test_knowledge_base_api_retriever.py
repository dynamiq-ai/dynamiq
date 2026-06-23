from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.connections import Dynamiq
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.retrievers.knowledge_base import (
    DynamiqKnowledgebaseVectorStoreRetriever,
    DynamiqKnowledgebaseVectorStoreRetrieverInputSchema,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


@pytest.fixture
def connection():
    return Dynamiq(url="https://api.example.ai/", api_key="secret-token")


@pytest.fixture
def retriever(connection):
    node = DynamiqKnowledgebaseVectorStoreRetriever(connection=connection, knowledge_base_id="kb-123", top_k=5)
    node.client = MagicMock()  # bypass  real connection initialization
    return node


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    response.text = str(payload)
    return response


def test_type_resolves_to_module_path(retriever):
    assert retriever.type == "dynamiq.nodes.retrievers.DynamiqKnowledgebaseVectorStoreRetriever"


def test_query_is_required():
    with pytest.raises(ValidationError):
        DynamiqKnowledgebaseVectorStoreRetrieverInputSchema()


def test_build_url(retriever):
    assert retriever._build_url() == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"


def test_execute_builds_request_and_parses_documents(retriever):
    payload = {
        "content": "FORMATTED CONTENT",
        "documents": [
            {"id": "1", "content": "Doc one", "metadata": {"title": "A"}, "score": 0.9},
            {"id": "2", "content": "Doc two", "metadata": {"title": "B"}, "score": 0.7},
        ],
    }
    retriever.client.request.return_value = _mock_response(payload)

    input_data = DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="hello", filters={"k": "v"}, top_k=3)
    result = retriever.execute(input_data, RunnableConfig(callbacks=[]))

    # Request built correctly
    _, kwargs = retriever.client.request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"
    assert kwargs["headers"] == {"Authorization": "Bearer secret-token"}
    assert kwargs["json"] == {"query": "hello", "top_k": 3, "filters": {"k": "v"}}

    # Response forwarded as retriever-shaped output
    assert result["content"] == "FORMATTED CONTENT"
    assert all(isinstance(doc, Document) for doc in result["documents"])
    assert [doc.content for doc in result["documents"]] == ["Doc one", "Doc two"]
    assert result["documents"][0].score == 0.9


def test_execute_uses_node_defaults_for_top_k(retriever):
    retriever.client.request.return_value = _mock_response({"documents": []})

    retriever.execute(DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="q"), RunnableConfig(callbacks=[]))

    _, kwargs = retriever.client.request.call_args
    assert kwargs["json"] == {"query": "q", "top_k": 5}  # node-level top_k default, no empty filters


def test_execute_content_falls_back_to_joined_documents(retriever):
    payload = {"documents": [{"content": "alpha"}, {"content": "beta"}]}
    retriever.client.request.return_value = _mock_response(payload)

    result = retriever.execute(
        DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert result["content"] == "alpha\n\nbeta"


def test_execute_handles_null_documents(retriever):
    retriever.client.request.return_value = _mock_response({"documents": None, "content": "no hits"})

    result = retriever.execute(
        DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert result["documents"] == []
    assert result["content"] == "no hits"


def test_execute_accepts_bare_document_list(retriever):
    retriever.client.request.return_value = _mock_response([{"content": "only"}])

    result = retriever.execute(
        DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="q"), RunnableConfig(callbacks=[])
    )
    assert [doc.content for doc in result["documents"]] == ["only"]


def test_execute_raises_on_error_status(retriever):
    retriever.client.request.return_value = _mock_response({"error": "forbidden"}, status_code=403)

    with pytest.raises(ToolExecutionException):
        retriever.execute(DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="q"), RunnableConfig(callbacks=[]))


@pytest.mark.asyncio
async def test_execute_async_builds_request_and_parses_documents(retriever):
    payload = {"content": "ASYNC CONTENT", "documents": [{"id": "1", "content": "Doc"}]}
    async_client = MagicMock()
    async_client.request = AsyncMock(return_value=_mock_response(payload))

    with patch.object(
        DynamiqKnowledgebaseVectorStoreRetriever, "get_async_client", AsyncMock(return_value=async_client)
    ):
        result = await retriever.execute_async(
            DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(query="hi"), RunnableConfig(callbacks=[])
        )

    _, kwargs = async_client.request.call_args
    assert kwargs["url"] == "https://api.example.ai/v1/knowledgebases/kb-123/vector-search"
    assert result["content"] == "ASYNC CONTENT"
    assert result["documents"][0].content == "Doc"
