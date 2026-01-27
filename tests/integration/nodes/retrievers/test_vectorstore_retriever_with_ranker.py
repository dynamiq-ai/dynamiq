import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.rankers import CohereReranker, LLMDocumentRanker
from dynamiq.nodes.retrievers import WeaviateDocumentRetriever
from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_embedder_response():
    """Mock embedder response with a sample embedding."""
    return {"embedding": [0.1, 0.2, 0.3], "query": "test query"}


@pytest.fixture
def mock_retriever_documents():
    """Mock documents returned by the retriever."""
    return [
        Document(content="Machine learning is a branch of AI", score=0.9),
        Document(content="Deep learning uses neural networks", score=0.85),
        Document(content="Python is a programming language", score=0.7),
        Document(content="Data science involves statistics", score=0.65),
        Document(content="AI has many applications", score=0.6),
    ]


@pytest.fixture
def mock_rerank_response():
    """Mock reranker response."""
    return Mock(
        results=[
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.88},
            {"index": 2, "relevance_score": 0.75},
        ]
    )


@pytest.fixture
def mock_rerank_executor(mock_rerank_response):
    """Mock the litellm rerank function."""
    with patch("litellm.rerank") as mock:
        mock.return_value = mock_rerank_response
        yield mock


@pytest.fixture
def mock_weaviate_vector_store():
    """Mock WeaviateVectorStore to avoid real API calls."""
    mock_store = MagicMock(spec=WeaviateVectorStore)
    mock_store.client = MagicMock()
    return mock_store


@patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.run")
@patch("dynamiq.nodes.embedders.openai.OpenAITextEmbedder.run")
def test_vectorstore_retriever_with_cohere_ranker(
    mock_embedder_run,
    mock_retriever_run,
    mock_embedder_response,
    mock_retriever_documents,
    mock_rerank_executor,
    mock_weaviate_vector_store,
):
    """Test VectorStoreRetriever with CohereReranker."""
    mock_embedder_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=mock_embedder_response,
    )
    mock_retriever_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"documents": mock_retriever_documents},
    )

    openai_connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )
    cohere_connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )

    text_embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )

    document_retriever = WeaviateDocumentRetriever(
        vector_store=mock_weaviate_vector_store,
        index_name="test-index",
        top_k=5,
    )

    ranker = CohereReranker(
        connection=cohere_connection,
        model="cohere/rerank-v3.5",
        top_k=3,
        threshold=0.7,
    )

    retriever = VectorStoreRetriever(
        name="Test Retriever with Ranker",
        text_embedder=text_embedder,
        document_retriever=document_retriever,
        document_reranker=ranker,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[retriever]),
    )

    input_data = {"query": "What is machine learning?"}
    result = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert result.status == RunnableStatus.SUCCESS
    output = result.output[retriever.id]["output"]
    assert "documents" in output
    assert "content" in output

    reranked_docs = output["documents"]
    assert len(reranked_docs) <= 3
    for doc in reranked_docs:
        score = doc.score if hasattr(doc, "score") else doc.get("score", 0)
        assert score >= 0.7

    mock_rerank_executor.assert_called_once()
    call_kwargs = mock_rerank_executor.call_args[1]
    assert call_kwargs["model"] == "cohere/rerank-v3.5"
    assert call_kwargs["query"] == "What is machine learning?"
    assert call_kwargs["top_n"] == 3


@patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.run")
@patch("dynamiq.nodes.embedders.openai.OpenAITextEmbedder.run")
def test_vectorstore_retriever_without_ranker(
    mock_embedder_run,
    mock_retriever_run,
    mock_embedder_response,
    mock_retriever_documents,
    mock_weaviate_vector_store,
):
    """Test VectorStoreRetriever without ranker (standard behavior)."""
    mock_embedder_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=mock_embedder_response,
    )
    mock_retriever_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"documents": mock_retriever_documents},
    )

    openai_connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )

    text_embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )

    document_retriever = WeaviateDocumentRetriever(
        vector_store=mock_weaviate_vector_store,
        index_name="test-index",
        top_k=5,
    )

    retriever = VectorStoreRetriever(
        name="Test Retriever without Ranker",
        text_embedder=text_embedder,
        document_retriever=document_retriever,
        document_reranker=None,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[retriever]),
    )

    input_data = {"query": "What is machine learning?"}
    result = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert result.status == RunnableStatus.SUCCESS
    output = result.output[retriever.id]["output"]
    assert "documents" in output
    assert "content" in output

    retrieved_docs = output["documents"]
    assert len(retrieved_docs) == len(mock_retriever_documents)


@patch("dynamiq.nodes.rankers.llm.LLMDocumentRanker.run")
@patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.run")
@patch("dynamiq.nodes.embedders.openai.OpenAITextEmbedder.run")
def test_vectorstore_retriever_with_llm_ranker(
    mock_embedder_run,
    mock_retriever_run,
    mock_ranker_run,
    mock_embedder_response,
    mock_retriever_documents,
    mock_weaviate_vector_store,
):
    """Test VectorStoreRetriever with LLMDocumentRanker."""
    from dynamiq.nodes.llms import OpenAI

    mock_embedder_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=mock_embedder_response,
    )
    mock_retriever_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"documents": mock_retriever_documents},
    )
    mock_ranker_run.return_value = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={
            "documents": [
                Document(content="Machine learning is a branch of AI", score=0.95),
                Document(content="Deep learning uses neural networks", score=0.90),
                Document(content="Data science involves statistics", score=0.85),
            ]
        },
    )

    openai_connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )

    text_embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )

    document_retriever = WeaviateDocumentRetriever(
        vector_store=mock_weaviate_vector_store,
        index_name="test-index",
        top_k=5,
    )

    llm = OpenAI(
        connection=openai_connection,
        model="gpt-4o-mini",
    )

    ranker = LLMDocumentRanker(
        llm=llm,
        top_k=3,
    )

    retriever = VectorStoreRetriever(
        name="Test Retriever with LLM Ranker",
        text_embedder=text_embedder,
        document_retriever=document_retriever,
        document_reranker=ranker,
    )

    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(nodes=[retriever]),
    )

    input_data = {"query": "What is machine learning?"}
    result = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert result.status == RunnableStatus.SUCCESS
    output = result.output[retriever.id]["output"]
    assert "documents" in output

    reranked_docs = output["documents"]
    assert len(reranked_docs) <= 3


def test_vectorstore_retriever_ranker_serialization(mock_weaviate_vector_store):
    """Test that VectorStoreRetriever with ranker can be serialized to dict."""
    openai_connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )
    cohere_connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )

    text_embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )

    document_retriever = WeaviateDocumentRetriever(
        vector_store=mock_weaviate_vector_store,
        index_name="test-index",
    )

    ranker = CohereReranker(
        connection=cohere_connection,
        model="cohere/rerank-v3.5",
        top_k=5,
    )

    retriever = VectorStoreRetriever(
        text_embedder=text_embedder,
        document_retriever=document_retriever,
        document_reranker=ranker,
    )

    retriever_dict = retriever.to_dict()

    assert "text_embedder" in retriever_dict
    assert "document_retriever" in retriever_dict
    assert "document_reranker" in retriever_dict
    assert retriever_dict["document_reranker"]["type"] == "dynamiq.nodes.rankers.CohereReranker"
    assert retriever_dict["document_reranker"]["top_k"] == 5


def test_vectorstore_retriever_ranker_init_components(mock_weaviate_vector_store):
    """Test that init_components properly initializes the ranker."""
    from dynamiq.connections.managers import ConnectionManager

    openai_connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )
    cohere_connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="test_api_key",
    )

    text_embedder = OpenAITextEmbedder(
        connection=openai_connection,
        model="text-embedding-3-small",
    )

    document_retriever = WeaviateDocumentRetriever(
        vector_store=mock_weaviate_vector_store,
        index_name="test-index",
    )

    ranker = CohereReranker(
        connection=cohere_connection,
        model="cohere/rerank-v3.5",
        top_k=5,
    )

    retriever = VectorStoreRetriever(
        text_embedder=text_embedder,
        document_retriever=document_retriever,
        document_reranker=ranker,
    )

    cm = ConnectionManager()
    retriever.init_components(cm)

    assert retriever.document_reranker is not None
    assert retriever.document_reranker == ranker
