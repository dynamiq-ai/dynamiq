from unittest.mock import MagicMock

import pytest

from dynamiq.components.retrievers.elasticsearch import ElasticsearchDocumentRetriever
from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=ElasticsearchVectorStore)
    store.search_by_vector.return_value = [Document(id="1", content="test content", metadata={}, score=0.9)]
    return store


@pytest.fixture
def retriever(mock_vector_store):
    return ElasticsearchDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)


def test_initialization(mock_vector_store):
    """Test component initialization with parameters."""
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_vector_store, filters={"field": "value"}, top_k=5)
    assert retriever.vector_store == mock_vector_store
    assert retriever.filters == {"field": "value"}
    assert retriever.top_k == 5


def test_initialization_with_defaults(mock_vector_store):
    """Test component initialization with default values."""
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_vector_store)
    assert retriever.vector_store == mock_vector_store
    assert retriever.filters == {}
    assert retriever.top_k == 10


def test_run_basic_search(retriever, mock_vector_store):
    """Test basic vector similarity search."""
    query = [0.1] * 768
    result = retriever.run(query=query, filters={"field": "value"}, top_k=5)

    mock_vector_store.search_by_vector.assert_called_once_with(
        query_embedding=query,
        filters={"field": "value"},
        top_k=5,
        exclude_document_embeddings=True,
        scale_scores=False,
        score_threshold=None,
    )
    assert "documents" in result
    assert len(result["documents"]) == 1


def test_run_with_score_scaling(retriever, mock_vector_store):
    """Test vector search with score scaling enabled."""
    query = [0.1] * 768
    result = retriever.run(query=query, scale_scores=True)

    mock_vector_store.search_by_vector.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=True,
        scale_scores=True,
        score_threshold=None,
    )
    assert "documents" in result


def test_run_with_score_threshold(retriever, mock_vector_store):
    """Test vector search with score threshold."""
    query = [0.1] * 768
    result = retriever.run(query=query, score_threshold=0.5)

    mock_vector_store.search_by_vector.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=True,
        scale_scores=False,
        score_threshold=0.5,
    )
    assert "documents" in result


def test_run_with_document_embeddings(retriever, mock_vector_store):
    """Test vector search with document embeddings included."""
    query = [0.1] * 768
    result = retriever.run(query=query, exclude_document_embeddings=False)

    mock_vector_store.search_by_vector.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=False,
        scale_scores=False,
        score_threshold=None,
    )
    assert "documents" in result


def test_run_with_default_parameters(retriever, mock_vector_store):
    """Test search with default parameters."""
    query = [0.1] * 768
    result = retriever.run(query=query)

    mock_vector_store.search_by_vector.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=True,
        scale_scores=False,
        score_threshold=None,
    )
    assert "documents" in result
