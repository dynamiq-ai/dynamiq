from unittest.mock import MagicMock

import pytest

from dynamiq.components.retrievers.elasticsearch import ElasticsearchDocumentRetriever
from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_vector_store():
    store = MagicMock(spec=ElasticsearchVectorStore)
    store._embedding_retrieval.return_value = [Document(id="1", content="test content", metadata={}, score=0.9)]
    return store


@pytest.fixture
def retriever(mock_vector_store, mock_es_filters):
    return ElasticsearchDocumentRetriever(vector_store=mock_vector_store, filters=mock_es_filters, top_k=5)


@pytest.fixture()
def mock_es_filters():
    return {
        "operator": "AND",
        "conditions": [
            {"field": "company", "operator": "==", "value": "BMW"},
            {"field": "year", "operator": ">", "value": 2010},
        ],
    }


def test_initialization(mock_vector_store, mock_es_filters):
    """Test component initialization with parameters."""
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_vector_store, filters=mock_es_filters, top_k=5)
    assert retriever.vector_store == mock_vector_store
    assert retriever.filters == mock_es_filters
    assert retriever.top_k == 5


def test_initialization_with_defaults(mock_vector_store):
    """Test component initialization with default values."""
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_vector_store)
    assert retriever.vector_store == mock_vector_store
    assert retriever.filters == {}
    assert retriever.top_k == 10


def test_run_basic_search(retriever, mock_vector_store, mock_es_filters):
    """Test basic vector similarity search."""
    query = [0.1] * 768
    result = retriever.run(query=query, filters=mock_es_filters, top_k=5)

    mock_vector_store._embedding_retrieval.assert_called_once_with(
        query_embedding=query,
        filters=mock_es_filters,
        top_k=5,
        exclude_document_embeddings=True,
        scale_scores=False,
        content_key=None,
        embedding_key=None,
    )
    assert "documents" in result
    assert len(result["documents"]) == 1


def test_run_with_score_scaling(retriever, mock_vector_store):
    """Test vector search with score scaling enabled."""
    query = [0.1] * 768
    result = retriever.run(query=query, scale_scores=True, content_key="content", embedding_key="embedding")

    mock_vector_store._embedding_retrieval.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=True,
        scale_scores=True,
        content_key="content",
        embedding_key="embedding",
    )
    assert "documents" in result


def test_run_with_document_embeddings(retriever, mock_vector_store):
    """Test vector search with document embeddings included."""
    query = [0.1] * 768
    result = retriever.run(query=query, exclude_document_embeddings=False)

    mock_vector_store._embedding_retrieval.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=False,
        scale_scores=False,
        content_key=None,
        embedding_key=None,
    )
    assert "documents" in result


def test_run_with_default_parameters(retriever, mock_vector_store):
    """Test search with default parameters."""
    query = [0.1] * 768
    result = retriever.run(query=query)

    mock_vector_store._embedding_retrieval.assert_called_once_with(
        query_embedding=query,
        filters=retriever.filters,
        top_k=retriever.top_k,
        exclude_document_embeddings=True,
        scale_scores=False,
        content_key=None,
        embedding_key=None,
    )
    assert "documents" in result
