from unittest.mock import MagicMock

import pytest
from elasticsearch import Elasticsearch

from dynamiq.connections import Elasticsearch as ElasticsearchConnection
from dynamiq.storages.vector import ElasticsearchVectorStore


@pytest.fixture
def mock_es_client():
    return MagicMock(spec=Elasticsearch)


@pytest.fixture
def mock_es_connection():
    connection = MagicMock(spec=ElasticsearchConnection)
    connection.connect.return_value = MagicMock(spec=Elasticsearch)
    return connection


@pytest.fixture
def es_vector_store(mock_es_connection):
    return ElasticsearchVectorStore(
        connection=mock_es_connection,
        index_name="test_index",
        dimension=768,
        similarity="cosine",
    )


def test_initialization_with_connection(mock_es_connection):
    store = ElasticsearchVectorStore(connection=mock_es_connection)
    assert store.client == mock_es_connection.connect.return_value
    assert store.index_name == "default"
    assert store.dimension == 1536
    assert store.similarity == "cosine"


def test_initialization_with_custom_params(mock_es_connection):
    store = ElasticsearchVectorStore(
        connection=mock_es_connection,
        index_name="custom_index",
        dimension=512,
        similarity="dot_product",
    )
    assert store.index_name == "custom_index"
    assert store.dimension == 512
    assert store.similarity == "dot_product"


def test_write_documents_empty_list(es_vector_store):
    result = es_vector_store.write_documents([])
    assert result == 0
    es_vector_store.client.bulk.assert_not_called()


def test_write_documents_invalid_type(es_vector_store):
    with pytest.raises(ValueError, match="Documents must be of type Document"):
        es_vector_store.write_documents([{"id": "1", "content": "invalid"}])


def test_search_by_vector(es_vector_store):
    """Test vector similarity search with basic parameters."""
    query_embedding = [0.1] * 768
    mock_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_score": 0.9,
                    "_source": {
                        "content": "test1",
                        "metadata": {"key": "value1"},
                        "embedding": [0.1] * 768,
                    },
                }
            ]
        }
    }
    es_vector_store.client.search.return_value = mock_hits

    filters = {
        "field": "key",
        "operator": "==",
        "value": "value1",
    }
    results = es_vector_store._embedding_retrieval(query_embedding=query_embedding, top_k=5, filters=filters)

    assert len(results) == 1
    assert results[0].id == "1"
    assert results[0].content == "test1"
    assert results[0].metadata == {"key": "value1"}
    assert results[0].score == 0.9


def test_search_by_vector_with_score_scaling(es_vector_store):
    """Test vector search with score scaling enabled."""
    query_embedding = [0.1] * 768
    mock_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "1",
                    "_score": 0.5,  # Score between -1 and 1 for cosine similarity
                    "_source": {"content": "test1", "metadata": {"key": "value1"}},
                }
            ]
        }
    }
    es_vector_store.client.search.return_value = mock_hits

    results = es_vector_store._embedding_retrieval(query_embedding=query_embedding, scale_scores=True)

    assert len(results) == 1
    # For cosine similarity, score should be scaled to (0.5 + 1) / 2 = 0.75
    assert results[0].score == 0.75


def test_search_by_vector_invalid_dimension(es_vector_store):
    """Test vector search with invalid embedding dimension."""
    with pytest.raises(ValueError, match="query_embedding must have dimension 768"):
        es_vector_store._embedding_retrieval([0.1] * 512)


def test_delete_documents(es_vector_store):
    document_ids = ["1", "2"]
    es_vector_store.delete_documents(document_ids=document_ids)
    es_vector_store.client.bulk.assert_called_once()


def test_delete_all_documents(es_vector_store):
    es_vector_store.delete_documents(delete_all=True)
    es_vector_store.client.delete_by_query.assert_called_once_with(
        index="test_index", query={"match_all": {}}, refresh=True
    )


def test_delete_documents_by_filters(es_vector_store):
    filters = {
        "field": "key",
        "operator": "==",
        "value": "value",
    }
    es_vector_store.delete_documents_by_filters(filters)
    es_vector_store.client.delete_by_query.assert_called_once()
    query = es_vector_store.client.delete_by_query.call_args[1]["query"]
    assert "bool" in query
    assert "must" in query["bool"]


def test_close(es_vector_store):
    es_vector_store.close()
    es_vector_store.client.close.assert_called_once()
