from unittest.mock import MagicMock

import pytest
from elasticsearch import Elasticsearch

from dynamiq.connections import Elasticsearch as ElasticsearchConnection
from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.types import Document


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


def test_create_index_if_not_exists(es_vector_store):
    es_vector_store.client.indices.exists.return_value = False
    es_vector_store._create_index_if_not_exists()

    es_vector_store.client.indices.create.assert_called_once()
    create_args = es_vector_store.client.indices.create.call_args[1]
    assert create_args["index"] == "test_index"
    mapping = create_args["body"]["mappings"]["properties"]
    assert mapping["content"]["type"] == "text"
    assert mapping["metadata"]["type"] == "object"
    assert mapping["embedding"]["type"] == "dense_vector"
    assert mapping["embedding"]["dims"] == 768
    assert mapping["embedding"]["similarity"] == "cosine"


def test_write_documents(es_vector_store):
    documents = [
        Document(id="1", content="test1", embedding=[0.1] * 768, metadata={"key": "value1"}),
        Document(id="2", content="test2", embedding=[0.2] * 768, metadata={"key": "value2"}),
    ]

    result = es_vector_store.write_documents(documents)

    assert result == 2
    es_vector_store.client.bulk.assert_called_once()
    bulk_operations = es_vector_store.client.bulk.call_args[0][0]
    assert len(bulk_operations) == 4  # 2 documents * 2 operations per document


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

    results = es_vector_store.search_by_vector(query_embedding=query_embedding, top_k=5, filters={"key": "value1"})

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

    results = es_vector_store.search_by_vector(query_embedding=query_embedding, scale_scores=True)

    assert len(results) == 1
    # For cosine similarity, score should be scaled to (0.5 + 1) / 2 = 0.75
    assert results[0].score == 0.75


def test_search_by_vector_invalid_dimension(es_vector_store):
    """Test vector search with invalid embedding dimension."""
    with pytest.raises(ValueError, match="query_embedding must have dimension 768"):
        es_vector_store.search_by_vector([0.1] * 512)


def test_search_by_vector_with_threshold(es_vector_store):
    """Test vector search with score threshold."""
    query_embedding = [0.1] * 768
    es_vector_store.client.search.return_value = {"hits": {"hits": []}}

    es_vector_store.search_by_vector(query_embedding=query_embedding, score_threshold=0.5)

    search_body = es_vector_store.client.search.call_args[1]["query"]
    assert search_body["knn"]["min_score"] == 0.5


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
    filters = {"key": "value"}
    es_vector_store.delete_documents_by_filters(filters)
    es_vector_store.client.delete_by_query.assert_called_once()
    query = es_vector_store.client.delete_by_query.call_args[1]["query"]
    assert "bool" in query
    assert "must" in query["bool"]


def test_list_documents(es_vector_store):
    mock_hits = {
        "hits": {
            "hits": [
                {
                    "_id": "1",
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

    documents = es_vector_store.list_documents(include_embeddings=True)
    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].content == "test1"
    assert documents[0].metadata == {"key": "value1"}
    assert len(documents[0].embedding) == 768


def test_count_documents(es_vector_store):
    es_vector_store.client.count.return_value = {"count": 42}
    count = es_vector_store.count_documents()
    assert count == 42
    es_vector_store.client.count.assert_called_once_with(index="test_index")


def test_close(es_vector_store):
    es_vector_store.close()
    es_vector_store.client.close.assert_called_once()
