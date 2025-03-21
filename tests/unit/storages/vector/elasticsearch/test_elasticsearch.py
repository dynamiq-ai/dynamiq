from unittest.mock import MagicMock, patch

import pytest

from dynamiq.storages.vector import ElasticsearchVectorStore
from dynamiq.types import Document


@pytest.fixture(autouse=True)
def set_dummy_es_credentials(monkeypatch):
    monkeypatch.setenv("ELASTICSEARCH_USERNAME", "test_user")
    monkeypatch.setenv("ELASTICSEARCH_PASSWORD", "test_pass")


@pytest.fixture
def mock_es_client():
    client = MagicMock()
    client.ping.return_value = True
    return client


@pytest.fixture
def es_vector_store(mock_es_client):
    with patch("elasticsearch.Elasticsearch", return_value=mock_es_client):
        return ElasticsearchVectorStore(
            content_key="content",
            index_name="test_index",
            dimension=768,
            similarity="cosine",
        )


@pytest.fixture()
def mock_es_filters():
    return {
        "operator": "AND",
        "conditions": [
            {"field": "company", "operator": "==", "value": "BMW"},
            {"field": "year", "operator": ">", "value": 2010},
        ],
    }


def test_initialization(es_vector_store):
    assert es_vector_store.client is not None


def test_initialization_with_connection(mock_es_client):
    store = ElasticsearchVectorStore(connection=mock_es_client)
    assert store.client == mock_es_client.connect.return_value
    assert store.index_name == "default"
    assert store.dimension == 1536
    assert store.similarity == "cosine"


def test_initialization_with_custom_params(mock_es_client):
    store = ElasticsearchVectorStore(
        connection=mock_es_client,
        index_name="custom_index",
        dimension=512,
        similarity="dot_product",
    )
    assert store.index_name == "custom_index"
    assert store.dimension == 512
    assert store.similarity == "dot_product"


def test_count_documents(es_vector_store, mock_es_client):
    mock_es_client.count.return_value = {"count": 5}
    assert es_vector_store.count_documents() == 5


def test_delete_documents(es_vector_store, mock_es_client):
    document_ids = ["1", "2", "3"]
    es_vector_store.delete_documents(document_ids=document_ids)
    mock_es_client.bulk.assert_called_once()


def test_delete_all_documents(es_vector_store, mock_es_client):
    es_vector_store.delete_documents(delete_all=True)

    mock_es_client.delete_by_query.assert_called_once_with(
        index=es_vector_store.index_name, query={"match_all": {}}, refresh=True
    )


def test_delete_documents_by_filters(es_vector_store, mock_es_client, mock_es_filters):
    norm_filters = {"must": [{"match": {"company": "BMW"}}, {"range": {"year": {"gt": 2010}}}]}

    es_vector_store.delete_documents_by_filters(mock_es_filters)

    mock_es_client.delete_by_query.assert_called_once_with(
        index=es_vector_store.index_name, query={"bool": norm_filters}, refresh=True
    )


def test_delete_documents_by_file_id(es_vector_store, mock_es_client, mock_es_filters):
    file_id = "file_id"
    with patch(
        "dynamiq.storages.vector.elasticsearch.elasticsearch.create_file_id_filter", return_value=mock_es_filters
    ):
        es_vector_store.delete_documents_by_file_id(file_id)
    norm_filters = {"must": [{"match": {"company": "BMW"}}, {"range": {"year": {"gt": 2010}}}]}

    mock_es_client.delete_by_query.assert_called_once_with(
        index=es_vector_store.index_name, query={"bool": norm_filters}, refresh=True
    )


def test_write_documents(es_vector_store, mock_es_client):
    mock_es_client.bulk.return_value = {"upsert_count": 2}
    documents = [
        Document(id="123", content="Document 1", embedding=[0.1, 0.2], metadata={"type": "test"}),
        Document(id="234", content="Document 2", embedding=[0.3, 0.4], metadata={"type": "test"}),
    ]
    assert es_vector_store.write_documents(documents, policy="overwrite") == 2
    mock_es_client.bulk.assert_called_once()


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


def test_get_field_statistics(es_vector_store, mock_es_client):
    """Test getting field statistics."""
    field = "field_name"
    es_vector_store.get_field_statistics(field)

    mock_es_client.search.assert_called_once_with(
        index=es_vector_store.index_name,
        body={"size": 0, "aggs": {"stats": {"stats": {"field": field}}}},
    )


def test_update_document_by_file_id(es_vector_store, mock_es_client):
    """Test updating document by their id."""
    file_id = "id"
    content = "updated content"
    metadata = {"key": "value"}
    embedding = [0.1] * 768
    es_vector_store.update_document_by_file_id(
        file_id=file_id,
        content=content,
        metadata=metadata,
        embedding=embedding,
    )

    mock_es_client.update.assert_called_once_with(
        index=es_vector_store.index_name,
        id=file_id,
        body={
            "doc": {
                es_vector_store.content_key: content,
                es_vector_store.embedding_key: embedding,
                "metadata": metadata,
            }
        },
        refresh=True,
    )


def test_create_alias(es_vector_store, mock_es_client):
    """Test creating an alias."""
    alias_name = "test_alias"
    index_names = ["test_index_1", "test_index_2"]
    es_vector_store.create_alias(alias_name, index_names)
    mock_es_client.indices.update_aliases.assert_called_once()


def test_search_by_vector_invalid_dimension(es_vector_store):
    """Test vector search with invalid embedding dimension."""
    with pytest.raises(ValueError, match="query_embedding must have dimension 768"):
        es_vector_store._embedding_retrieval([0.1] * 512)


def test_close(es_vector_store):
    es_vector_store.close()
    es_vector_store.client.close.assert_called_once()
