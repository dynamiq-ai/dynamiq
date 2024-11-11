from unittest.mock import MagicMock, patch

import pytest

from dynamiq.storages.vector.milvus.filter import Filter
from dynamiq.storages.vector.milvus.milvus import MilvusVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_milvus_client():
    return MagicMock()


@pytest.fixture
def milvus_vector_store(mock_milvus_client):
    return MilvusVectorStore(client=mock_milvus_client, index_name="test_collection")


def test_initialization(milvus_vector_store):
    assert milvus_vector_store.client is not None


def test_count_documents(milvus_vector_store, mock_milvus_client):
    mock_milvus_client.get_collection_stats.return_value = {"row_count": 5}
    assert milvus_vector_store.count_documents() == 5


def test_write_documents(milvus_vector_store, mock_milvus_client):
    mock_milvus_client.upsert.return_value = {"upsert_count": 2}
    documents = [
        Document(id="1", content="Document 1", embedding=[0.1, 0.2], metadata={"type": "test"}),
        Document(id="2", content="Document 2", embedding=[0.3, 0.4], metadata={"type": "test"}),
    ]
    assert milvus_vector_store.write_documents(documents) == 2
    mock_milvus_client.upsert.assert_called_once()


def test_delete_documents_with_ids(milvus_vector_store, mock_milvus_client):
    document_ids = ["1", "2", "3"]
    milvus_vector_store.delete_documents(document_ids=document_ids)
    mock_milvus_client.delete.assert_called_once_with(collection_name="test_collection", ids=document_ids)


def test_delete_all_documents(milvus_vector_store, mock_milvus_client):
    mock_milvus_client.reset_mock()

    milvus_vector_store.delete_documents(delete_all=True)
    mock_milvus_client.drop_collection.assert_called_once_with(collection_name="test_collection")
    mock_milvus_client.create_collection.assert_called_once()
    mock_milvus_client.load_collection.assert_called_once_with("test_collection")


def test_delete_documents_raises_value_error(milvus_vector_store):
    with pytest.raises(ValueError, match="Either `document_ids` or `delete_all` must be provided."):
        milvus_vector_store.delete_documents()


def test_delete_documents_by_filters(milvus_vector_store, mock_milvus_client):
    filters = {"type": "test"}
    with patch.object(Filter, "build_filter_expression", return_value="type == 'test'"):
        milvus_vector_store.delete_documents_by_filters(filters)
    mock_milvus_client.delete.assert_called_once_with(collection_name="test_collection", filter="type == 'test'")


def test_delete_documents_by_filters_raises_value_error(milvus_vector_store):
    with pytest.raises(ValueError, match="Filters must be provided to delete documents."):
        milvus_vector_store.delete_documents_by_filters(filters=None)


def test_delete_documents_by_file_id(milvus_vector_store, mock_milvus_client):
    file_id = "file_id"
    with patch("dynamiq.storages.vector.utils.create_file_id_filter", return_value={"file_id": file_id}):
        with patch.object(Filter, "build_filter_expression", return_value=f"file_id == '{file_id}'"):
            milvus_vector_store.delete_documents_by_file_id(file_id)
    mock_milvus_client.delete.assert_called_once_with(
        collection_name="test_collection", filter=f"file_id == '{file_id}'"
    )


def test_list_documents(milvus_vector_store, mock_milvus_client):
    mock_milvus_client.has_collection.return_value = True
    mock_milvus_client.describe_collection.return_value = {
        "fields": [{"name": "id"}, {"name": "content"}, {"name": "vector"}, {"name": "metadata"}]
    }
    mock_milvus_client.query.return_value = [
        {"id": "1", "content": "Document 1", "vector": [0.1, 0.2], "metadata": {"type": "test"}}
    ]
    documents = milvus_vector_store.list_documents(limit=10)
    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].content == "Document 1"


def test_search_embeddings(milvus_vector_store, mock_milvus_client):
    query_embeddings = [[0.1, 0.2]]
    mock_milvus_client.search.return_value = [
        [
            {
                "id": "1",
                "entity": {"content": "Document 1", "vector": [0.1, 0.2], "metadata": {"type": "test"}},
                "distance": 0.1,
            }
        ]
    ]
    documents = milvus_vector_store.search_embeddings(query_embeddings, top_k=1)
    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].content == "Document 1"
    assert documents[0].score == 0.1


def test_filter_documents(milvus_vector_store, mock_milvus_client):
    filters = {"type": "test"}
    with patch.object(Filter, "build_filter_expression", return_value="type == 'test'"):
        mock_milvus_client.query.return_value = [
            {"id": "1", "content": "Document 1", "vector": [0.1, 0.2], "metadata": {"type": "test"}}
        ]
        documents = milvus_vector_store.filter_documents(filters)
        assert len(documents) == 1
        assert documents[0].id == "1"
        assert documents[0].content == "Document 1"


def test_filter_documents_raises_value_error(milvus_vector_store):
    with pytest.raises(ValueError, match="No filters provided. No documents will be retrieved with filters."):
        milvus_vector_store.filter_documents(filters=None)


def test_get_result_to_documents():
    result = [{"id": "1", "content": "Document 1", "vector": [0.1, 0.2], "metadata": {"type": "test"}}]
    documents = MilvusVectorStore._get_result_to_documents(result)
    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].content == "Document 1"


def test_convert_query_result_to_documents(milvus_vector_store):
    query_result = [
        {
            "id": "1",
            "entity": {"content": "Document 1", "vector": [0.1, 0.2], "metadata": {"type": "test"}},
            "distance": 0.1,
        }
    ]
    documents = milvus_vector_store._convert_query_result_to_documents(query_result)
    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].content == "Document 1"
    assert documents[0].score == 0.1
