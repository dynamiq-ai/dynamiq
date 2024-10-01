from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models as rest

from dynamiq.storages.vector.qdrant.converters import convert_id
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_qdrant_client():
    return MagicMock()


@pytest.fixture
def qdrant_vector_store(mock_qdrant_client):
    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        return QdrantVectorStore()


def test_initialization(qdrant_vector_store):
    assert qdrant_vector_store.client is not None


def test_count_documents(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.count.return_value.count = 5
    assert qdrant_vector_store.count_documents() == 5


def test_filter_documents(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.scroll.return_value = ([], None)
    filters = {"operator": "==", "field": "field", "value": "value"}
    assert qdrant_vector_store.filter_documents(filters) == []


def test_write_documents(qdrant_vector_store, mock_documents):
    with patch.object(qdrant_vector_store, "_set_up_collection", return_value=None):
        assert qdrant_vector_store.write_documents(mock_documents) == 2


def test_delete_documents(qdrant_vector_store, mock_qdrant_client):
    qdrant_vector_store.delete_documents(document_ids=["1"])
    mock_qdrant_client.delete.assert_called_once()


def test_delete_documents_by_ids(qdrant_vector_store, mock_qdrant_client):
    document_ids = ["1", "2", "3"]
    qdrant_vector_store.delete_documents(document_ids=document_ids)
    mock_qdrant_client.delete.assert_called_once_with(
        collection_name=qdrant_vector_store.index_name,
        points_selector=[convert_id(_id) for _id in document_ids],
        wait=qdrant_vector_store.wait_result_from_api,
    )


def test_delete_all_documents(qdrant_vector_store, mock_qdrant_client):
    qdrant_vector_store.delete_documents(delete_all=True)
    mock_qdrant_client.delete_collection.assert_called_once_with(collection_name=qdrant_vector_store.index_name)


def test_delete_documents_raises_value_error(qdrant_vector_store):
    with pytest.raises(ValueError, match="Either `document_ids` or `delete_all` must be provided."):
        qdrant_vector_store.delete_documents()


def test_delete_documents_by_filters(qdrant_vector_store, mock_qdrant_client, mock_documents, mock_filters):
    with patch.object(qdrant_vector_store, "filter_documents", return_value=mock_documents):
        qdrant_vector_store.delete_documents_by_filters(mock_filters)
    mock_qdrant_client.delete.assert_called_once_with(
        collection_name=qdrant_vector_store.index_name,
        points_selector=[convert_id(doc.id) for doc in mock_documents],
        wait=qdrant_vector_store.wait_result_from_api,
    )


def test_delete_documents_by_filters_raises_value_error(qdrant_vector_store):
    with pytest.raises(ValueError, match="No filters provided to delete documents."):
        qdrant_vector_store.delete_documents_by_filters(filters=None)


def test_delete_documents_by_file_id(qdrant_vector_store, mock_qdrant_client, mock_documents):
    file_id = "file_id"

    with patch.object(qdrant_vector_store, "filter_documents", return_value=mock_documents):
        qdrant_vector_store.delete_documents_by_file_id(file_id)
    mock_qdrant_client.delete.assert_called_once_with(
        collection_name=qdrant_vector_store.index_name,
        points_selector=[convert_id(doc.id) for doc in mock_documents],
        wait=qdrant_vector_store.wait_result_from_api,
    )


def test_get_documents_generator(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.scroll.return_value = ([], None)
    generator = qdrant_vector_store.get_documents_generator()
    assert list(generator) == []


def test_list_documents(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.scroll.return_value = ([], None)
    assert qdrant_vector_store.list_documents() == []


def test_get_documents_by_id(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.retrieve.return_value = []
    assert qdrant_vector_store.get_documents_by_id(["1"]) == []


def test_query_by_embedding(qdrant_vector_store, mock_qdrant_client):
    query_embedding = [0.1, 0.2, 0.3]
    mock_qdrant_client.query_points.return_value.points = []
    assert qdrant_vector_store._query_by_embedding(query_embedding) == []


def test_get_distance(qdrant_vector_store):
    assert qdrant_vector_store.get_distance("cosine") == rest.Distance.COSINE


def test_create_payload_index(qdrant_vector_store, mock_qdrant_client):
    qdrant_vector_store._create_payload_index("collection", [{"field_name": "field", "field_schema": "schema"}])
    mock_qdrant_client.create_payload_index.assert_called_once()


def test_set_up_collection(qdrant_vector_store, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = False
    qdrant_vector_store._set_up_collection("collection", 128, True, False, "cosine", False, False)
    mock_qdrant_client.create_collection.assert_called_once()


def test_recreate_collection(qdrant_vector_store, mock_qdrant_client):
    qdrant_vector_store.recreate_collection("collection", rest.Distance.COSINE, 128)
    mock_qdrant_client.create_collection.assert_called_once()


def test_handle_duplicate_documents(qdrant_vector_store):
    documents = [Document(id="1", content="Document 1")]
    assert qdrant_vector_store._handle_duplicate_documents(documents) == documents


def test_drop_duplicate_documents(qdrant_vector_store):
    documents = [Document(id="1", content="Document 1"), Document(id="1", content="Document 1")]
    assert qdrant_vector_store._drop_duplicate_documents(documents) == [documents[0]]
