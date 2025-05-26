from unittest.mock import MagicMock

import pytest

from dynamiq.storages.vector.base import BaseVectorStore


class MockVectorStore(BaseVectorStore):
    def __init__(self):
        self.filter_calls = []
        self.last_filters = None

    def delete_documents_by_filters(self, filters):
        self.filter_calls.append(filters)
        self.last_filters = filters


@pytest.fixture
def mock_store():
    return MockVectorStore()


@pytest.fixture
def mock_filter():
    return {"mock": "filter"}


@pytest.fixture
def mock_create_file_id_filter(monkeypatch, mock_filter):
    mock_func = MagicMock(return_value=mock_filter)
    monkeypatch.setattr("dynamiq.storages.vector.base.create_file_id_filter", mock_func)
    return mock_func


@pytest.fixture
def mock_create_file_ids_filter(monkeypatch, mock_filter):
    mock_func = MagicMock(return_value=mock_filter)
    monkeypatch.setattr("dynamiq.storages.vector.base.create_file_ids_filter", mock_func)
    return mock_func


def test_delete_documents_by_file_id(mock_store, mock_create_file_id_filter, mock_filter):
    mock_store.delete_documents_by_file_id("file1")

    mock_create_file_id_filter.assert_called_once_with("file1")
    assert mock_store.last_filters == mock_filter


def test_delete_documents_by_file_ids_simple(mock_store, mock_create_file_ids_filter, mock_filter):
    test_ids = ["file1", "file2", "file3"]
    mock_store.delete_documents_by_file_ids(test_ids)

    mock_create_file_ids_filter.assert_called_once_with(test_ids)
    assert mock_store.last_filters == mock_filter


def test_delete_documents_by_file_ids_batching(mock_store, mock_create_file_ids_filter, mock_filter):
    file_ids = [f"file{i}" for i in range(1200)]
    mock_store.delete_documents_by_file_ids(file_ids, batch_size=500)

    assert mock_create_file_ids_filter.call_count == 3
    assert len(mock_store.filter_calls) == 3

    expected_calls = [[file_ids[0:500]], [file_ids[500:1000]], [file_ids[1000:]]]

    for i, call in enumerate(mock_create_file_ids_filter.call_args_list):
        assert call[0][0] == expected_calls[i][0]


def test_delete_documents_by_file_ids_empty(mock_store):
    mock_store.delete_documents_by_file_ids([])

    assert len(mock_store.filter_calls) == 0
    assert mock_store.last_filters is None
