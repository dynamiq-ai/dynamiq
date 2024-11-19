from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.writers.milvus import MilvusDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams
from dynamiq.storages.vector.milvus.milvus import MilvusVectorStore


@pytest.fixture
def mock_milvus_vector_store():
    mock_store = MagicMock(spec=MilvusVectorStore)
    mock_store.write_documents.return_value = 2
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture
def milvus_document_writer(mock_milvus_vector_store):
    writer = MilvusDocumentWriter(vector_store=mock_milvus_vector_store)
    return writer


def test_initialization_with_vector_store(mock_milvus_vector_store):
    writer = MilvusDocumentWriter(vector_store=mock_milvus_vector_store)
    assert writer.vector_store == mock_milvus_vector_store


def test_vector_store_cls(milvus_document_writer):
    assert milvus_document_writer.vector_store_cls == MilvusVectorStore
    assert isinstance(milvus_document_writer.vector_store, MilvusVectorStore)


def test_vector_store_params(milvus_document_writer):
    expected_store_params = BaseWriterVectorStoreParams.model_fields

    with patch.object(milvus_document_writer, "client", new_callable=MagicMock):
        params = milvus_document_writer.vector_store_params
        for store_param in expected_store_params:
            assert store_param in params


def test_execute(milvus_document_writer, mock_milvus_vector_store):
    input_data = {
        "documents": [
            {"id": "1", "content": "Document 1", "embedding": [0.1, 0.2, 0.3]},
            {"id": "2", "content": "Document 2", "embedding": [0.4, 0.5, 0.6]},
        ]
    }
    config = RunnableConfig(callbacks=[])

    result = milvus_document_writer.execute(input_data, config)

    mock_milvus_vector_store.write_documents.assert_called_once_with(input_data["documents"])

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(milvus_document_writer):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        milvus_document_writer.execute(input_data, config)
