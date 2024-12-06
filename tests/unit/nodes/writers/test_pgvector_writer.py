from unittest.mock import MagicMock, patch

import pytest

from dynamiq.nodes.writers.pgvector import PGVectorDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams
from dynamiq.storages.vector.pgvector.pgvector import PGVectorStore


@pytest.fixture
def mock_pgvector_vector_store():
    mock_store = MagicMock(spec=PGVectorStore)
    mock_store.write_documents.return_value = 2
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture
def pgvector_document_writer(mock_pgvector_vector_store):
    writer = PGVectorDocumentWriter(vector_store=mock_pgvector_vector_store)
    return writer


def test_initialization_with_vector_store(mock_pgvector_vector_store):
    writer = PGVectorDocumentWriter(vector_store=mock_pgvector_vector_store)
    assert writer.vector_store == mock_pgvector_vector_store


def test_vector_store_cls(pgvector_document_writer):
    assert pgvector_document_writer.vector_store_cls == PGVectorStore
    assert isinstance(pgvector_document_writer.vector_store, PGVectorStore)


def test_vector_store_params(pgvector_document_writer):
    expected_store_params = BaseWriterVectorStoreParams.model_fields

    with patch.object(pgvector_document_writer, "client", new_callable=MagicMock):
        params = pgvector_document_writer.vector_store_params
        for store_param in expected_store_params:
            assert store_param in params


def test_execute(pgvector_document_writer, mock_pgvector_vector_store):
    input_data = {
        "documents": [
            {"id": "1", "content": "Document 1", "embedding": [0.1, 0.2, 0.3]},
            {"id": "2", "content": "Document 2", "embedding": [0.4, 0.5, 0.6]},
        ]
    }
    config = RunnableConfig(callbacks=[])

    result = pgvector_document_writer.execute(input_data, config)

    mock_pgvector_vector_store.write_documents.assert_called_once_with(
        input_data["documents"], content_key=None, embedding_key=None
    )

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(pgvector_document_writer):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        pgvector_document_writer.execute(input_data, config)
