from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.nodes.writers.base import WriterInputSchema
from dynamiq.nodes.writers.qdrant import QdrantDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore, QdrantWriterVectorStoreParams


@pytest.fixture
def mock_qdrant_vector_store():
    mock_store = MagicMock(spec=QdrantVectorStore)
    mock_store.write_documents.return_value = 2
    return mock_store


@pytest.fixture
def qdrant_document_writer(mock_qdrant_vector_store):
    writer = QdrantDocumentWriter(vector_store=mock_qdrant_vector_store)
    return writer


def test_initialization_with_vector_store(mock_qdrant_vector_store):
    writer = QdrantDocumentWriter(vector_store=mock_qdrant_vector_store)
    assert writer.vector_store == mock_qdrant_vector_store


def test_vector_store_cls(qdrant_document_writer):
    assert qdrant_document_writer.vector_store_cls == QdrantVectorStore
    assert isinstance(qdrant_document_writer.vector_store, QdrantVectorStore)


def test_vector_store_params(qdrant_document_writer):
    expected_store_params = QdrantWriterVectorStoreParams.model_fields

    with patch.object(qdrant_document_writer, "client", new_callable=MagicMock):
        params = qdrant_document_writer.vector_store_params
        for store_param in expected_store_params:
            assert store_param in params


def test_execute(qdrant_document_writer, mock_qdrant_vector_store):
    input_data = WriterInputSchema(
        documents=[
            {"id": "1", "content": "Document 1"},
            {"id": "2", "content": "Document 2"},
        ],
    )
    config = RunnableConfig(callbacks=[])

    result = qdrant_document_writer.execute(input_data, config)

    mock_qdrant_vector_store.write_documents.assert_called_once_with(input_data.documents, content_key=None)

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(qdrant_document_writer):
    config = RunnableConfig(callbacks=[])
    with pytest.raises(ValidationError):
        qdrant_document_writer.execute(WriterInputSchema(), config)
