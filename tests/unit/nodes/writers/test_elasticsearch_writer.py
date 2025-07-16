from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.nodes.writers.base import WriterInputSchema
from dynamiq.nodes.writers.elasticsearch import ElasticsearchDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.base import BaseWriterVectorStoreParams
from dynamiq.storages.vector.elasticsearch.elasticsearch import ElasticsearchVectorStore
from dynamiq.storages.vector.elasticsearch.elasticsearch import DuplicatePolicy


@pytest.fixture
def mock_elasticsearch_vector_store():
    mock_store = MagicMock(spec=ElasticsearchVectorStore)
    mock_store.write_documents.return_value = 2
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture
def elasticsearch_document_writer(mock_elasticsearch_vector_store):
    writer = ElasticsearchDocumentWriter(vector_store=mock_elasticsearch_vector_store)
    return writer


def test_initialization_with_vector_store(mock_elasticsearch_vector_store):
    writer = ElasticsearchDocumentWriter(vector_store=mock_elasticsearch_vector_store)
    assert writer.vector_store == mock_elasticsearch_vector_store


def test_vector_store_cls(elasticsearch_document_writer):
    assert elasticsearch_document_writer.vector_store_cls == ElasticsearchVectorStore
    assert isinstance(elasticsearch_document_writer.vector_store, ElasticsearchVectorStore)


def test_vector_store_params(elasticsearch_document_writer):
    expected_store_params = BaseWriterVectorStoreParams.model_fields

    with patch.object(elasticsearch_document_writer, "client", new_callable=MagicMock):
        params = elasticsearch_document_writer.vector_store_params
        for store_param in expected_store_params:
            assert store_param in params


def test_execute(elasticsearch_document_writer, mock_elasticsearch_vector_store):
    input_data = WriterInputSchema(
        documents=[
            {"id": "1", "content": "Document 1", "embedding": [0.1, 0.2, 0.3]},
            {"id": "2", "content": "Document 2", "embedding": [0.4, 0.5, 0.6]},
        ]
    )
    config = RunnableConfig(callbacks=[])

    result = elasticsearch_document_writer.execute(input_data, config)

    mock_elasticsearch_vector_store.write_documents.assert_called_once_with(
        documents=input_data.documents,
        content_key=None,
        embedding_key=None,
        policy=DuplicatePolicy.FAIL,
    )

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(elasticsearch_document_writer):
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        elasticsearch_document_writer.execute(WriterInputSchema(), config)
