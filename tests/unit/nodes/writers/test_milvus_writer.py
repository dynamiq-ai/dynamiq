from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.connections import Milvus
from dynamiq.nodes.writers.base import WriterInputSchema
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
def mock_milvus_connection():
    mock_connection = MagicMock(spec=Milvus)
    mock_connection.id = "test-connection-id"
    mock_connection.type = "dynamiq.connections.Milvus"
    mock_connection.model_dump_json.return_value = "{}"
    mock_client = MagicMock()
    mock_connection.connect.return_value = mock_client
    return mock_connection, mock_client


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


@pytest.mark.parametrize("dimension", [384, 768])
def test_writer_passes_dimension_to_params(dimension, mock_milvus_connection):
    mock_connection, mock_client = mock_milvus_connection

    with patch("dynamiq.connections.managers.ConnectionManager.get_connection_client", return_value=mock_client):
        writer = MilvusDocumentWriter(connection=mock_connection, dimension=dimension, create_if_not_exist=True)

        params = writer.vector_store_params
        assert params["dimension"] == dimension
        assert params["create_if_not_exist"] is True


@pytest.mark.parametrize("dimension", [512, 1024])
def test_writer_initializes_store_with_dimension(dimension, mock_milvus_connection):
    mock_connection, mock_client = mock_milvus_connection

    with patch("dynamiq.connections.managers.ConnectionManager.get_connection_client", return_value=mock_client):
        with patch.object(MilvusDocumentWriter, "connect_to_vector_store") as mock_connect:
            mock_vector_store = MagicMock()
            mock_connect.return_value = mock_vector_store

            writer = MilvusDocumentWriter(connection=mock_connection, dimension=dimension, create_if_not_exist=True)

            params = writer.vector_store_params
            assert params["dimension"] == dimension


def test_execute(milvus_document_writer, mock_milvus_vector_store):
    input_data = WriterInputSchema(
        documents=[
            {"id": "1", "content": "Document 1", "embedding": [0.1, 0.2, 0.3]},
            {"id": "2", "content": "Document 2", "embedding": [0.4, 0.5, 0.6]},
        ]
    )
    config = RunnableConfig(callbacks=[])

    result = milvus_document_writer.execute(input_data, config)

    mock_milvus_vector_store.write_documents.assert_called_once_with(
        input_data.documents, content_key=None, embedding_key=None
    )

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(milvus_document_writer):
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        milvus_document_writer.execute(WriterInputSchema(), config)
