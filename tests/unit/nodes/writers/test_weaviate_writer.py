from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.connections import Weaviate
from dynamiq.nodes.writers.base import WriterInputSchema
from dynamiq.nodes.writers.weaviate import WeaviateDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.weaviate import WeaviateVectorStore
from dynamiq.types import Document


class MockWeaviateVectorStore(WeaviateVectorStore):
    """A mock WeaviateVectorStore that can be used in tests."""

    def __init__(self, *args, **kwargs):
        # Skip the initialization of the actual class
        # This prevents real calls to Weaviate
        self.client = MagicMock()
        self.write_documents = MagicMock(return_value=2)


@pytest.fixture
def mock_weaviate_vector_store():
    """Create a mock WeaviateVectorStore."""
    mock_store = MockWeaviateVectorStore()
    return mock_store


@pytest.fixture
def weaviate_document_writer(mock_weaviate_vector_store):
    """Create a WeaviateDocumentWriter with a mocked WeaviateVectorStore."""
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(vector_store=mock_weaviate_vector_store)
    return writer


@patch("dynamiq.connections.Weaviate.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect):
    """Test initialization with default parameters."""
    mock_store = MockWeaviateVectorStore()

    with patch.object(WeaviateDocumentWriter, "init_components"):
        with patch.object(WeaviateDocumentWriter, "connect_to_vector_store", return_value=mock_store):
            writer = WeaviateDocumentWriter()
            assert isinstance(writer.connection, Weaviate)


def test_initialization_with_vector_store(mock_weaviate_vector_store):
    """Test initialization with a provided vector store."""
    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(vector_store=mock_weaviate_vector_store)
        assert writer.vector_store == mock_weaviate_vector_store
        assert writer.connection is None


def test_vector_store_cls(weaviate_document_writer):
    """Test that the correct vector store class is returned."""
    assert weaviate_document_writer.vector_store_cls == WeaviateVectorStore


def test_vector_store_params(weaviate_document_writer):
    """Test that the vector_store_params property returns the correct parameters."""
    params = weaviate_document_writer.vector_store_params
    assert "connection" in params
    assert "client" in params


def test_tenant_in_vector_store_params():
    """Test that tenant_name is included in vector_store_params when specified."""
    mock_store = MockWeaviateVectorStore()

    with patch.object(WeaviateDocumentWriter, "init_components"):
        writer = WeaviateDocumentWriter(tenant_name="test_tenant", vector_store=mock_store)

        params = writer.vector_store_params
        assert "tenant_name" in params
        assert params["tenant_name"] == "test_tenant"


def test_execute(weaviate_document_writer, mock_weaviate_vector_store):
    """Test the execute method with valid input data."""
    documents = [
        Document(id="1", content="Document 1", metadata={"key": "value"}, embedding=[0.1, 0.2, 0.3]),
        Document(id="2", content="Document 2", metadata={"key": "value"}, embedding=[0.4, 0.5, 0.6]),
    ]

    input_data = WriterInputSchema(documents=documents)
    config = RunnableConfig(callbacks=[])

    result = weaviate_document_writer.execute(input_data, config)

    mock_weaviate_vector_store.write_documents.assert_called_once_with(documents, content_key=None)

    assert result == {"upserted_count": 2}


def test_execute_with_content_key(weaviate_document_writer, mock_weaviate_vector_store):
    """Test the execute method with a custom content key."""
    documents = [
        Document(id="1", content="Document 1", metadata={"key": "value"}, embedding=[0.1, 0.2, 0.3]),
        Document(id="2", content="Document 2", metadata={"key": "value"}, embedding=[0.4, 0.5, 0.6]),
    ]

    input_data = WriterInputSchema(documents=documents, content_key="custom_content")
    config = RunnableConfig(callbacks=[])

    result = weaviate_document_writer.execute(input_data, config)

    mock_weaviate_vector_store.write_documents.assert_called_once_with(documents, content_key="custom_content")

    assert result == {"upserted_count": 2}


def test_execute_with_missing_documents_key(weaviate_document_writer):
    """Test that execute raises a ValidationError when documents are missing."""
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        weaviate_document_writer.execute(WriterInputSchema(), config)


def test_writer_with_tenant():
    """Test that a writer can be created with a tenant name."""
    mock_connect = MagicMock()

    mock_vector_store = MockWeaviateVectorStore()

    with patch("dynamiq.connections.Weaviate.connect", return_value=mock_connect):
        with patch.object(WeaviateDocumentWriter, "connect_to_vector_store", return_value=mock_vector_store):
            with patch.object(WeaviateDocumentWriter, "init_components"):
                writer = WeaviateDocumentWriter(
                    index_name="test_collection", create_if_not_exist=True, tenant_name="test_tenant"
                )

                assert writer.tenant_name == "test_tenant"

                params = writer.vector_store_params
                assert params["tenant_name"] == "test_tenant"
                assert params["index_name"] == "test_collection"
                assert params["create_if_not_exist"] is True
