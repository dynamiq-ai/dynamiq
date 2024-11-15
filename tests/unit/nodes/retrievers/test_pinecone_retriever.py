from unittest.mock import MagicMock, patch

import pytest

from dynamiq.components.retrievers.pinecone import PineconeDocumentRetriever as PineconeDocumentRetrieverComponent
from dynamiq.connections import Pinecone
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.pinecone import (
    PineconeDocumentRetriever,
)  # Adjust the import based on your module structure
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PineconeVectorStore


@pytest.fixture
def mock_pinecone_vector_store():
    mock_store = MagicMock(spec=PineconeVectorStore)
    mock_store.client = MagicMock()  # Add the client attribute to the mock
    return mock_store


@pytest.fixture
def pinecone_document_retriever(mock_pinecone_vector_store):
    retriever = PineconeDocumentRetriever(vector_store=mock_pinecone_vector_store)
    return retriever


@patch("dynamiq.storages.vector.pinecone.pinecone.PineconeVectorStore.connect_to_index", return_value=MagicMock())
@patch("dynamiq.connections.Pinecone.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect, mock_connect_to_index):
    retriever = PineconeDocumentRetriever()
    assert isinstance(retriever.connection, Pinecone)


def test_initialization_with_vector_store(mock_pinecone_vector_store):
    retriever = PineconeDocumentRetriever(vector_store=mock_pinecone_vector_store)
    assert retriever.vector_store == mock_pinecone_vector_store
    assert retriever.connection is None


def test_vector_store_cls(pinecone_document_retriever):
    assert pinecone_document_retriever.vector_store_cls == PineconeVectorStore


def test_to_dict_exclude_params(pinecone_document_retriever):
    exclude_params = pinecone_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(pinecone_document_retriever, mock_pinecone_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    pinecone_document_retriever.init_components(connection_manager)
    assert isinstance(pinecone_document_retriever.document_retriever, PineconeDocumentRetrieverComponent)
    assert pinecone_document_retriever.document_retriever.vector_store == mock_pinecone_vector_store


def test_execute(pinecone_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3], "filters": {"field": "value"}, "top_k": 5}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    pinecone_document_retriever.document_retriever = MagicMock(spec=PineconeDocumentRetrieverComponent)
    pinecone_document_retriever.document_retriever.run.return_value = mock_output

    result = pinecone_document_retriever.execute(input_data, config)

    pinecone_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=input_data["filters"], top_k=input_data["top_k"]
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(pinecone_document_retriever):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        pinecone_document_retriever.execute(input_data, config)


def test_execute_with_default_filters_and_top_k(pinecone_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3]}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    pinecone_document_retriever.document_retriever = MagicMock(spec=PineconeDocumentRetrieverComponent)
    pinecone_document_retriever.document_retriever.run.return_value = mock_output

    result = pinecone_document_retriever.execute(input_data, config)

    pinecone_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=pinecone_document_retriever.filters, top_k=pinecone_document_retriever.top_k
    )

    assert result == {"documents": mock_output["documents"]}
