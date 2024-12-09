from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.components.retrievers.weaviate import WeaviateDocumentRetriever as WeaviateDocumentRetrieverComponent
from dynamiq.connections import Weaviate
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers import WeaviateDocumentRetriever
from dynamiq.nodes.retrievers.base import RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore


@pytest.fixture
def mock_weaviate_vector_store():
    mock_store = MagicMock(spec=WeaviateVectorStore)
    mock_store.client = MagicMock()  # Add the client attribute to the mock
    return mock_store


@pytest.fixture
def weaviate_document_retriever(mock_weaviate_vector_store):
    retriever = WeaviateDocumentRetriever(vector_store=mock_weaviate_vector_store)
    return retriever


@patch("dynamiq.connections.Weaviate.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect):
    retriever = WeaviateDocumentRetriever()
    assert isinstance(retriever.connection, Weaviate)


def test_initialization_with_vector_store(mock_weaviate_vector_store):
    retriever = WeaviateDocumentRetriever(vector_store=mock_weaviate_vector_store)
    assert retriever.vector_store == mock_weaviate_vector_store
    assert retriever.connection is None


def test_vector_store_cls(weaviate_document_retriever):
    assert weaviate_document_retriever.vector_store_cls == WeaviateVectorStore


def test_to_dict_exclude_params(weaviate_document_retriever):
    exclude_params = weaviate_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(weaviate_document_retriever, mock_weaviate_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    weaviate_document_retriever.init_components(connection_manager)
    assert isinstance(weaviate_document_retriever.document_retriever, WeaviateDocumentRetrieverComponent)
    assert weaviate_document_retriever.document_retriever.vector_store == mock_weaviate_vector_store


def test_execute(weaviate_document_retriever):
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], filters={"field": "value"}, top_k=5)
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    weaviate_document_retriever.document_retriever = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    weaviate_document_retriever.document_retriever.run.return_value = mock_output

    result = weaviate_document_retriever.execute(input_data, config)

    weaviate_document_retriever.document_retriever.run.assert_called_once_with(
        input_data.embedding, filters=input_data.filters, top_k=input_data.top_k, content_key=None
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(weaviate_document_retriever):
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        weaviate_document_retriever.execute(RetrieverInputSchema(), config)


def test_execute_with_default_filters_and_top_k(weaviate_document_retriever):
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3])
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    weaviate_document_retriever.document_retriever = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    weaviate_document_retriever.document_retriever.run.return_value = mock_output

    result = weaviate_document_retriever.execute(input_data, config)

    weaviate_document_retriever.document_retriever.run.assert_called_once_with(
        input_data.embedding,
        filters=weaviate_document_retriever.filters,
        top_k=weaviate_document_retriever.top_k,
        content_key=None,
    )

    assert result == {"documents": mock_output["documents"]}
