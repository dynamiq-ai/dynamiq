from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.components.retrievers.chroma import ChromaDocumentRetriever as ChromaDocumentRetrieverComponent
from dynamiq.connections import Chroma
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.base import RetrieverInputSchema
from dynamiq.nodes.retrievers.chroma import ChromaDocumentRetriever
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import ChromaVectorStore


@pytest.fixture
def mock_chroma_vector_store():
    mock_store = MagicMock(spec=ChromaVectorStore)
    mock_store.client = MagicMock()  # Add the client attribute to the mock
    return mock_store


@pytest.fixture
def chroma_document_retriever(mock_chroma_vector_store):
    retriever = ChromaDocumentRetriever(vector_store=mock_chroma_vector_store)
    return retriever


@patch("dynamiq.connections.Chroma.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect):
    retriever = ChromaDocumentRetriever()
    assert isinstance(retriever.connection, Chroma)


def test_initialization_with_vector_store(mock_chroma_vector_store):
    retriever = ChromaDocumentRetriever(vector_store=mock_chroma_vector_store)
    assert retriever.vector_store == mock_chroma_vector_store
    assert retriever.connection is None


def test_vector_store_cls(chroma_document_retriever):
    assert chroma_document_retriever.vector_store_cls == ChromaVectorStore


def test_to_dict_exclude_params(chroma_document_retriever):
    exclude_params = chroma_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(chroma_document_retriever, mock_chroma_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    chroma_document_retriever.init_components(connection_manager)
    assert isinstance(chroma_document_retriever.document_retriever, ChromaDocumentRetrieverComponent)
    assert chroma_document_retriever.document_retriever.vector_store == mock_chroma_vector_store


def test_execute(chroma_document_retriever):
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], filters={"field": "value"}, top_k=5)
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    chroma_document_retriever.document_retriever = MagicMock(spec=ChromaDocumentRetrieverComponent)
    chroma_document_retriever.document_retriever.run.return_value = mock_output

    result = chroma_document_retriever.execute(input_data, config)

    chroma_document_retriever.document_retriever.run.assert_called_once_with(
        input_data.embedding, filters=input_data.filters, top_k=input_data.top_k
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(chroma_document_retriever):
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        chroma_document_retriever.execute(RetrieverInputSchema(), config)


def test_execute_with_default_filters_and_top_k(chroma_document_retriever):
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3])
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    chroma_document_retriever.document_retriever = MagicMock(spec=ChromaDocumentRetrieverComponent)
    chroma_document_retriever.document_retriever.run.return_value = mock_output

    result = chroma_document_retriever.execute(input_data, config)

    chroma_document_retriever.document_retriever.run.assert_called_once_with(
        input_data.embedding, filters=chroma_document_retriever.filters, top_k=chroma_document_retriever.top_k
    )

    assert result == {"documents": mock_output["documents"]}
