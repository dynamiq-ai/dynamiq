from unittest.mock import MagicMock, patch

import pytest

from dynamiq.components.retrievers.pgvector import PGVectorDocumentRetriever as PGVectorDocumentRetrieverComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.pgvector import PGVectorDocumentRetriever
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import PGVectorStore


@pytest.fixture
def mock_pg_vector_store():
    mock_store = MagicMock(spec=PGVectorStore)
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture(autouse=True)
def mock_pgvector_connect():
    with patch("dynamiq.connections.connections.PostgreSQL.connect", return_value=MagicMock()) as mock_connect:
        yield mock_connect


@pytest.fixture
def pgvector_document_retriever(mock_pg_vector_store):
    retriever = PGVectorDocumentRetriever(vector_store=mock_pg_vector_store)
    return retriever


@patch.object(PGVectorDocumentRetriever, "connect_to_vector_store")
def test_initialization_with_defaults(mock_connect_to_vector_store):
    mock_pg_vector_store = MagicMock(spec=PGVectorStore)
    mock_connect_to_vector_store.return_value = mock_pg_vector_store

    retriever = PGVectorDocumentRetriever()

    mock_connect_to_vector_store.assert_called_once()
    assert retriever.vector_store == mock_pg_vector_store


def test_initialization_with_vector_store(mock_pg_vector_store):
    retriever = PGVectorDocumentRetriever(vector_store=mock_pg_vector_store)
    assert retriever.vector_store == mock_pg_vector_store
    assert retriever.connection is None


def test_vector_store_cls(pgvector_document_retriever):
    assert pgvector_document_retriever.vector_store_cls == PGVectorStore


def test_to_dict_exclude_params(pgvector_document_retriever):
    exclude_params = pgvector_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(pgvector_document_retriever, mock_pg_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    pgvector_document_retriever.init_components(connection_manager)
    assert isinstance(pgvector_document_retriever.document_retriever, PGVectorDocumentRetrieverComponent)
    assert pgvector_document_retriever.document_retriever.vector_store == mock_pg_vector_store


def test_execute(pgvector_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3], "filters": {"field": "value"}, "top_k": 5}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    pgvector_document_retriever.document_retriever = MagicMock(spec=PGVectorDocumentRetrieverComponent)
    pgvector_document_retriever.document_retriever.run.return_value = mock_output

    result = pgvector_document_retriever.execute(input_data, config)

    pgvector_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"],
        filters=input_data["filters"],
        top_k=input_data["top_k"],
        content_key=None,
        embedding_key=None,
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(pgvector_document_retriever):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        pgvector_document_retriever.execute(input_data, config)


def test_execute_with_default_filters_and_top_k(pgvector_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3]}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    pgvector_document_retriever.document_retriever = MagicMock(spec=PGVectorDocumentRetrieverComponent)
    pgvector_document_retriever.document_retriever.run.return_value = mock_output

    result = pgvector_document_retriever.execute(input_data, config)

    pgvector_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"],
        filters=pgvector_document_retriever.filters,
        top_k=pgvector_document_retriever.top_k,
        content_key=None,
        embedding_key=None,
    )

    assert result == {"documents": mock_output["documents"]}
