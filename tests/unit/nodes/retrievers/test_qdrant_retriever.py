from unittest.mock import MagicMock

import pytest

from dynamiq.components.retrievers.qdrant import QdrantDocumentRetriever as QdrantDocumentRetrieverComponent
from dynamiq.connections import Qdrant
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.qdrant import QdrantDocumentRetriever
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import QdrantVectorStore


@pytest.fixture
def mock_qdrant_vector_store():
    mock_store = MagicMock(spec=QdrantVectorStore)
    return mock_store


@pytest.fixture
def qdrant_document_retriever(mock_qdrant_vector_store):
    retriever = QdrantDocumentRetriever(vector_store=mock_qdrant_vector_store)
    return retriever


def test_initialization_with_defaults():
    retriever = QdrantDocumentRetriever()
    assert isinstance(retriever.connection, Qdrant)


def test_initialization_with_vector_store(mock_qdrant_vector_store):
    retriever = QdrantDocumentRetriever(vector_store=mock_qdrant_vector_store)
    assert retriever.vector_store == mock_qdrant_vector_store
    assert retriever.connection is None


def test_vector_store_cls(qdrant_document_retriever):
    assert qdrant_document_retriever.vector_store_cls == QdrantVectorStore


def test_to_dict_exclude_params(qdrant_document_retriever):
    exclude_params = qdrant_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(qdrant_document_retriever, mock_qdrant_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    qdrant_document_retriever.init_components(connection_manager)
    assert isinstance(qdrant_document_retriever.document_retriever, QdrantDocumentRetrieverComponent)
    assert qdrant_document_retriever.document_retriever.vector_store == mock_qdrant_vector_store


def test_execute(qdrant_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3], "filters": {"field": "value"}, "top_k": 5}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    qdrant_document_retriever.document_retriever = MagicMock(spec=QdrantDocumentRetrieverComponent)
    qdrant_document_retriever.document_retriever.run.return_value = mock_output

    result = qdrant_document_retriever.execute(input_data, config)

    qdrant_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=input_data["filters"], top_k=input_data["top_k"]
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(qdrant_document_retriever):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        qdrant_document_retriever.execute(input_data, config)


def test_execute_with_default_filters_and_top_k(qdrant_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3]}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    qdrant_document_retriever.document_retriever = MagicMock(spec=QdrantDocumentRetrieverComponent)
    qdrant_document_retriever.document_retriever.run.return_value = mock_output

    result = qdrant_document_retriever.execute(input_data, config)

    qdrant_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=qdrant_document_retriever.filters, top_k=qdrant_document_retriever.top_k
    )

    assert result == {"documents": mock_output["documents"]}
