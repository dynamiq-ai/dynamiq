from unittest.mock import MagicMock, patch

import pytest

from dynamiq.components.retrievers.milvus import MilvusDocumentRetriever as MilvusDocumentRetrieverComponent
from dynamiq.connections import Milvus
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.milvus import MilvusDocumentRetriever
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import MilvusVectorStore


@pytest.fixture
def mock_milvus_vector_store():
    mock_store = MagicMock(spec=MilvusVectorStore)
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture
def milvus_document_retriever(mock_milvus_vector_store):
    retriever = MilvusDocumentRetriever(vector_store=mock_milvus_vector_store)
    return retriever


def test_initialization_with_defaults():
    mock_milvus_client = MagicMock()
    mock_milvus_client.has_collection.return_value = True

    with patch("dynamiq.connections.Milvus.connect", return_value=mock_milvus_client):
        retriever = MilvusDocumentRetriever()
        assert isinstance(retriever.connection, Milvus)
        assert retriever.vector_store is not None
        assert retriever.vector_store.client == mock_milvus_client


def test_initialization_with_vector_store(mock_milvus_vector_store):
    retriever = MilvusDocumentRetriever(vector_store=mock_milvus_vector_store)
    assert retriever.vector_store == mock_milvus_vector_store
    assert retriever.connection is None


def test_vector_store_cls(milvus_document_retriever):
    assert milvus_document_retriever.vector_store_cls == MilvusVectorStore


def test_to_dict_exclude_params(milvus_document_retriever):
    exclude_params = milvus_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(milvus_document_retriever, mock_milvus_vector_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    milvus_document_retriever.init_components(connection_manager)
    assert isinstance(milvus_document_retriever.document_retriever, MilvusDocumentRetrieverComponent)
    assert milvus_document_retriever.document_retriever.vector_store == mock_milvus_vector_store


def test_execute(milvus_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3], "filters": {"field": "value"}, "top_k": 5}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    milvus_document_retriever.document_retriever = MagicMock(spec=MilvusDocumentRetrieverComponent)
    milvus_document_retriever.document_retriever.run.return_value = mock_output

    result = milvus_document_retriever.execute(input_data, config)

    milvus_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=input_data["filters"], top_k=input_data["top_k"]
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_embedding_key(milvus_document_retriever):
    input_data = {}
    config = RunnableConfig(callbacks=[])

    with pytest.raises(KeyError):
        milvus_document_retriever.execute(input_data, config)


def test_execute_with_default_filters_and_top_k(milvus_document_retriever):
    input_data = {"embedding": [0.1, 0.2, 0.3]}
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    milvus_document_retriever.document_retriever = MagicMock(spec=MilvusDocumentRetrieverComponent)
    milvus_document_retriever.document_retriever.run.return_value = mock_output

    result = milvus_document_retriever.execute(input_data, config)

    milvus_document_retriever.document_retriever.run.assert_called_once_with(
        input_data["embedding"], filters=milvus_document_retriever.filters, top_k=milvus_document_retriever.top_k
    )

    assert result == {"documents": mock_output["documents"]}
