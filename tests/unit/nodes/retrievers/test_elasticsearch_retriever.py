from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dynamiq.components.retrievers.elasticsearch import (
    ElasticsearchDocumentRetriever as ElasticsearchDocumentRetrieverComponent,
)
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers.elasticsearch import ElasticsearchDocumentRetriever, ElasticsearchRetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import ElasticsearchVectorStore


@pytest.fixture
def mock_elasticsearch_store():
    mock_store = MagicMock(spec=ElasticsearchVectorStore)
    mock_store.client = MagicMock()
    return mock_store


@pytest.fixture(autouse=True)
def mock_elasticsearch_connect():
    with patch(
        "dynamiq.connections.connections.Elasticsearch.connect",
        return_value=MagicMock(),
    ) as mock_connect:
        yield mock_connect


@pytest.fixture
def elasticsearch_document_retriever(mock_elasticsearch_store):
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_elasticsearch_store)
    return retriever


@patch.object(ElasticsearchDocumentRetriever, "connect_to_vector_store")
def test_initialization_with_defaults(mock_connect_to_vector_store):
    mock_elasticsearch_store = MagicMock(spec=ElasticsearchVectorStore)
    mock_connect_to_vector_store.return_value = mock_elasticsearch_store

    retriever = ElasticsearchDocumentRetriever()

    mock_connect_to_vector_store.assert_called_once()
    assert retriever.vector_store == mock_elasticsearch_store


def test_initialization_with_vector_store(mock_elasticsearch_store):
    retriever = ElasticsearchDocumentRetriever(vector_store=mock_elasticsearch_store)
    assert retriever.vector_store == mock_elasticsearch_store
    assert retriever.connection is None


def test_vector_store_cls(elasticsearch_document_retriever):
    assert elasticsearch_document_retriever.vector_store_cls == ElasticsearchVectorStore


def test_to_dict_exclude_params(elasticsearch_document_retriever):
    exclude_params = elasticsearch_document_retriever.to_dict_exclude_params
    assert "document_retriever" in exclude_params


def test_init_components(elasticsearch_document_retriever, mock_elasticsearch_store):
    connection_manager = MagicMock(spec=ConnectionManager)
    elasticsearch_document_retriever.init_components(connection_manager)
    assert isinstance(
        elasticsearch_document_retriever.document_retriever,
        ElasticsearchDocumentRetrieverComponent,
    )
    assert elasticsearch_document_retriever.document_retriever.vector_store == mock_elasticsearch_store


def test_execute_basic_search(elasticsearch_document_retriever):
    """Test basic vector similarity search."""
    input_data = ElasticsearchRetrieverInputSchema(query=[0.1, 0.2, 0.3], filters={"field": "value"}, top_k=5)
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    elasticsearch_document_retriever.document_retriever = MagicMock(spec=ElasticsearchDocumentRetrieverComponent)
    elasticsearch_document_retriever.document_retriever.run.return_value = mock_output

    result = elasticsearch_document_retriever.execute(input_data, config)

    elasticsearch_document_retriever.document_retriever.run.assert_called_once_with(
        query=input_data.query,
        filters=input_data.filters,
        top_k=input_data.top_k,
        exclude_document_embeddings=input_data.exclude_document_embeddings,
        scale_scores=input_data.scale_scores,
    )

    assert result == {"documents": mock_output["documents"]}


def test_execute_with_missing_query(elasticsearch_document_retriever):
    """Test that validation error is raised when query is missing."""
    config = RunnableConfig(callbacks=[])

    with pytest.raises(ValidationError):
        elasticsearch_document_retriever.execute(ElasticsearchRetrieverInputSchema(), config)


def test_execute_with_default_filters_and_top_k(elasticsearch_document_retriever):
    """Test search with default filters and top_k values."""
    input_data = ElasticsearchRetrieverInputSchema(query=[0.1, 0.2, 0.3])
    config = RunnableConfig(callbacks=[])

    mock_output = {"documents": [{"id": "1", "content": "Document 1"}]}
    elasticsearch_document_retriever.document_retriever = MagicMock(spec=ElasticsearchDocumentRetrieverComponent)
    elasticsearch_document_retriever.document_retriever.run.return_value = mock_output

    result = elasticsearch_document_retriever.execute(input_data, config)

    elasticsearch_document_retriever.document_retriever.run.assert_called_once_with(
        query=input_data.query,
        filters=elasticsearch_document_retriever.filters,
        top_k=elasticsearch_document_retriever.top_k,
        exclude_document_embeddings=input_data.exclude_document_embeddings,
        scale_scores=input_data.scale_scores,
    )

    assert result == {"documents": mock_output["documents"]}


def test_elasticsearch_specific_params(elasticsearch_document_retriever):
    """Test Elasticsearch-specific parameters like similarity metric."""
    assert hasattr(elasticsearch_document_retriever, "similarity")
    assert elasticsearch_document_retriever.similarity == "cosine"  # Default value


def test_elasticsearch_dimension_param(elasticsearch_document_retriever):
    """Test the dimension parameter specific to Elasticsearch vector store."""
    assert hasattr(elasticsearch_document_retriever, "dimension")
    assert elasticsearch_document_retriever.dimension == 1536  # Default value


def test_custom_initialization():
    """Test initialization with custom Elasticsearch parameters."""
    retriever = ElasticsearchDocumentRetriever(index_name="custom_index", dimension=768, similarity="dot_product")
    assert retriever.index_name == "custom_index"
    assert retriever.dimension == 768
    assert retriever.similarity == "dot_product"
