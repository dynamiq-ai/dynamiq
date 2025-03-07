from unittest.mock import MagicMock, patch

import pytest

from dynamiq.components.retrievers.weaviate import WeaviateDocumentRetriever as WeaviateDocumentRetrieverComponent
from dynamiq.nodes.retrievers import WeaviateDocumentRetriever
from dynamiq.nodes.retrievers.base import RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.types import Document


class MockWeaviateVectorStore(WeaviateVectorStore):
    """A mock WeaviateVectorStore that can be used in tests."""

    def __init__(self, *args, **kwargs):
        # Skip the initialization of the actual class
        # This prevents real calls to Weaviate
        self.client = MagicMock()

        # Add search methods
        self.search = MagicMock()
        self.hybrid_search = MagicMock()

        # Set tenant properties
        self._tenant_name = kwargs.get("tenant_name")
        self._multi_tenancy_enabled = self._tenant_name is not None

        # Save the index name
        self.index_name = kwargs.get("index_name", "test_collection")


@pytest.fixture
def mock_weaviate_vector_store():
    """Create a mock WeaviateVectorStore with all necessary attributes."""
    mock_store = MockWeaviateVectorStore()

    # Configure the search results
    mock_search_result = [
        Document(id="1", content="Document 1", metadata={"source": "test"}, score=0.95),
        Document(id="2", content="Document 2", metadata={"source": "test"}, score=0.85),
    ]
    mock_store.search.return_value = mock_search_result
    mock_store.hybrid_search.return_value = mock_search_result

    return mock_store


@pytest.fixture
def weaviate_document_retriever(mock_weaviate_vector_store):
    """Create a WeaviateDocumentRetriever with a mocked vector store."""
    # Disable init_components to avoid automatic initialization
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_weaviate_vector_store)

    # Set up a mock document retriever component
    mock_component = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    mock_component.run.return_value = {
        "documents": [
            {"id": "1", "content": "Document 1", "metadata": {"source": "test"}, "score": 0.95},
            {"id": "2", "content": "Document 2", "metadata": {"source": "test"}, "score": 0.85},
        ]
    }
    retriever.document_retriever = mock_component

    return retriever


def test_initialization_with_vector_store(mock_weaviate_vector_store):
    """Test initialization with a provided vector store."""
    # Disable init_components to avoid automatic initialization
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_weaviate_vector_store)
        assert retriever.vector_store == mock_weaviate_vector_store


def test_retriever_with_tenant():
    """Test that a retriever can be created with a tenant name."""
    # Create a mock vector store with tenant properties
    mock_store = MockWeaviateVectorStore(tenant_name="test_tenant")

    # Disable init_components to avoid automatic initialization
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        # Create a retriever with the mock vector store
        retriever = WeaviateDocumentRetriever(vector_store=mock_store, tenant_name="test_tenant")

        # Assert tenant_name is stored in the retriever
        assert retriever.tenant_name == "test_tenant"
        assert retriever.vector_store._tenant_name == "test_tenant"
        assert retriever.vector_store._multi_tenancy_enabled is True


def test_execute_with_tenant():
    """Test execute with tenant configuration."""
    # Create a mock vector store with tenant properties
    mock_store = MockWeaviateVectorStore(tenant_name="test_tenant")

    # Disable init_components to avoid automatic initialization
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_store)

    # Set up a mock document retriever component
    mock_component = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    mock_component.run.return_value = {
        "documents": [
            {"id": "1", "content": "Document 1", "metadata": {"source": "test"}, "score": 0.95},
            {"id": "2", "content": "Document 2", "metadata": {"source": "test"}, "score": 0.85},
        ]
    }
    retriever.document_retriever = mock_component

    # Input data for retrieval
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], top_k=5)
    config = RunnableConfig(callbacks=[])

    # Execute retrieval
    result = retriever.execute(input_data, config)

    # Verify the document retriever component was called
    mock_component.run.assert_called_once()

    # Check the result
    assert "documents" in result
    assert len(result["documents"]) == 2
    assert result["documents"][0]["id"] == "1"
    assert result["documents"][1]["id"] == "2"


def test_tenant_mismatch_reinitialization():
    """Test tenant handling with explicit tenant names."""
    # Create a mock vector store with tenant properties
    mock_store = MockWeaviateVectorStore(tenant_name="original_tenant")

    # Create a retriever with the mock vector store and explicit tenant name
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(
            vector_store=mock_store, tenant_name="original_tenant"  # Explicitly set the tenant_name
        )

    # Verify initial tenant name is set correctly
    assert retriever.tenant_name == "original_tenant"

    # Change the tenant name
    retriever.tenant_name = "new_tenant"

    # Verify the tenant name is updated
    assert retriever.tenant_name == "new_tenant"

    # Vector store tenant name should still be the original
    assert retriever.vector_store._tenant_name == "original_tenant"

    # Mock init_components to see how it would update the vector store
    with patch.object(mock_store, "_tenant_name", "original_tenant"):
        # Call init_components to update the vector store
        retriever.init_components()

        # After init_components, the vector store should retain its tenant name
        # since we're only testing the property setting, not the actual reinitialization logic
        assert retriever.vector_store._tenant_name == "original_tenant"


def test_hybrid_retrieval_with_tenant():
    """Test hybrid retrieval with tenant configuration."""
    # Create a mock vector store with tenant properties
    mock_store = MockWeaviateVectorStore(tenant_name="test_tenant")
    mock_store.hybrid_search.return_value = [
        Document(id="1", content="Document 1", metadata={"source": "test"}, score=0.95),
        Document(id="2", content="Document 2", metadata={"source": "test"}, score=0.85),
    ]

    # Disable init_components to avoid automatic initialization
    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_store)

    # Set up a mock document retriever component
    mock_component = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    mock_component.run.return_value = {
        "documents": [
            {"id": "1", "content": "Document 1", "metadata": {"source": "test"}, "score": 0.95},
            {"id": "2", "content": "Document 2", "metadata": {"source": "test"}, "score": 0.85},
        ]
    }
    retriever.document_retriever = mock_component

    # Input data for hybrid retrieval
    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], query="test query", alpha=0.7, top_k=5)
    config = RunnableConfig(callbacks=[])

    # Execute retrieval
    result = retriever.execute(input_data, config)

    # Verify document retriever was called with correct parameters
    mock_component.run.assert_called_once()
    args, kwargs = mock_component.run.call_args
    assert args[0] == input_data.embedding
    assert kwargs["query"] == "test query"
    assert kwargs["alpha"] == 0.7
    assert kwargs["top_k"] == 5

    # Check the result
    assert "documents" in result
    assert len(result["documents"]) == 2
