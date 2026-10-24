from unittest.mock import MagicMock, patch

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

        self.search = MagicMock()
        self.hybrid_search = MagicMock()

        self._tenant_name = kwargs.get("tenant_name")
        self._multi_tenancy_enabled = self._tenant_name is not None

        self.index_name = kwargs.get("index_name", "test_collection")


def test_execute_with_tenant():
    """Test execute with tenant configuration."""
    mock_store = MockWeaviateVectorStore(tenant_name="test_tenant")

    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_store)

    mock_component = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    mock_component.run.return_value = {
        "documents": [
            {"id": "1", "content": "Document 1", "metadata": {"source": "test"}, "score": 0.95},
            {"id": "2", "content": "Document 2", "metadata": {"source": "test"}, "score": 0.85},
        ]
    }
    retriever.document_retriever = mock_component

    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], top_k=5)
    config = RunnableConfig(callbacks=[])

    result = retriever.execute(input_data, config)

    mock_component.run.assert_called_once()

    assert "documents" in result
    assert len(result["documents"]) == 2
    assert result["documents"][0]["id"] == "1"
    assert result["documents"][1]["id"] == "2"


def test_hybrid_retrieval_with_tenant():
    """Test hybrid retrieval with tenant configuration."""
    mock_store = MockWeaviateVectorStore(tenant_name="test_tenant")
    mock_store.hybrid_search.return_value = [
        Document(id="1", content="Document 1", metadata={"source": "test"}, score=0.95),
        Document(id="2", content="Document 2", metadata={"source": "test"}, score=0.85),
    ]

    with patch("dynamiq.nodes.retrievers.weaviate.WeaviateDocumentRetriever.init_components"):
        retriever = WeaviateDocumentRetriever(vector_store=mock_store)

    mock_component = MagicMock(spec=WeaviateDocumentRetrieverComponent)
    mock_component.run.return_value = {
        "documents": [
            {"id": "1", "content": "Document 1", "metadata": {"source": "test"}, "score": 0.95},
            {"id": "2", "content": "Document 2", "metadata": {"source": "test"}, "score": 0.85},
        ]
    }
    retriever.document_retriever = mock_component

    input_data = RetrieverInputSchema(embedding=[0.1, 0.2, 0.3], query="test query", alpha=0.7, top_k=5)
    config = RunnableConfig(callbacks=[])

    result = retriever.execute(input_data, config)

    mock_component.run.assert_called_once()
    args, kwargs = mock_component.run.call_args
    assert args[0] == input_data.embedding
    assert kwargs["query"] == "test query"
    assert kwargs["alpha"] == 0.7
    assert kwargs["top_k"] == 5

    assert "documents" in result
    assert len(result["documents"]) == 2
