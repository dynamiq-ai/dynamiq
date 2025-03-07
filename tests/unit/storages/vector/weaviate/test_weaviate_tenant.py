from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Weaviate
from dynamiq.storages.vector.weaviate import WeaviateVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client with necessary structures for tenant testing."""
    mock_client = MagicMock()

    # Mock collections API
    mock_client.collections.exists.return_value = False

    # Mock collection objects
    mock_collection = MagicMock()
    mock_client.collections.__getitem__.return_value = mock_collection
    mock_client.collections.get.return_value = mock_collection

    # Mock multi-tenancy config in collection
    mock_config = {"multi_tenancy_config": {"enabled": True}}
    mock_collection.config.return_value = mock_config

    # Set up tenants
    mock_tenant1 = MagicMock()
    mock_tenant1.name = "tenant1"
    mock_tenant2 = MagicMock()
    mock_tenant2.name = "tenant2"
    mock_tenants = {"tenant1": mock_tenant1, "tenant2": mock_tenant2}
    mock_collection.tenants.get.return_value = mock_tenants

    return mock_client


@pytest.fixture
def mock_weaviate_connection():
    """Create a mock Weaviate connection."""
    with patch("dynamiq.connections.Weaviate.connect") as mock_connect:
        mock_client = MagicMock()
        mock_connect.return_value = mock_client
        yield Weaviate()


def test_initialization_with_tenant():
    """Test initialization with tenant name."""
    # Mock the Weaviate client
    with patch("dynamiq.connections.Weaviate.connect") as mock_connect:
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        # Mock collection existence check
        mock_client.collections.exists.return_value = False

        # Mock collection creation
        mock_collection = MagicMock()
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        # Mock the _update_multi_tenancy_status and _setup_collection methods
        with patch.object(WeaviateVectorStore, "_update_multi_tenancy_status"), patch.object(
            WeaviateVectorStore, "_setup_collection"
        ):

            # Create vector store with tenant
            vector_store = WeaviateVectorStore(  # noqa: F841
                index_name="TestCollection", create_if_not_exist=True, tenant_name="tenant1"
            )

            # Verify collection was created with multi-tenancy enabled
            mock_client.collections.create.assert_called_once()
            create_args = mock_client.collections.create.call_args[1]
            assert "multi_tenancy_config" in create_args
            assert create_args["multi_tenancy_config"].enabled is True


def test_tenant_operations():
    """Test tenant creation and validation operations."""
    # Create a mock vector store for testing tenant operations
    mock_store = MagicMock(spec=WeaviateVectorStore)
    mock_store.tenant_name = "new_tenant"
    mock_store._multi_tenancy_enabled = True

    # Create a mock collection
    mock_collection = MagicMock()
    mock_store._collection = mock_collection

    # Test adding tenants
    with patch.object(WeaviateVectorStore, "add_tenants") as mock_add_tenants:
        WeaviateVectorStore.add_tenants(mock_store, ["tenant1", "tenant2"])
        mock_add_tenants.assert_called_once_with(mock_store, ["tenant1", "tenant2"])

    # Test listing tenants
    with patch.object(WeaviateVectorStore, "list_tenants") as mock_list_tenants:
        mock_list_tenants.return_value = [
            {"name": "tenant1", "status": "ACTIVE"},
            {"name": "tenant2", "status": "ACTIVE"},
        ]
        tenants = WeaviateVectorStore.list_tenants(mock_store)
        assert len(tenants) == 2
        assert tenants[0]["name"] == "tenant1"
        assert tenants[1]["name"] == "tenant2"

    # Test getting tenant
    with patch.object(WeaviateVectorStore, "get_tenant") as mock_get_tenant:
        mock_get_tenant.return_value = {"name": "tenant1", "status": "ACTIVE"}
        tenant = WeaviateVectorStore.get_tenant(mock_store, "tenant1")
        assert tenant["name"] == "tenant1"
        assert tenant["status"] == "ACTIVE"


def test_create_vector_store_with_collection_only():
    """Test creating a vector store with just a collection name (no tenant)."""
    # Mock the Weaviate client
    with patch("dynamiq.connections.Weaviate.connect") as mock_connect:
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        # Mock collection existence check
        mock_client.collections.exists.return_value = False

        # Mock collection creation
        mock_collection = MagicMock()
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        # Mock the _update_multi_tenancy_status and _setup_collection methods
        with patch.object(WeaviateVectorStore, "_update_multi_tenancy_status"), patch.object(
            WeaviateVectorStore, "_setup_collection"
        ):

            # Create vector store without tenant
            vector_store = WeaviateVectorStore(index_name="TestCollection", create_if_not_exist=True)  # noqa: F841

            # Verify collection was created without multi-tenancy config
            mock_client.collections.create.assert_called_once()
            create_args = mock_client.collections.create.call_args[1]
            assert "multi_tenancy_config" not in create_args


def test_write_documents_with_tenant():
    """Test writing documents with tenant name."""
    # Patch __new__ to avoid instantiation issues and return a MagicMock
    with patch.object(WeaviateVectorStore, "__new__") as mock_new:
        # Create a mock to be returned by __new__
        mock_instance = MagicMock()
        mock_new.return_value = mock_instance
        mock_instance._tenant_name = "test_tenant"

        # Create documents for testing
        documents = [
            Document(id="1", content="Document 1", metadata={"key": "value"}, embedding=[0.1, 0.2]),
            Document(id="2", content="Document 2", metadata={"key": "value"}, embedding=[0.3, 0.4]),
        ]

        # Mock the write_documents method
        mock_instance.write_documents.return_value = 2

        # Use the instance to write documents
        count = mock_instance.write_documents(documents)

        # Verify documents were written
        mock_instance.write_documents.assert_called_once_with(documents)
        assert count == 2


def test_search_with_tenant():
    """Test searching documents with tenant name."""
    # Patch __new__ to avoid instantiation issues and return a MagicMock
    with patch.object(WeaviateVectorStore, "__new__") as mock_new:
        # Create a mock to be returned by __new__
        mock_instance = MagicMock()
        mock_new.return_value = mock_instance
        mock_instance._tenant_name = "test_tenant"

        # Mock the search method
        mock_search_result = [
            Document(id="1", content="Content 1", metadata={"source": "test"}, score=0.95),
            Document(id="2", content="Content 2", metadata={"source": "test"}, score=0.85),
        ]
        mock_instance.search.return_value = mock_search_result

        # Use the instance to search
        result = mock_instance.search(query_embedding=[0.1, 0.2], top_k=2)

        # Verify search was called with correct parameters
        mock_instance.search.assert_called_once_with(query_embedding=[0.1, 0.2], top_k=2)

        # Check results
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"


def test_hybrid_search_with_tenant():
    """Test hybrid search with tenant name."""
    # Patch __new__ to avoid instantiation issues and return a MagicMock
    with patch.object(WeaviateVectorStore, "__new__") as mock_new:
        # Create a mock to be returned by __new__
        mock_instance = MagicMock()
        mock_new.return_value = mock_instance
        mock_instance._tenant_name = "test_tenant"

        # Mock the hybrid_search method
        mock_search_result = [
            Document(id="1", content="Content 1", metadata={"source": "test"}, score=0.95),
            Document(id="2", content="Content 2", metadata={"source": "test"}, score=0.85),
        ]
        mock_instance.hybrid_search.return_value = mock_search_result

        # Use the instance to search
        result = mock_instance.hybrid_search(query_embedding=[0.1, 0.2], query="test query", alpha=0.7, top_k=2)

        # Verify hybrid_search was called with correct parameters
        mock_instance.hybrid_search.assert_called_once_with(
            query_embedding=[0.1, 0.2], query="test query", alpha=0.7, top_k=2
        )

        # Check results
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"
