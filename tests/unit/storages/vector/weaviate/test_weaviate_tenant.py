from unittest.mock import MagicMock, create_autospec, patch

from weaviate.collections.data import _DataCollection

from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.weaviate import WeaviateVectorStore
from dynamiq.types import Document


def test_initialization_with_tenant():
    """Test initialization with tenant name."""
    with patch("dynamiq.connections.Weaviate.connect") as mock_connect:
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        with patch.object(WeaviateVectorStore, "_update_multi_tenancy_status"), patch.object(
            WeaviateVectorStore, "_setup_collection"
        ):
            vector_store = WeaviateVectorStore(  # noqa: F841
                index_name="TestCollection", create_if_not_exist=True, tenant_name="tenant1"
            )

            mock_client.collections.create.assert_called_once()
            create_args = mock_client.collections.create.call_args[1]
            assert "multi_tenancy_config" in create_args
            assert create_args["multi_tenancy_config"].enabled is True


def test_create_vector_store_with_collection_only():
    """Test creating a vector store with just a collection name (no tenant)."""
    with patch("dynamiq.connections.Weaviate.connect") as mock_connect:
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        mock_client.collections.exists.return_value = False

        mock_collection = MagicMock()
        mock_client.collections.create.return_value = mock_collection
        mock_client.collections.get.return_value = mock_collection

        with patch.object(WeaviateVectorStore, "_update_multi_tenancy_status"), patch.object(
            WeaviateVectorStore, "_setup_collection"
        ):
            vector_store = WeaviateVectorStore(index_name="TestCollection", create_if_not_exist=True)  # noqa: F841

            mock_client.collections.create.assert_called_once()
            create_args = mock_client.collections.create.call_args[1]
            assert "multi_tenancy_config" not in create_args


def test_write_documents_with_a_duplicate_policy_writes_to_the_tenant():
    client = MagicMock()
    collection = client.collections.get.return_value
    collection.config.get.return_value.multi_tenancy_config.enabled = True
    tenant_collection = collection.with_tenant.return_value
    # Unlike a plain mock, an autospec rejects arguments that the client's insert does not take.
    tenant_collection.data = create_autospec(_DataCollection, instance=True)
    tenant_collection.data.exists.return_value = False
    store = WeaviateVectorStore(client=client, index_name="Docs", tenant_name="tenant1")

    documents = [Document(id="1", content="One", metadata={"source": "a"}, embedding=[0.1, 0.2])]

    assert store.write_documents(documents, policy=DuplicatePolicy.SKIP) == 1
    collection.with_tenant.assert_called_once_with("tenant1")
    tenant_collection.data.insert.assert_called_once()
