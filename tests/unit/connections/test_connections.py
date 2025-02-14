from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections.connections import Milvus as MilvusConnection
from dynamiq.connections.connections import MilvusDeploymentType
from dynamiq.connections.connections import Qdrant as QdrantConnection


@pytest.fixture
def mock_qdrant_env_vars(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://mocked_qdrant_url")
    monkeypatch.setenv("QDRANT_API_KEY", "mocked_api_key")


def test_qdrant_initialization_with_env_vars(mock_qdrant_env_vars):
    qdrant = QdrantConnection()
    assert qdrant.url == "http://mocked_qdrant_url"
    assert qdrant.api_key == "mocked_api_key"


def test_qdrant_initialization_with_provided_values():
    qdrant = QdrantConnection(url="http://custom_qdrant_url", api_key="custom_api_key")
    assert qdrant.url == "http://custom_qdrant_url"
    assert qdrant.api_key == "custom_api_key"


@patch("qdrant_client.QdrantClient")
def test_qdrant_connect(mock_qdrant_client_class, mock_qdrant_env_vars):
    mock_qdrant_client_instance = MagicMock()
    mock_qdrant_client_class.return_value = mock_qdrant_client_instance

    qdrant = QdrantConnection()
    client = qdrant.connect()

    mock_qdrant_client_class.assert_called_once_with(url="http://mocked_qdrant_url", api_key="mocked_api_key")
    assert client == mock_qdrant_client_instance


@pytest.fixture
def mock_milvus_env_vars(monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://mocked_milvus_url")
    monkeypatch.setenv("MILVUS_API_TOKEN", "mocked_api_key")


def test_milvus_initialization_with_env_vars(mock_milvus_env_vars):
    milvus = MilvusConnection()
    assert milvus.uri == "http://mocked_milvus_url"
    assert milvus.api_key == "mocked_api_key"


def test_milvus_initialization_with_provided_values():
    milvus = MilvusConnection(
        deployment_type=MilvusDeploymentType.HOST, uri="http://custom_milvus_url", api_key="custom_api_key"
    )
    assert milvus.uri == "http://custom_milvus_url"
    assert milvus.api_key == "custom_api_key"


@patch("pymilvus.MilvusClient")
def test_milvus_connect_file(mock_milvus_client_class):
    mock_milvus_client_instance = MagicMock()
    mock_milvus_client_class.return_value = mock_milvus_client_instance

    milvus = MilvusConnection(deployment_type=MilvusDeploymentType.FILE, uri="path/to/milvus.db")
    client = milvus.connect()

    mock_milvus_client_class.assert_called_once_with(uri="path/to/milvus.db")
    assert client == mock_milvus_client_instance


@patch("pymilvus.MilvusClient")
def test_milvus_connect_host_with_token(mock_milvus_client_class):
    mock_milvus_client_instance = MagicMock()
    mock_milvus_client_class.return_value = mock_milvus_client_instance

    milvus = MilvusConnection(
        deployment_type=MilvusDeploymentType.HOST, uri="https://cloud.milvus.io", api_key="mocked_api_key"
    )
    client = milvus.connect()

    mock_milvus_client_class.assert_called_once_with(uri="https://cloud.milvus.io", token="mocked_api_key")
    assert client == mock_milvus_client_instance


@patch("pymilvus.MilvusClient")
def test_milvus_connect_host_without_token(mock_milvus_client_class):
    mock_milvus_client_instance = MagicMock()
    mock_milvus_client_class.return_value = mock_milvus_client_instance

    milvus = MilvusConnection(deployment_type=MilvusDeploymentType.HOST, uri="http://localhost:19530")
    client = milvus.connect()

    mock_milvus_client_class.assert_called_once_with(uri="http://localhost:19530")
    assert client == mock_milvus_client_instance


def test_milvus_connect_file_invalid_uri():
    with pytest.raises(ValueError, match="For FILE deployment, URI should point to a file ending with '.db'"):
        MilvusConnection(deployment_type=MilvusDeploymentType.FILE, uri="not_a_db_path")
