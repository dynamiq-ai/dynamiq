from unittest.mock import Mock

import pytest

from dynamiq.connections import PostgreSQL
from dynamiq.connections.managers import ConnectionManager, ConnectionManagerException
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, VectorStoreNode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.pgvector.pgvector import PGVectorStoreWriterParams


class MockClosedClient:
    """Mock client that can be closed."""

    def __init__(self, closed=False):
        self.closed = closed

    def close(self):
        self.closed = True


class MockClientWithMethod:
    """Mock client with is_closed() method."""

    def __init__(self, closed=False):
        self._closed = closed

    def is_closed(self):
        return self._closed

    def close(self):
        self._closed = True


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self, client=None, **kwargs):
        self.client = client
        # Accept any kwargs to match VectorStore interface

    def write_documents(self, documents, **kwargs):
        if self.client is None or getattr(self.client, "closed", False):
            raise ConnectionError("Vector store connection closed")
        return len(documents)


@pytest.fixture
def mock_connection():
    """Create a mock PostgreSQL connection."""
    return PostgreSQL(
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_pass",
    )


@pytest.fixture
def mock_open_client():
    """Create a mock open client."""
    return MockClosedClient(closed=False)


@pytest.fixture
def mock_closed_client():
    """Create a mock closed client."""
    return MockClosedClient(closed=True)


@pytest.fixture
def mock_connection_manager(mocker):
    """Create a mock ConnectionManager."""
    cm = Mock(spec=ConnectionManager)
    cm.get_connection_client = Mock(return_value=MockClosedClient(closed=False))
    return cm


@pytest.fixture
def success_result():
    return {"result": "success"}


@pytest.fixture
def connection_node_instance(mock_connection, mock_open_client, mock_connection_manager, success_result):
    """Create a ConnectionNode instance for testing."""
    from dynamiq.nodes import NodeGroup

    class TestConnectionNode(ConnectionNode):
        """Test implementation of ConnectionNode."""

        group: str = NodeGroup.TOOLS

        def execute(self, input_data, config=None, **kwargs):
            if self.client is None or getattr(self.client, "closed", False):
                raise ConnectionError("Client connection closed")
            return success_result

    node = TestConnectionNode(
        id="test-node",
        name="TestNode",
        connection=mock_connection,
        client=mock_open_client,
    )
    node._connection_manager = mock_connection_manager
    return node


@pytest.fixture
def vector_store_node_instance(mock_connection, mock_open_client, mock_connection_manager, success_result):
    """Create a VectorStoreNode instance for testing."""
    from dynamiq.nodes import NodeGroup

    class TestVectorStoreNode(VectorStoreNode, PGVectorStoreWriterParams):
        """Test implementation of VectorStoreNode."""

        group: str = NodeGroup.WRITERS

        @property
        def vector_store_cls(self):
            return MockVectorStore

        def execute(self, input_data, config=None, **kwargs):
            if self.vector_store is None:
                raise ValueError("Vector store not initialized")
            return success_result

    # Provide client and vector_store directly to avoid init_components trying to connect
    vector_store = MockVectorStore(client=mock_open_client)
    node = TestVectorStoreNode(
        id="test-vector-node",
        name="TestVectorNode",
        connection=mock_connection,
        client=mock_open_client,
        vector_store=vector_store,
    )
    node._connection_manager = mock_connection_manager
    return node


@pytest.mark.parametrize(
    "client,expected_closed,test_id",
    [
        (None, False, "client_is_none"),
        (MockClosedClient(closed=True), True, "closed_client_with_attribute"),
        (MockClosedClient(closed=False), False, "open_client_with_attribute"),
        (MockClientWithMethod(closed=True), True, "closed_client_with_method"),
        (MockClientWithMethod(closed=False), False, "open_client_with_method"),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_is_client_closed(connection_node_instance, client, expected_closed, test_id):
    connection_node_instance.client = client
    assert connection_node_instance.is_client_closed() is expected_closed


def test_ensure_client_does_nothing_when_client_is_none(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = None
    connection_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_not_called()


def test_ensure_client_does_nothing_when_client_is_open(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = MockClosedClient(closed=False)
    connection_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_not_called()


def test_ensure_client_reinitializes_when_client_is_closed(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = MockClosedClient(closed=True)
    new_client = MockClosedClient(closed=False)
    mock_connection_manager.get_connection_client.return_value = new_client

    connection_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_called_once()
    assert connection_node_instance.client == new_client
    assert connection_node_instance.client.closed is False


def test_ensure_client_raises_exception_on_reinitialization_failure(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = MockClosedClient(closed=True)
    mock_connection_manager.get_connection_client.side_effect = Exception("Connection failed")

    with pytest.raises(ConnectionManagerException) as exc_info:
        connection_node_instance.ensure_client()

    assert "Failed to reinitialize client" in str(exc_info.value)


def test_ensure_client_uses_cached_connection_manager(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = MockClosedClient(closed=True)
    connection_node_instance._connection_manager = mock_connection_manager

    connection_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_called_once_with(
        connection=connection_node_instance.connection
    )


def test_ensure_client_skips_reinitialization_when_no_connection(connection_node_instance, mock_connection_manager):
    """Test that ensure_client doesn't attempt reinitialization when connection is None."""
    connection_node_instance.client = MockClosedClient(closed=True)
    connection_node_instance.connection = None

    connection_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_not_called()


@pytest.mark.parametrize(
    "client,expected_result,test_id",
    [
        (None, False, "none_vector_store_client"),
        (MockClosedClient(closed=True), True, "closed_vector_store_client"),
        (MockClosedClient(closed=False), False, "open_vector_store_client"),
    ],
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_vector_store_node_is_client_closed(vector_store_node_instance, client, expected_result, test_id):
    """Test that VectorStoreNode correctly checks vector store client state.

    Note: None client returns False (not initialized, not closed).
    """
    vector_store_node_instance.vector_store.client = client
    assert vector_store_node_instance.is_client_closed() is expected_result


def test_vector_store_node_ensure_client_does_nothing_when_client_is_none(
    vector_store_node_instance, mock_connection_manager
):
    vector_store_node_instance.vector_store.client = None
    vector_store_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_not_called()


def test_vector_store_node_ensure_client_reinitializes_both_client_and_vector_store(
    vector_store_node_instance, mock_connection_manager
):
    vector_store_node_instance.vector_store.client = MockClosedClient(closed=True)
    new_client = MockClosedClient(closed=False)
    mock_connection_manager.get_connection_client.return_value = new_client

    vector_store_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_called_once_with(
        connection=vector_store_node_instance.connection,
    )
    assert vector_store_node_instance.vector_store is not None
    assert vector_store_node_instance.client == new_client


def test_vector_store_node_ensure_client_skips_reinitialization_when_no_connection(
    vector_store_node_instance, mock_connection_manager
):
    """Test that VectorStoreNode ensure_client doesn't attempt reinitialization when connection is None."""
    vector_store_node_instance.vector_store.client = MockClosedClient(closed=True)
    vector_store_node_instance.connection = None

    vector_store_node_instance.ensure_client()

    mock_connection_manager.get_connection_client.assert_not_called()


def test_execute_with_retry_calls_ensure_client_before_execution(connection_node_instance, mocker, success_result):
    ensure_client_mock = mocker.patch.object(type(connection_node_instance), "ensure_client", return_value=None)

    config = RunnableConfig()
    result = connection_node_instance.execute_with_retry(input_data={"test": "data"}, config=config)

    ensure_client_mock.assert_called_once()
    assert result == success_result


def test_execute_with_retry_reinitializes_on_connection_error(
    connection_node_instance, mock_connection_manager, mocker, success_result
):
    connection_node_instance.client = MockClosedClient(closed=False)
    connection_node_instance.error_handling = ErrorHandling(max_retries=2, retry_interval_seconds=0.01)

    call_count = 0

    def execute_side_effect(self, input_data, config=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: close client and raise connection error
            connection_node_instance.client.close()
            raise ConnectionError("Connection closed unexpectedly")
        else:
            # Second call: should succeed after reinitialization
            return success_result

    mocker.patch.object(type(connection_node_instance), "execute", execute_side_effect)
    new_client = MockClosedClient(closed=False)
    mock_connection_manager.get_connection_client.return_value = new_client

    config = RunnableConfig()
    result = connection_node_instance.execute_with_retry(input_data={"test": "data"}, config=config)

    assert result == success_result
    assert mock_connection_manager.get_connection_client.call_count >= 1
    assert call_count == 2


def test_execute_with_retry_handles_ensure_client_failure(connection_node_instance, mock_connection_manager):
    connection_node_instance.client = MockClosedClient(closed=True)
    connection_node_instance.error_handling = ErrorHandling(max_retries=2, retry_interval_seconds=0.01)

    mock_connection_manager.get_connection_client.side_effect = Exception("Connection unavailable")

    config = RunnableConfig()

    with pytest.raises(ConnectionManagerException) as exc_info:
        connection_node_instance.execute_with_retry(input_data={"test": "data"}, config=config)

    assert "Failed to reinitialize client" in str(exc_info.value)


def test_execute_with_retry_succeeds_after_reconnection(
    connection_node_instance, mock_connection_manager, mocker, success_result
):
    connection_node_instance.client = MockClosedClient(closed=False)
    connection_node_instance.error_handling = ErrorHandling(max_retries=3, retry_interval_seconds=0.01)

    attempt_count = 0

    def execute_side_effect(self, input_data, config=None, **kwargs):
        nonlocal attempt_count
        attempt_count += 1

        if attempt_count == 1:
            # First attempt: simulate connection failure
            connection_node_instance.client.close()
            raise ConnectionError("Connection closed")
        elif attempt_count == 2:
            # Second attempt: still closed (before ensure_client reinitializes)
            if connection_node_instance.client.closed:
                raise ConnectionError("Still closed")
            return success_result
        else:
            # Third attempt: should work after reinitialization
            return success_result

    mocker.patch.object(type(connection_node_instance), "execute", execute_side_effect)

    new_client = MockClosedClient(closed=False)
    mock_connection_manager.get_connection_client.return_value = new_client

    config = RunnableConfig()
    result = connection_node_instance.execute_with_retry(input_data={"test": "data"}, config=config)

    assert result == success_result
    assert attempt_count >= 2


def test_connection_manager_caching_in_init_components(
    mock_connection, mock_connection_manager, mock_open_client, success_result
):
    from dynamiq.nodes import NodeGroup

    class TestConnectionNode(ConnectionNode):
        group: str = NodeGroup.TOOLS

        def execute(self, input_data, config=None, **kwargs):
            return success_result

    node = TestConnectionNode(id="test-node", name="TestNode", connection=mock_connection, client=mock_open_client)
    node.init_components(connection_manager=mock_connection_manager)

    assert node._connection_manager == mock_connection_manager


def test_vector_store_node_caching_in_init_components(
    mock_connection, mock_connection_manager, mock_open_client, success_result
):
    from dynamiq.nodes import NodeGroup

    class TestVectorStoreNode(VectorStoreNode):
        group: str = NodeGroup.WRITERS

        @property
        def vector_store_cls(self):
            return MockVectorStore

        def execute(self, input_data, config=None, **kwargs):
            return success_result

    vector_store = MockVectorStore(client=mock_open_client)
    node = TestVectorStoreNode(
        id="test-node", name="TestNode", connection=mock_connection, client=mock_open_client, vector_store=vector_store
    )

    node.init_components(connection_manager=mock_connection_manager)

    assert node._connection_manager == mock_connection_manager
