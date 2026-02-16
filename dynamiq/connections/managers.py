import enum
import hashlib
import importlib
import threading
from contextlib import contextmanager
from typing import Any

from dynamiq.components.serializers import JsonPickleSerializer
from dynamiq.connections import BaseConnection
from dynamiq.utils.logger import logger


class ConnectionManagerException(Exception):
    """Exception raised for errors in the ConnectionManager."""
    pass


class ConnectionClientInitType(str, enum.Enum):
    """Enumeration of connection client initialization types."""
    DEFAULT = "DEFAULT"


CONNECTION_METHOD_BY_INIT_TYPE = {
    ConnectionClientInitType.DEFAULT: "connect",
}


class ConnectionManager:
    """
    Manages connections to various services and databases.

    This class handles the creation, retrieval, and management of connection clients
    for different types of services and databases.

    Attributes:
        serializer: An object used for serializing and deserializing data.
        connection_clients: A dictionary storing initialized connection clients.
    """

    def __init__(self, serializer: Any | None = None):
        """
        Initializes the ConnectionManager.

        Args:
            serializer: An optional serializer object. If not provided, JsonPickleSerializer is used.
        """
        self.serializer = serializer or JsonPickleSerializer()
        self.connection_clients: dict[str, Any] = {}
        self._connection_locks_guard = threading.Lock()
        self._connection_locks: dict[str, threading.Lock] = {}

    @staticmethod
    def get_connection_by_type(conn_type: str) -> type[BaseConnection]:
        """
        Retrieves the connection class based on the given connection type.

        Args:
            conn_type: The type of connection to retrieve.

        Returns:
            The connection class corresponding to the given type.

        Raises:
            ValueError: If the connection type is not found.
        """
        try:
            entity_module, entity_name = conn_type.rsplit(".", 1)
            imported_module = importlib.import_module(entity_module)
            if entity := getattr(imported_module, entity_name, None):
                return entity
        except (ModuleNotFoundError, ImportError):
            raise ValueError(f"Connection type {conn_type} not found")

    def _get_conn_lock(self, conn_id: str) -> threading.Lock:
        """Get or create a per-connection lock for the given conn_id."""
        with self._connection_locks_guard:
            if conn_id not in self._connection_locks:
                self._connection_locks[conn_id] = threading.Lock()
            return self._connection_locks[conn_id]

    def _is_client_alive(self, conn_client: Any) -> bool:
        """Check if an existing connection client is still usable."""
        conn_client_closed = getattr(conn_client, "closed", None)
        if conn_client_closed is None:
            return True
        return not conn_client_closed

    def get_connection_client(
        self,
        connection: BaseConnection,
        init_type: ConnectionClientInitType = ConnectionClientInitType.DEFAULT,
    ) -> Any | None:
        """
        Retrieves or initializes a connection client for the given connection.

        Thread-safe: uses per-connection locks so different connections can be
        created in parallel, while the same connection is never created twice concurrently.

        Args:
            connection: The connection object.
            init_type: The initialization type for the connection client.

        Returns:
            The initialized connection client.

        Raises:
            ConnectionManagerException: If the connection does not support the specified initialization type.
        """
        logger.debug(
            f"Get connection client for '{connection.id}-{connection.type}' "
            f"with '{init_type.value.lower()}' initialization"
        )
        conn_id = self.get_connection_id(connection, init_type)

        if conn_client := self.connection_clients.get(conn_id):
            if self._is_client_alive(conn_client):
                return conn_client

        conn_lock = self._get_conn_lock(conn_id)
        with conn_lock:
            logger.info(f"Acquired lock for '{conn_id}' connection for initialization")
            # Double-check after acquiring lock (another thread may have created it)
            if conn_client := self.connection_clients.get(conn_id):
                if self._is_client_alive(conn_client):
                    return conn_client

            logger.debug(
                f"Init connection client for '{connection.id}-{connection.type}' "
                f"with '{init_type.value.lower()}' initialization"
            )
            conn_method_name = CONNECTION_METHOD_BY_INIT_TYPE[init_type]
            if not (conn_method := getattr(connection, conn_method_name, None)) or not callable(conn_method):
                raise ConnectionManagerException(
                    f"Connection '{connection.id}-{connection.type}' not support '{init_type.value}' initialization"
                )

            conn_client = conn_method()
            self.connection_clients[conn_id] = conn_client

            return conn_client

    def get_connection_id(
        self,
        connection: BaseConnection,
        init_type: ConnectionClientInitType = ConnectionClientInitType.DEFAULT,
    ) -> str:
        """
        Generates a unique connection ID based on the connection and initialization type.

        Args:
            connection: The connection object.
            init_type: The initialization type for the connection client.

        Returns:
            A unique string identifier for the connection.
        """
        conn_hash = self.hash(connection.model_dump_json())
        return f"{connection.type.lower()}:{init_type.lower()}:{conn_hash}"

    def close(self):
        """
        Closes all open connection clients and clears the connection_clients dictionary.
        """
        logger.debug("Close connection clients")
        for conn_client in self.connection_clients.values():
            if hasattr(conn_client, "close"):
                conn_client.close()
        self.connection_clients = {}

    @staticmethod
    def hash(data: str) -> str:
        """
        Generates a SHA256 hash of the input string.

        Args:
            data: The input string to hash.

        Returns:
            The hexadecimal representation of the SHA256 hash.
        """
        return hashlib.sha256(data.encode()).hexdigest()

    def dumps(self, data: Any):
        """
        Serializes the given data using the serializer.

        Args:
            data: The data to serialize.

        Returns:
            The serialized data.
        """
        return self.serializer.dumps(data)

    def loads(self, value: str):
        """
        Deserializes the given value using the serializer.

        Args:
            value: The serialized string to deserialize.

        Returns:
            The deserialized data.
        """
        return self.serializer.loads(value)


@contextmanager
def get_connection_manager():
    """
    A context manager that yields a ConnectionManager instance and ensures it's closed properly.

    Yields:
        A ConnectionManager instance.
    """
    cm = ConnectionManager()
    yield cm
    cm.close()
