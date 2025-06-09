from dynamiq.connections.connections import Whisper
from dynamiq.connections.managers import ConnectionManager


def test_connection_manager_caches_client_on_first_connect_and_reuses():
    cm = ConnectionManager()
    connection = Whisper(id="test", api_key="test_key_123")
    connection_another_instance = Whisper(id="test", api_key="test_key_123")

    assert connection.headers == {"Authorization": f"Bearer {connection.api_key}"}
    assert connection_another_instance.headers == {"Authorization": f"Bearer {connection_another_instance.api_key}"}

    client1 = cm.get_connection_client(connection)
    assert client1 is not None
    assert len(cm.connection_clients) == 1

    client2 = cm.get_connection_client(connection)
    assert client1 is client2
    assert len(cm.connection_clients) == 1

    client3 = cm.get_connection_client(connection_another_instance)
    assert client1 is client3
    assert len(cm.connection_clients) == 1

    client4 = cm.get_connection_client(connection_another_instance)
    assert client1 is client4
    assert len(cm.connection_clients) == 1

    cm.close()
    assert len(cm.connection_clients) == 0
