import json
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.storages.memory import DynamiqMemoryStore, MemoryNotFoundError, MemoryPermissionError, MemoryStoreError


@pytest.fixture
def connection():
    return Dynamiq(url="https://api.example.ai/", api_key="secret-token")


@pytest.fixture
def client(monkeypatch):
    """Patch Dynamiq.connect on the class - the connection is a frozen pydantic model."""
    client = MagicMock()
    monkeypatch.setattr(Dynamiq, "connect", lambda self: client)
    return client


@pytest.fixture
def store(connection, client):
    return DynamiqMemoryStore(connection=connection, memory_store_id="ms-123", user_id="u-42")


def _mock_response(payload=None, status_code=200, content=b"{}"):
    response = MagicMock()
    response.status_code = status_code
    response.content = content
    response.text = json.dumps(payload) if payload is not None else ""
    response.json.return_value = payload
    return response


def test_user_id_is_required(connection):
    with pytest.raises(ValueError):
        DynamiqMemoryStore(connection=connection, memory_store_id="ms-123")


def test_type_is_package_path(store):
    assert store.type == "dynamiq.storages.memory.DynamiqMemoryStore"


def test_write_sends_plain_text_and_user_id(store, client):
    client.request.return_value = _mock_response({"data": {"path": "prefs.md", "size": 24}})

    entry = store.write("prefs.md", "Prefers British English.")

    verb, url = client.request.call_args.args
    body = client.request.call_args.kwargs["json"]
    assert verb == "PUT"
    assert url == "https://api.example.ai/v1/memory-stores/ms-123/files"
    assert body == {"path": "prefs.md", "content": "Prefers British English.", "user_id": "u-42"}
    assert client.request.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert entry.path == "prefs.md"


def test_read_returns_text_from_the_data_envelope(store, client):
    client.request.return_value = _mock_response({"data": {"path": "prefs.md", "content": "remembered"}})

    assert store.read("prefs.md") == "remembered"

    verb, url = client.request.call_args.args
    assert verb == "GET"
    assert url.endswith("/v1/memory-stores/ms-123/files/content")
    assert client.request.call_args.kwargs["params"] == {"path": "prefs.md", "user_id": "u-42"}


def test_read_raises_when_absent(store, client):
    client.request.return_value = _mock_response(status_code=404)

    with pytest.raises(MemoryNotFoundError):
        store.read("gone.md")


def test_read_raises_when_the_server_returns_an_empty_envelope(store, client):
    client.request.return_value = _mock_response({"data": None})

    with pytest.raises(MemoryNotFoundError):
        store.read("gone.md")


def test_list_parses_entries(store, client):
    client.request.return_value = _mock_response(
        {"data": [{"path": "prefs.md", "size": 24, "updated_at": "2026-01-15T10:30:00Z"}, {"path": "team/x.md"}]}
    )

    entries = store.list("")

    assert [entry.path for entry in entries] == ["prefs.md", "team/x.md"]
    assert entries[0].size == 24
    assert entries[0].updated_at.year == 2026
    assert entries[1].updated_at is None
    assert client.request.call_args.kwargs["params"] == {"path": "", "user_id": "u-42"}


def test_list_forwards_the_prefix_as_path(store, client):
    client.request.return_value = _mock_response({"data": []})

    assert store.list("team/") == []
    assert client.request.call_args.kwargs["params"] == {"path": "team/", "user_id": "u-42"}


def test_delete_returns_true(store, client):
    client.request.return_value = _mock_response({"data": {"deleted": True}})

    assert store.delete("prefs.md") is True
    assert client.request.call_args.args[0] == "DELETE"
    assert client.request.call_args.kwargs["params"] == {"path": "prefs.md", "user_id": "u-42"}


def test_delete_returns_false_when_absent(store, client):
    """A 404 on delete means it was not there, which is not an error."""
    client.request.return_value = _mock_response(status_code=404)

    assert store.delete("gone.md") is False


def test_delete_tolerates_an_empty_body(store, client):
    client.request.return_value = _mock_response(status_code=204, content=b"")

    assert store.delete("prefs.md") is True


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(404, MemoryNotFoundError), (403, MemoryPermissionError), (413, MemoryStoreError), (500, MemoryStoreError)],
)
def test_status_codes_map_to_memory_exceptions(store, client, status_code, expected):
    client.request.return_value = _mock_response(status_code=status_code)

    with pytest.raises(expected):
        store.read("prefs.md")


def test_transport_failure_raises_memory_store_error(store, client):
    client.request.side_effect = ConnectionError("boom")

    with pytest.raises(MemoryStoreError):
        store.read("prefs.md")


def test_invalid_json_raises_memory_store_error(store, client):
    response = _mock_response()
    response.json.side_effect = ValueError("not json")
    client.request.return_value = response

    with pytest.raises(MemoryStoreError):
        store.read("prefs.md")


def test_description_reaches_describe_namespaces(connection, client):
    store = DynamiqMemoryStore(
        connection=connection, memory_store_id="ms-123", user_id="u-42", description="About this user."
    )

    assert store.describe_namespaces() == {"": "About this user."}


def test_to_dict_withholds_credentials_by_default(store):
    assert "secret-token" not in json.dumps(store.to_dict())
    assert store.to_dict()["memory_store_id"] == "ms-123"
    assert store.to_dict()["user_id"] == "u-42"


def test_to_dict_includes_credentials_only_when_asked(store):
    assert "secret-token" in json.dumps(store.to_dict(include_secure_params=True))
    assert "secret-token" not in json.dumps(store.to_dict(for_tracing=True))
