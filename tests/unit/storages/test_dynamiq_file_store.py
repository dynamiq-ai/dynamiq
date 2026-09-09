import base64
import json
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.storages.file import DynamiqFileStore
from dynamiq.storages.file.base import FileExistsError, FileNotFoundError, PermissionError, StorageError


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
    return DynamiqFileStore(connection=connection, memory_store_id="ms-123", user="u-42")


def _mock_response(payload=None, status_code=200, content=b"{}"):
    response = MagicMock()
    response.status_code = status_code
    response.content = content
    response.text = json.dumps(payload) if payload is not None else ""
    response.json.return_value = payload
    return response


def _file_payload(path="memories/notes.md", content=b"hello", **overrides):
    payload = {
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "size": len(content),
        "content_type": "text/markdown",
        "created_at": "2026-01-15T10:30:00Z",
        "metadata": {"source": "agent"},
        "content": base64.b64encode(content).decode("ascii"),
    }
    payload.update(overrides)
    return payload


def test_type_is_package_path(store):
    assert store.type == "dynamiq.storages.file.DynamiqFileStore"


def test_store_sends_base64_content_and_user(store, client):
    client.request.return_value = _mock_response({"data": _file_payload()})

    info = store.store("memories/notes.md", "hello", metadata={"source": "agent"}, overwrite=True)

    verb, url = client.request.call_args.args
    body = client.request.call_args.kwargs["json"]
    assert verb == "PUT"
    assert url == "https://api.example.ai/v1/memory-stores/ms-123/files"
    assert base64.b64decode(body["content"]) == b"hello"
    assert body["user"] == "u-42"
    assert body["overwrite"] is True
    assert body["metadata"] == {"source": "agent"}
    assert client.request.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert info.path == "memories/notes.md"
    assert info.size == 5


def test_retrieve_decodes_base64_and_unwraps_data_envelope(store, client):
    client.request.return_value = _mock_response({"data": _file_payload(content=b"remembered")})

    assert store.retrieve("memories/notes.md") == b"remembered"

    verb, url = client.request.call_args.args
    assert verb == "GET"
    assert url.endswith("/v1/memory-stores/ms-123/files/content")
    assert client.request.call_args.kwargs["params"] == {"path": "memories/notes.md", "user": "u-42"}


def test_exists_reads_flag(store, client):
    client.request.return_value = _mock_response({"data": {"exists": True}})
    assert store.exists("memories/notes.md") is True

    client.request.return_value = _mock_response({"data": {"exists": False}})
    assert store.exists("memories/gone.md") is False


def test_exists_is_false_when_api_reports_not_found(store, client):
    client.request.return_value = _mock_response(status_code=404)
    assert store.exists("memories/gone.md") is False


def test_delete_returns_false_when_missing(store, client):
    client.request.return_value = _mock_response(status_code=404)
    assert store.delete("memories/gone.md") is False


def test_delete_returns_true(store, client):
    client.request.return_value = _mock_response({"data": {"deleted": True}})
    assert store.delete("memories/notes.md") is True
    assert client.request.call_args.args[0] == "DELETE"


def test_list_files_forwards_pattern_and_parses_entries(store, client):
    client.request.return_value = _mock_response({"data": [_file_payload(), _file_payload(path="memories/b.md")]})

    files = store.list_files(directory="memories/", recursive=True, pattern="*.md")

    assert [f.path for f in files] == ["memories/notes.md", "memories/b.md"]
    assert client.request.call_args.kwargs["params"] == {
        "path": "memories/",
        "recursive": True,
        "pattern": "*.md",
        "user": "u-42",
    }


def test_list_files_parses_created_at_and_metadata(store, client):
    client.request.return_value = _mock_response({"data": [_file_payload()]})

    info = store.list_files()[0]

    assert info.name == "notes.md"
    assert info.content_type == "text/markdown"
    assert info.metadata == {"source": "agent"}
    assert info.created_at.year == 2026


def test_list_files_bytes_without_paths_does_not_call_the_api(store, client):
    """The agent calls this with no arguments on every tool invocation; it must not fan out."""
    assert store.list_files_bytes() == []
    assert store.list_files_bytes(None) == []
    client.request.assert_not_called()


def test_list_files_bytes_with_paths_retrieves_each(store, client):
    client.request.return_value = _mock_response({"data": _file_payload(content=b"data")})

    files = store.list_files_bytes(["memories/notes.md"])

    assert len(files) == 1
    assert files[0].read() == b"data"
    assert files[0].name == "memories/notes.md"


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (404, FileNotFoundError),
        (403, PermissionError),
        (409, FileExistsError),
        (500, StorageError),
    ],
)
def test_status_codes_map_to_storage_exceptions(store, client, status_code, expected):
    client.request.return_value = _mock_response(status_code=status_code)
    with pytest.raises(expected):
        store.retrieve("memories/notes.md")


def test_transport_failure_raises_storage_error(store, client):
    client.request.side_effect = ConnectionError("boom")
    with pytest.raises(StorageError):
        store.retrieve("memories/notes.md")


def test_invalid_json_raises_storage_error(store, client):
    response = _mock_response()
    response.json.side_effect = ValueError("not json")
    client.request.return_value = response
    with pytest.raises(StorageError):
        store.retrieve("memories/notes.md")


def test_user_is_omitted_when_unset(connection, client):
    store = DynamiqFileStore(connection=connection, memory_store_id="ms-123")
    client.request.return_value = _mock_response({"data": {"exists": False}})

    store.exists("memories/notes.md")

    assert "user" not in client.request.call_args.kwargs["params"]


def test_to_dict_withholds_credentials_by_default(store):
    serialized = json.dumps(store.to_dict())
    assert "secret-token" not in serialized
    assert store.to_dict()["memory_store_id"] == "ms-123"


def test_to_dict_includes_credentials_only_when_asked(store):
    assert "secret-token" in json.dumps(store.to_dict(include_secure_params=True))
    assert "secret-token" not in json.dumps(store.to_dict(for_tracing=True))


def test_extracted_text_cache_is_declined(store):
    """Caching converter output remotely would cost a write per read and litter the namespace."""
    assert store.supports_extracted_text_cache("memories/report.pdf") is False
