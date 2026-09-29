import json
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import Dynamiq
from dynamiq.nodes.tools.file_tools import EditOperation
from dynamiq.storages.artifact import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactPermissionError,
    ArtifactStoreConfig,
    ArtifactStoreError,
    DynamiqArtifactStore,
    infer_kind,
)

ARTIFACT = {
    "id": "a1",
    "name": "q3.html",
    "title": "Q3 report",
    "kind": "html",
    "media_type": "text/html",
    "url": "https://app.example/artifacts/a1",
    "latest": {"id": "v2", "version": 2, "size": 11, "checksum": "sha256:abc", "url": "https://x/a1/v/2"},
}


@pytest.fixture
def client(monkeypatch):
    """Patch Dynamiq.connect on the class - the connection is a frozen pydantic model."""
    client = MagicMock()
    monkeypatch.setattr(Dynamiq, "connect", lambda self: client)
    return client


@pytest.fixture
def store(client):
    return DynamiqArtifactStore(connection=Dynamiq(url="https://api.example.ai/", api_key="secret-token"))


def _mock_response(payload=None, status_code=200, content=None):
    response = MagicMock()
    response.status_code = status_code
    response.text = json.dumps(payload) if payload is not None else ""
    response.content = content if content is not None else response.text.encode()
    response.json.return_value = payload
    return response


def test_type_is_package_path(store):
    assert store.type == "dynamiq.storages.artifact.DynamiqArtifactStore"


def test_create_text_sends_json(store, client):
    client.request.return_value = _mock_response({"data": ARTIFACT})

    artifact = store.create(
        name="q3.html", title="Q3 report", kind=ArtifactKind.HTML, content="<html/>", source={"run_id": "r1"}
    )

    verb, url = client.request.call_args.args
    kwargs = client.request.call_args.kwargs
    assert (verb, url) == ("POST", "https://api.example.ai/v1/artifacts")
    assert kwargs["json"] == {
        "name": "q3.html",
        "title": "Q3 report",
        "kind": "html",
        "media_type": "text/html",
        "source": {"run_id": "r1"},
        "content": "<html/>",
    }
    assert kwargs["files"] is None
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert artifact.id == "a1" and artifact.version == 2
    assert artifact.to_ref() == {
        "id": "a1",
        "version": 2,
        "name": "q3.html",
        "title": "Q3 report",
        "kind": "html",
        "media_type": "text/html",
        "size": 11,
        "url": "https://app.example/artifacts/a1",
    }


def test_create_binary_sends_multipart_without_a_json_content_type(store, client):
    client.request.return_value = _mock_response({"data": {**ARTIFACT, "kind": "pdf", "name": "r.pdf"}})

    store.create(name="r.pdf", title="R", kind=ArtifactKind.PDF, content=b"%PDF", metadata={"a": 1})

    kwargs = client.request.call_args.kwargs
    assert kwargs["json"] is None
    assert kwargs["files"] == {"file": ("r.pdf", b"%PDF", "application/pdf")}
    assert kwargs["data"]["metadata"] == json.dumps({"a": 1})
    assert "Content-Type" not in kwargs["headers"]


def test_project_id_owns_created_artifacts(client):
    store = DynamiqArtifactStore(connection=Dynamiq(url="https://api.example.ai", api_key="k"), project_id="p1")
    client.request.return_value = _mock_response({"data": ARTIFACT})

    store.create(name="q3.html", title="Q3", kind="html", content="x")

    assert client.request.call_args.kwargs["json"]["project_id"] == "p1"


def test_update_sends_edits_and_if_match(store, client):
    client.request.return_value = _mock_response({"data": ARTIFACT})

    store.update("a1", edits=[EditOperation(find="Q2", replace="Q3")], summary="fix label", if_match="sha256:abc")

    verb, url = client.request.call_args.args
    kwargs = client.request.call_args.kwargs
    assert (verb, url) == ("PUT", "https://api.example.ai/v1/artifacts/a1")
    assert kwargs["json"] == {
        "summary": "fix label",
        "edits": [{"find": "Q2", "replace": "Q3", "replace_all": False}],
    }
    assert kwargs["headers"]["If-Match"] == "sha256:abc"


def test_update_with_nothing_is_refused_locally(store, client):
    with pytest.raises(ArtifactStoreError, match="Nothing to update"):
        store.update("a1")
    client.request.assert_not_called()


def test_get_fetches_metadata_then_decodes_text_content(store, client):
    client.request.side_effect = [_mock_response({"data": ARTIFACT}), _mock_response(content=b"<html>hi</html>")]

    artifact, content = store.get("a1")

    urls = [call.args[1] for call in client.request.call_args_list]
    assert urls == [
        "https://api.example.ai/v1/artifacts/a1",
        "https://api.example.ai/v1/artifacts/a1/versions/2/content",
    ]
    assert content == "<html>hi</html>"
    assert artifact.title == "Q3 report"


def test_get_keeps_binary_content_as_bytes(store, client):
    client.request.side_effect = [
        _mock_response({"data": {**ARTIFACT, "kind": "pdf"}}),
        _mock_response(content=b"%PDF-1.7"),
    ]

    _, content = store.get("a1", version=1)

    assert content == b"%PDF-1.7"
    assert client.request.call_args_list[0].kwargs["params"] == {"version": 1}
    assert client.request.call_args_list[1].args[1].endswith("/versions/1/content")


def test_get_without_content_makes_one_call(store, client):
    client.request.return_value = _mock_response({"data": ARTIFACT})

    _, content = store.get("a1", include_content=False)

    assert content is None
    assert client.request.call_count == 1


def test_list_sends_filters(store, client):
    client.request.return_value = _mock_response({"data": [ARTIFACT]})

    artifacts = store.list(kind=ArtifactKind.HTML, query="q3", limit=5)

    assert client.request.call_args.kwargs["params"] == {
        "page_size": 5,
        "sort": "-updated_at",
        "kind": "html",
        "query": "q3",
    }
    assert [a.id for a in artifacts] == ["a1"]


@pytest.mark.parametrize(
    "status, error",
    [
        (404, ArtifactNotFoundError),
        (403, ArtifactPermissionError),
        (409, ArtifactConflictError),
        (412, ArtifactConflictError),
        (500, ArtifactStoreError),
    ],
)
def test_status_codes_map_to_errors(store, client, status, error):
    client.request.return_value = _mock_response({"message": "no"}, status_code=status)

    with pytest.raises(error):
        store.update("a1", content="x")


def test_transport_failure_is_an_artifact_error(store, client):
    client.request.side_effect = ConnectionError("boom")

    with pytest.raises(ArtifactStoreError, match="boom"):
        store.list()


def test_missing_base_url_is_refused(client):
    store = DynamiqArtifactStore(connection=Dynamiq(url="", api_key="k"))

    with pytest.raises(ArtifactStoreError, match="base URL"):
        store.list()


def test_serialization_withholds_credentials_unless_asked(store):
    config = ArtifactStoreConfig(enabled=True, backend=store)

    assert "secret-token" not in json.dumps(config.to_dict(), default=str)
    assert "secret-token" in json.dumps(config.to_dict(include_secure_params=True), default=str)
    assert config.to_dict()["backend"]["type"] == "dynamiq.storages.artifact.DynamiqArtifactStore"


@pytest.mark.parametrize(
    "name, kind",
    [
        ("r.html", ArtifactKind.HTML),
        ("notes.md", ArtifactKind.MARKDOWN),
        ("chart.svg", ArtifactKind.SVG),
        ("flow.mmd", ArtifactKind.MERMAID),
        ("data.csv", ArtifactKind.CSV),
        ("main.py", ArtifactKind.CODE),
        ("shot.png", ArtifactKind.IMAGE),
        ("deck.pptx", ArtifactKind.FILE),
    ],
)
def test_infer_kind_from_extension(name, kind):
    assert infer_kind(name) == kind
