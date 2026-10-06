import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dynamiq.artifacts import (
    ArtifactConfig,
    ArtifactConflictError,
    ArtifactError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactPermissionError,
    default_mime_type,
    infer_kind,
    version_mime_type,
)
from dynamiq.artifacts.backends import Dynamiq
from dynamiq.connections import Dynamiq as DynamiqConnection

BASE = "https://api.example.ai/v1/artifacts"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

VERSION = {
    "id": "v2",
    "artifact_id": "a1",
    "version": 2,
    "name": "Q3 report",
    "description": None,
    "mime_type": "text/html",
    "size": 11,
    "checksum": "sha256:abc",
    "entry_path": None,
    "created_at": "2026-10-01T12:00:00Z",
}

ARTIFACT = {
    "id": "a1",
    "org_id": "o1",
    "store_id": None,
    "user_id": None,
    "file_name": "q3.html",
    "name": "Q3 report",
    "kind": "html",
    "mime_type": "text/html",
    "visibility": "private",
    "latest_version_id": "v2",
    "latest_version": VERSION,
    "url": "https://app.example/artifacts/a1",
}


@pytest.fixture
def client(monkeypatch):
    """Patch connect on the class - the connection is a frozen pydantic model."""
    client = MagicMock()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: client)
    return client


def _backend(**kwargs) -> Dynamiq:
    return Dynamiq(connection=DynamiqConnection(url="https://api.example.ai/", api_key="secret-token"), **kwargs)


@pytest.fixture
def backend(client):
    return _backend()


def _response(payload=None, status_code=200, content=None):
    response = MagicMock()
    response.status_code = status_code
    response.text = json.dumps(payload) if payload is not None else ""
    response.content = content if content is not None else response.text.encode()
    response.json.return_value = payload
    return response


def _calls(client) -> list[tuple[str, str]]:
    return [tuple(call.args) for call in client.request.call_args_list]


def test_type_matches_the_platform_node(backend):
    assert backend.type == "dynamiq.artifacts.backends.Dynamiq"


def test_create_text_sends_json(backend, client):
    client.request.return_value = _response({"data": ARTIFACT}, status_code=201)

    artifact = backend.create(file_name="q3.html", name="Q3 report", kind=ArtifactKind.HTML, content="<html/>")

    assert _calls(client) == [("POST", BASE)]
    kwargs = client.request.call_args.kwargs
    assert kwargs["json"] == {
        "file_name": "q3.html",
        "name": "Q3 report",
        "kind": "html",
        "mime_type": "text/html",
        "content": "<html/>",
    }
    assert kwargs["files"] is None
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["headers"]["User-Agent"].startswith("dynamiq-python/")
    assert artifact.to_ref() == {
        "id": "a1",
        "version_id": "v2",
        "version": 2,
        "name": "Q3 report",
        "kind": "html",
        "url": "https://app.example/artifacts/a1",
    }


def test_create_in_a_store_names_the_store_and_end_user(client):
    backend = _backend(artifact_store_id="s1", user_id="customer-42")
    client.request.return_value = _response({"data": {**ARTIFACT, "store_id": "s1", "user_id": "customer-42"}})

    artifact = backend.create(file_name="q3.html", name="Q3", kind="html", content="x")

    kwargs = client.request.call_args.kwargs
    assert kwargs["json"]["store_id"] == "s1"
    assert kwargs["json"]["user_id"] == "customer-42"
    assert (artifact.store_id, artifact.user_id) == ("s1", "customer-42")


def test_create_binary_uploads_the_fields_as_one_json_part(backend, client):
    client.request.return_value = _response({"data": {**ARTIFACT, "kind": "pdf", "file_name": "r.pdf"}})

    backend.create(file_name="r.pdf", name="R", kind=ArtifactKind.PDF, content=b"%PDF", description="draft")

    assert _calls(client) == [("POST", f"{BASE}/upload")]
    kwargs = client.request.call_args.kwargs
    assert kwargs["json"] is None
    assert kwargs["files"] == {"file": ("r.pdf", b"%PDF", "application/pdf")}
    assert json.loads(kwargs["data"]["data"]) == {
        "file_name": "r.pdf",
        "name": "R",
        "description": "draft",
        "kind": "pdf",
        "mime_type": "application/pdf",
    }
    assert "Content-Type" not in kwargs["headers"]


def test_a_bundle_is_uploaded_with_its_entry_path(backend, client):
    client.request.return_value = _response({"data": {**ARTIFACT, "kind": "bundle", "file_name": "site.zip"}})

    backend.create(file_name="site.zip", name="Site", kind=ArtifactKind.BUNDLE, content=b"PK", entry_path="home.html")

    assert _calls(client) == [("POST", f"{BASE}/upload")]
    assert json.loads(client.request.call_args.kwargs["data"]["data"])["entry_path"] == "home.html"


def test_update_text_adds_a_version_with_if_match(backend, client):
    added = {**VERSION, "id": "v3", "version": 3, "name": "Q3 report, final", "description": "final numbers"}
    client.request.side_effect = [_response({"data": ARTIFACT}), _response({"data": added}, status_code=201)]

    artifact = backend.update(
        "a1", content="<html>v3</html>", name="Q3 report, final", description="final numbers", if_match="v2"
    )

    assert _calls(client) == [("GET", f"{BASE}/a1"), ("POST", f"{BASE}/a1/versions")]
    kwargs = client.request.call_args.kwargs
    assert kwargs["json"] == {"name": "Q3 report, final", "description": "final numbers", "content": "<html>v3</html>"}
    assert kwargs["headers"]["If-Match"] == '"v2"'
    assert (artifact.version, artifact.latest_version.id, artifact.name) == (3, "v3", "Q3 report, final")


def test_update_binary_uploads_a_version(backend, client):
    pdf = {**ARTIFACT, "kind": "pdf", "file_name": "r.pdf", "mime_type": "application/pdf"}
    client.request.side_effect = [_response({"data": pdf}), _response({"data": {**VERSION, "id": "v3", "version": 3}})]

    backend.update("a1", content=b"%PDF-2")

    assert _calls(client) == [("GET", f"{BASE}/a1"), ("POST", f"{BASE}/a1/versions/upload")]
    kwargs = client.request.call_args.kwargs
    assert kwargs["files"] == {"file": ("r.pdf", b"%PDF-2", "application/pdf")}
    assert json.loads(kwargs["data"]["data"]) == {}
    assert "If-Match" not in kwargs["headers"]


def test_get_downloads_the_latest_version_and_decodes_text(backend, client):
    client.request.side_effect = [_response({"data": ARTIFACT}), _response(content=b"<html>hi</html>")]

    artifact, content = backend.get("a1")

    assert _calls(client) == [("GET", f"{BASE}/a1"), ("GET", f"{BASE}/a1/versions/v2/download")]
    assert content == "<html>hi</html>"
    assert artifact.name == "Q3 report"


def test_get_an_earlier_version_resolves_its_id_and_keeps_bytes(backend, client):
    pdf = {**ARTIFACT, "kind": "pdf"}
    versions = [VERSION, {**VERSION, "id": "v1", "version": 1}]
    client.request.side_effect = [
        _response({"data": pdf}),
        _response({"data": versions}),
        _response(content=b"%PDF-1.7"),
    ]

    _, content = backend.get("a1", version=1)

    assert _calls(client) == [
        ("GET", f"{BASE}/a1"),
        ("GET", f"{BASE}/a1/versions"),
        ("GET", f"{BASE}/a1/versions/v1/download"),
    ]
    assert content == b"%PDF-1.7"


def test_get_a_version_that_is_gone(backend, client):
    client.request.side_effect = [_response({"data": ARTIFACT}), _response({"data": [VERSION]})]

    with pytest.raises(ArtifactNotFoundError, match="Version 1"):
        backend.get("a1", version=1)


def test_get_without_content_makes_one_call(backend, client):
    client.request.return_value = _response({"data": ARTIFACT})

    _, content = backend.get("a1", include_content=False)

    assert content is None
    assert client.request.call_count == 1


@pytest.mark.parametrize(
    "owner, payload",
    [
        ({"artifact_store_id": "s1"}, {"store_id": "s2"}),
        ({"artifact_store_id": "s1"}, {"store_id": None}),
        ({"artifact_store_id": "s1", "user_id": "customer-42"}, {"store_id": "s1", "user_id": "customer-7"}),
    ],
)
def test_an_artifact_of_another_store_or_end_user_is_not_found(client, owner, payload):
    """The platform does not check them on reads by id, so the backend does."""
    client.request.return_value = _response({"data": {**ARTIFACT, **payload}})

    with pytest.raises(ArtifactNotFoundError):
        _backend(**owner).get("a1")


def test_list_sends_the_owner_and_filters(client):
    backend = _backend(artifact_store_id="s1", user_id="customer-42")
    client.request.return_value = _response({"data": [ARTIFACT, {**ARTIFACT, "id": "a2"}]})

    artifacts = backend.list(kind=ArtifactKind.HTML, limit=1)

    assert client.request.call_args.kwargs["params"] == {
        "store_id": "s1",
        "user_id": "customer-42",
        "page_size": 10,
        "sort": "-updated_at",
        "kind": "html",
    }
    assert [a.id for a in artifacts] == ["a1"]


@pytest.mark.parametrize("configured", [None, "default-user"])
def test_a_calls_end_user_overrides_the_configured_one(client, configured):
    """An agent passes the run's user_id per call: one backend serves every end user of an app."""
    backend = _backend(artifact_store_id="s1", user_id=configured)
    client.request.return_value = _response({"data": {**ARTIFACT, "store_id": "s1", "user_id": "customer-a"}})

    backend.create(file_name="q3.html", name="Q3", kind=ArtifactKind.HTML, content="<html/>", user_id="customer-a")
    assert client.request.call_args.kwargs["json"]["user_id"] == "customer-a"

    backend.get("a1", include_content=False, user_id="customer-a")
    with pytest.raises(ArtifactNotFoundError):
        backend.get("a1", include_content=False, user_id="customer-b")

    client.request.return_value = _response({"data": []})
    backend.list(user_id="customer-a")
    assert client.request.call_args.kwargs["params"]["user_id"] == "customer-a"


def test_without_a_call_end_user_the_configured_one_applies(client):
    backend = _backend(artifact_store_id="s1", user_id="customer-42")
    client.request.return_value = _response({"data": []})

    backend.list()

    assert client.request.call_args.kwargs["params"]["user_id"] == "customer-42"


def test_outside_a_store_a_calls_end_user_does_not_apply(backend, client):
    """Without a store the artifacts belong to the token's user; a chat run's user_id names no end user."""
    client.request.return_value = _response({"data": ARTIFACT})

    backend.create(file_name="q3.html", name="Q3", kind=ArtifactKind.HTML, content="<html/>", user_id="chat-user")
    backend.get("a1", include_content=False, user_id="chat-user")

    assert "user_id" not in client.request.call_args_list[0].kwargs["json"]


def test_share_pins_a_version_and_an_expiry(backend, client):
    share = {"id": "sh1", "artifact_id": "a1", "pinned_version_id": "v2", "url": "https://app.example/a/sh1"}
    client.request.side_effect = [
        _response({"data": ARTIFACT}),
        _response({"data": [VERSION]}),
        _response({"data": {**share, "expires_at": "2026-11-01T00:00:00Z"}}),
    ]

    result = backend.share("a1", pinned_version=2, expires_at=datetime(2026, 11, 1))

    assert _calls(client)[-1] == ("POST", f"{BASE}/a1/share")
    assert client.request.call_args.kwargs["json"] == {
        "pinned_version_id": "v2",
        "expires_at": "2026-11-01T00:00:00+00:00",
    }
    assert result.url == "https://app.example/a/sh1"
    assert result.expires_at == datetime(2026, 11, 1, tzinfo=timezone.utc)


def test_unshare_deletes_the_share(backend, client):
    client.request.side_effect = [_response({"data": ARTIFACT}), _response({"message": "The object was deleted."})]

    backend.unshare("a1")

    assert _calls(client) == [("GET", f"{BASE}/a1"), ("DELETE", f"{BASE}/a1/share")]


@pytest.mark.parametrize(
    "status, error",
    [
        (404, ArtifactNotFoundError),
        (403, ArtifactPermissionError),
        (409, ArtifactConflictError),
        (412, ArtifactConflictError),
        (500, ArtifactError),
    ],
)
def test_status_codes_map_to_errors(backend, client, status, error):
    client.request.return_value = _response({"error": {"code": "x", "message": "no"}}, status_code=status)

    with pytest.raises(error):
        backend.list()


def test_a_platform_error_carries_its_message_and_details(backend, client):
    body = {
        "error": {
            "code": "bad_request",
            "message": "The request could not be processed due to invalid input.",
            "details": {"file_name": "cannot be blank"},
        }
    }
    client.request.return_value = _response(body, status_code=400)

    with pytest.raises(ArtifactError, match="invalid input.*file_name"):
        backend.create(file_name="", name="Q3", kind="html", content="x")


def test_transport_failure_is_an_artifact_error(backend, client):
    client.request.side_effect = ConnectionError("boom")

    with pytest.raises(ArtifactError, match="boom"):
        backend.list()


def test_missing_base_url_is_refused(client):
    backend = Dynamiq(connection=DynamiqConnection(url="", api_key="k"))

    with pytest.raises(ArtifactError, match="base URL"):
        backend.list()


def test_an_end_user_needs_a_store():
    with pytest.raises(ValidationError, match="'user_id' requires 'artifact_store_id'"):
        Dynamiq(connection=DynamiqConnection(url="https://api.example.ai", api_key="k"), user_id="customer-42")


def test_serialization_withholds_credentials_unless_asked(backend):
    config = ArtifactConfig(enabled=True, backend=backend)

    assert "secret-token" not in json.dumps(config.to_dict(), default=str)
    assert "secret-token" in json.dumps(config.to_dict(include_secure_params=True), default=str)
    assert config.to_dict()["backend"]["type"] == "dynamiq.artifacts.backends.Dynamiq"


@pytest.mark.parametrize(
    "file_name, kind",
    [
        ("r.html", ArtifactKind.HTML),
        ("notes.md", ArtifactKind.MARKDOWN),
        ("chart.svg", ArtifactKind.SVG),
        ("flow.mmd", ArtifactKind.MERMAID),
        ("data.csv", ArtifactKind.CSV),
        ("main.py", ArtifactKind.CODE),
        ("shot.png", ArtifactKind.IMAGE),
        ("deck.pptx", ArtifactKind.FILE),
        ("site.zip", ArtifactKind.FILE),
    ],
)
def test_infer_kind_from_extension(file_name, kind):
    assert infer_kind(file_name) == kind


@pytest.mark.parametrize(
    "file_name, host_guess",
    [
        ("main.rs", "application/rls-services+xml"),
        ("app.ts", "text/vnd.trolltech.linguist"),
        ("app.ts", "video/mp2t"),
        ("run.sh", "application/x-sh"),
        ("query.sql", "application/x-sql"),
    ],
)
def test_code_is_plain_text_whatever_the_table_guesses(file_name, host_guess, mocker):
    mocker.patch("dynamiq.artifacts.types._MIME_TYPES.guess_type", return_value=(host_guess, None))

    assert default_mime_type(ArtifactKind.CODE, file_name) == "text/plain"


@pytest.mark.parametrize(
    "kind, file_name, mime_type",
    [
        (ArtifactKind.FILE, "notes.docx", DOCX),
        (ArtifactKind.FILE, "q3.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (ArtifactKind.FILE, "deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        (ArtifactKind.FILE, "site.zip", "application/zip"),
        (ArtifactKind.FILE, "blob", "application/octet-stream"),
        (ArtifactKind.FILE, None, "application/octet-stream"),
        (ArtifactKind.IMAGE, "shot.webp", "image/webp"),
        (ArtifactKind.IMAGE, "shot.JPG", "image/jpeg"),
        (ArtifactKind.IMAGE, None, "image/png"),
    ],
)
def test_files_and_images_take_the_file_names_type(kind, file_name, mime_type):
    assert default_mime_type(kind, file_name) == mime_type


def test_the_hosts_mime_database_is_never_read(mocker):
    """A Mac's MIME files know .docx and a slim image has none; the type must not differ between them."""
    host = mocker.patch("dynamiq.artifacts.types.mimetypes.guess_type", return_value=("application/x-host", None))

    assert default_mime_type(ArtifactKind.FILE, "notes.docx") == DOCX
    assert default_mime_type(ArtifactKind.FILE, "blob.bin") == "application/octet-stream"
    host.assert_not_called()


def test_plain_text_is_not_taken_for_mermaid():
    assert default_mime_type(ArtifactKind.MERMAID) == "text/plain", "the platform's type for mermaid"
    assert infer_kind("notes", "text/plain") == ArtifactKind.FILE


@pytest.mark.parametrize(
    "kind, mime_type",
    [(ArtifactKind.FILE, DOCX), (ArtifactKind.CODE, "text/plain"), (ArtifactKind.HTML, None), (ArtifactKind.PDF, None)],
)
def test_only_kinds_typed_by_their_file_declare_a_versions_type(kind, mime_type):
    assert version_mime_type(kind, "notes.docx") == mime_type
