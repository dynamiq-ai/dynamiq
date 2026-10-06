"""Integration tests for agent artifacts (OPENAI_API_KEY required).

``ArtifactAPISimulator`` stands in for the platform, so everything above the socket is real: the
same client, URLs, multipart bodies, If-Match headers and status codes. Each test gets a fresh one.
"""

import hashlib
import io
import json as json_lib
import mimetypes
import posixpath
import zipfile
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from dynamiq.artifacts import (
    TEXT_KINDS,
    ArtifactConfig,
    ArtifactConflictError,
    ArtifactError,
    ArtifactKind,
    ArtifactNotFoundError,
)
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools import ArtifactTool
from dynamiq.nodes.tools.artifact_tool import ArtifactToolInputSchema
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file import FileStoreConfig, InMemoryFileStore

MODEL = "gpt-5.4"
API_URL = "https://artifacts.simulated"
KINDS = {kind.value for kind in ArtifactKind}
TEXT_KIND_VALUES = {kind.value for kind in TEXT_KINDS}

# The platform's content rules (nexus artifacts/service/content.go): limits and MIME types per kind.
MAX_BYTES = {"html": 16 << 20, "bundle": 100 << 20}
DEFAULT_MAX_BYTES = 25 << 20
KIND_MIME_TYPES = {
    "html": "text/html",
    "markdown": "text/markdown",
    "svg": "image/svg+xml",
    "mermaid": "text/plain",
    "json": "application/json",
    "chart": "application/json",
    "csv": "text/csv",
    "pdf": "application/pdf",
    "bundle": "application/zip",
}
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

# Arbitrary on purpose: a model invents plausible metrics, so only these prove the record was read.
EMPLOYEE = "Dana Okafor"
MTTR = "37 minutes"
RATING = "Exceeds expectations"
REVIEW_REQUEST = f"""I'm finishing the Q3 performance reviews for the platform team and need the record \
for {EMPLOYEE} (Senior Platform Engineer) as a single HTML page I can share with HR and her.

Her Q3 numbers:
- Production deploys shipped: 142
- Incidents led as incident commander: 3, mean time to recovery {MTTR}
- Pull requests reviewed: 211
- Goal "Migrate the billing database to Postgres 16": completed two weeks early
- Goal "Mentor two new hires through onboarding": completed
- Manager rating: {RATING}

Include a short summary paragraph, a metrics table, the goals with their status, and the rating. \
Keep it clean and self-contained."""

READ_REQUEST = (
    f"HR is asking about {EMPLOYEE}'s Q3 performance record. What was her mean time to recovery on "
    "incidents, and what rating did she get? Answer from the record itself."
)


class _Response:
    """The subset of a `requests` response that the Dynamiq artifact backend reads."""

    def __init__(self, status_code: int, payload=None, raw: bytes | None = None):
        self.status_code = status_code
        self._payload = payload
        if raw is not None:
            self.content, self.text = raw, raw.decode(errors="replace")
        else:
            self.text = json_lib.dumps(payload) if payload is not None else ""
            self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> _Response:
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return _Response(status_code, {"error": error})


def _checksum(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _invalid_input(message: str) -> _Response:
    return _error(400, "bad_request", message)


def _check_content(kind: str, content: bytes, entry_path: str | None) -> tuple[_Response | None, str | None]:
    """Validate a version's bytes against the artifact's kind, as the platform does.

    Returns the error response, or None and the bundle's entry path (index.html unless given).
    """
    if len(content) > MAX_BYTES.get(kind, DEFAULT_MAX_BYTES):
        return _error(413, "payload_too_large", "The content is too large."), None
    if not content:
        return _invalid_input("The content is empty."), None
    if kind in TEXT_KIND_VALUES:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return _invalid_input("The content is not valid UTF-8 text."), None
        if kind in ("json", "chart"):
            try:
                spec = json_lib.loads(text)
            except ValueError:
                return _invalid_input("The content is not valid JSON."), None
            if kind == "chart" and "vega-lite" not in str((spec if isinstance(spec, dict) else {}).get("$schema")):
                return _invalid_input('A chart must be a Vega-Lite spec with a "$schema" naming Vega-Lite.'), None
    if kind == "image" and not (
        content.startswith(IMAGE_SIGNATURES) or (content[:4] == b"RIFF" and content[8:12] == b"WEBP")
    ):
        return _invalid_input("The content is not an image."), None
    if kind == "pdf" and not content.startswith(b"%PDF-"):
        return _invalid_input("The content is not a PDF."), None
    if kind != "bundle":
        return None, None
    try:
        names = [i.filename for i in zipfile.ZipFile(io.BytesIO(content)).infolist() if not i.is_dir()]
    except zipfile.BadZipFile:
        return _invalid_input("The bundle is not a valid zip archive."), None
    if any(n.startswith("/") or ".." in n.split("/") for n in names):
        return _invalid_input("The bundle has an entry outside the archive or a symbolic link."), None
    entry = posixpath.normpath(entry_path) if entry_path else "index.html"
    if entry not in names:
        return _invalid_input("The bundle has no file at entry_path."), None
    return None, entry


def _mime_type(kind: str, file_name: str, declared: str | None) -> str:
    """The declared type, else the one the kind fixes, else a guess from the file name."""
    return declared or KIND_MIME_TYPES.get(kind) or mimetypes.guess_type(file_name)[0] or "application/octet-stream"


class ArtifactAPISimulator:
    """In-process implementation of the platform's /v1/artifacts routes.

    Deliberately strict where the platform is: bearer token, known kinds, text kinds only as JSON,
    content validated against the artifact's kind, a MIME type and bundle entry page per version,
    immutable versions, a repeated upload returning the latest version, and If-Match on the latest
    version id.
    """

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.if_matches: list[str | None] = []

    # -- transport ---------------------------------------------------------
    def request(self, verb, url, headers=None, params=None, json=None, data=None, files=None, timeout=None):
        path = url.split("/v1/artifacts", 1)[1]
        self.calls.append((verb, path))
        headers, params = headers or {}, params or {}

        if not headers.get("Authorization"):
            return _error(401, "unauthorized", "The request requires authentication.")

        parts = [p for p in path.split("/") if p]
        client = headers.get("User-Agent")
        if verb == "POST" and not parts:
            return self._create(dict(json or {}), client, upload=False)
        if verb == "POST" and parts == ["upload"]:
            return self._create(self._upload_body(data, files), client, upload=True)
        if verb == "GET" and not parts:
            return self._list(params)
        if verb == "GET" and len(parts) == 1:
            return self._get(parts[0])
        if verb == "POST" and len(parts) == 2 and parts[1] == "versions":
            return self._add(parts[0], dict(json or {}), headers.get("If-Match"), client, upload=False)
        if verb == "POST" and len(parts) == 3 and parts[1:] == ["versions", "upload"]:
            return self._add(parts[0], self._upload_body(data, files), headers.get("If-Match"), client, upload=True)
        if verb == "GET" and len(parts) == 2 and parts[1] == "versions":
            return self._versions(parts[0])
        if verb == "GET" and len(parts) == 4 and parts[1] == "versions" and parts[3] == "download":
            return self._download(parts[0], parts[2])
        if verb in ("POST", "DELETE") and len(parts) == 2 and parts[1] == "share":
            return self._share(parts[0], dict(json or {}), revoke=verb == "DELETE")
        return _error(404, "not_found", f"no route for {verb} {path}")

    @staticmethod
    def _upload_body(data, files) -> dict:
        """The JSON fields of the `data` part, plus the file's name and bytes."""
        body = json_lib.loads((data or {}).get("data") or "{}")
        name, content, _ = files["file"]
        body["_file_name"] = name
        body["content"] = content if isinstance(content, bytes) else content.read()
        return body

    # -- endpoints ---------------------------------------------------------
    def _create(self, body, client, upload):
        file_name = body.get("file_name") or body.get("_file_name")
        missing = {k: "cannot be blank" for k in ("name", "kind") if not body.get(k)}
        if not file_name:
            missing["file_name"] = "cannot be blank"
        if not upload and not body.get("content"):
            missing["content"] = "cannot be blank"
        if missing:
            return _error(400, "bad_request", "The request could not be processed due to invalid input.", missing)
        if body["kind"] not in KINDS or (not upload and body["kind"] not in TEXT_KIND_VALUES):
            return _error(
                400,
                "bad_request",
                "The request could not be processed due to invalid input.",
                {"kind": "must be a valid value"},
            )
        content = body["content"].encode() if isinstance(body["content"], str) else body["content"]
        problem, entry_path = _check_content(body["kind"], content, body.get("entry_path"))
        if problem:
            return problem
        artifact_id = str(uuid4())
        self.artifacts[artifact_id] = {
            "id": artifact_id,
            "store_id": body.get("store_id"),
            "user_id": body.get("user_id"),
            "file_name": file_name,
            "name": body["name"],
            "kind": body["kind"],
            "visibility": "private",
            "versions": [],
            "share": None,
        }
        mime_type = _mime_type(body["kind"], file_name, body.get("mime_type"))
        self._record(artifact_id, content, body, client, mime_type, entry_path)
        return _Response(201, {"data": self._as_json(artifact_id)})

    def _add(self, artifact_id, body, if_match, client, upload):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        latest = artifact["versions"][-1]
        self.if_matches.append(if_match)
        if if_match and if_match.strip('"') != latest["id"]:
            return _error(
                412,
                "precondition_failed",
                f"The artifact has a newer version than If-Match. The latest version is {latest['version']}.",
            )
        content = body.get("content")
        if content is None:
            return _error(
                400,
                "bad_request",
                "The request could not be processed due to invalid input.",
                {"content": "cannot be blank"},
            )
        content = content.encode() if isinstance(content, str) else content
        # Every version is checked against the artifact's kind, which never changes, and a bundle
        # opens on index.html unless this version names another page.
        problem, entry_path = _check_content(artifact["kind"], content, body.get("entry_path"))
        if problem:
            return problem
        repeats = _checksum(content) == latest["checksum"] and all(
            body.get(field) is None or body[field] == latest[field]
            for field in ("name", "description", "mime_type", "entry_path")
        )
        if repeats:
            return _Response(200, {"data": self._version_json(artifact_id, latest)})
        mime_type = body.get("mime_type") or KIND_MIME_TYPES.get(artifact["kind"]) or latest["mime_type"]
        version = self._record(artifact_id, content, body, client, mime_type, entry_path)
        return _Response(201, {"data": self._version_json(artifact_id, version)})

    def _get(self, artifact_id):
        if artifact_id not in self.artifacts:
            return _error(404, "not_found", "The requested resource was not found.")
        return _Response(200, {"data": self._as_json(artifact_id)})

    def _versions(self, artifact_id):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        versions = [self._version_json(artifact_id, v) for v in reversed(artifact["versions"])]
        return _Response(200, {"data": versions})

    def _download(self, artifact_id, version_id):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        if version_id == "latest":
            return _Response(200, raw=artifact["versions"][-1]["content"])
        for version in artifact["versions"]:
            if version["id"] == version_id:
                return _Response(200, raw=version["content"])
        return _error(404, "not_found", "The requested resource was not found.")

    def _list(self, params):
        found = [a for a in self.artifacts.values() if a["store_id"] == params.get("store_id")]
        if params.get("user_id"):
            found = [a for a in found if a["user_id"] == params["user_id"]]
        if params.get("kind"):
            found = [a for a in found if a["kind"] == params["kind"]]
        found.sort(key=lambda a: a["versions"][-1]["created_at"], reverse=True)
        return _Response(200, {"data": [self._as_json(a["id"]) for a in found[: int(params.get("page_size", 25))]]})

    def _share(self, artifact_id, body, revoke):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        if revoke:
            artifact["share"], artifact["visibility"] = None, "private"
            return _Response(200, {"message": "The object was deleted."})
        share_id = (artifact["share"] or {}).get("id") or str(uuid4())
        artifact["share"] = {
            "id": share_id,
            "artifact_id": artifact_id,
            "pinned_version_id": body.get("pinned_version_id"),
            "expires_at": body.get("expires_at"),
            "url": f"https://app.simulated/a/{share_id}",
        }
        artifact["visibility"] = "link"
        return _Response(200, {"data": artifact["share"]})

    # -- helpers -----------------------------------------------------------
    def _record(
        self,
        artifact_id,
        content: bytes,
        body: dict,
        client: str | None,
        mime_type: str,
        entry_path: str | None,
    ) -> dict:
        artifact = self.artifacts[artifact_id]
        versions = artifact["versions"]
        version = {
            "id": str(uuid4()),
            "version": len(versions) + 1,
            "name": body.get("name") or artifact["name"],
            "description": body.get("description"),
            "mime_type": mime_type,
            "entry_path": entry_path,
            "content": content,
            "checksum": _checksum(content),
            "client": client,
            "created_at": _now(),
        }
        versions.append(version)
        artifact["name"] = version["name"]
        return version

    def _version_json(self, artifact_id, version):
        return {
            "id": version["id"],
            "artifact_id": artifact_id,
            "version": version["version"],
            "name": version["name"],
            "description": version["description"],
            "mime_type": version["mime_type"],
            "size": len(version["content"]),
            "checksum": version["checksum"],
            "entry_path": version["entry_path"],
            "source": {"client": version["client"]},
            "created_at": version["created_at"],
        }

    def _as_json(self, artifact_id):
        artifact = self.artifacts[artifact_id]
        latest = artifact["versions"][-1]
        return {
            **{k: artifact[k] for k in ("id", "store_id", "user_id", "file_name", "name", "kind")},
            "mime_type": latest["mime_type"],
            "visibility": artifact["visibility"],
            "latest_version_id": latest["id"],
            "latest_version": self._version_json(artifact_id, latest),
            "url": f"https://app.simulated/artifacts/{artifact_id}",
        }

    def latest_text(self, artifact_id) -> str:
        return self.artifacts[artifact_id]["versions"][-1]["content"].decode(errors="replace")


@pytest.fixture(scope="module")
def openai_llm():
    return OpenAI(model=MODEL, connection=OpenAIConnection())


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=240)


@pytest.fixture
def api(monkeypatch):
    """The artifact API, served by the simulator for the duration of one test."""
    simulator = ArtifactAPISimulator()
    monkeypatch.setattr(DynamiqConnection, "connect", lambda self: simulator)
    return simulator


@pytest.fixture
def backend(api):
    return DynamiqArtifacts(connection=DynamiqConnection(url=API_URL, api_key="simulated-key"))


@pytest.mark.integration
def test_backend_crud_against_the_api(backend, api):
    """Every route of the contract, without a model in the loop.

    Run this first when bringing a server up: it isolates the HTTP contract from agent behaviour,
    so a failure here is the server's, not the model's.
    """
    created = backend.create(file_name="probe.md", name="Probe", kind=ArtifactKind.MARKDOWN, content="Revenue Q2")
    assert created.version == 1 and created.url

    updated = backend.update(created.id, content="Revenue Q3", description="fix quarter")
    assert updated.version == 2

    artifact, latest = backend.get(created.id)
    assert (artifact.version, latest) == (2, "Revenue Q3")
    _, first = backend.get(created.id, version=1)
    assert first == "Revenue Q2", "Earlier versions must stay readable."

    image = backend.create(file_name="chart.png", name="Chart", kind=ArtifactKind.IMAGE, content=PNG)
    _, raw = backend.get(image.id)
    assert raw == PNG, "Binary content must round-trip as bytes."

    assert [a.id for a in backend.list(kind=ArtifactKind.MARKDOWN)] == [created.id]

    share = backend.share(created.id, pinned_version=1)
    assert share.url
    assert share.pinned_version_id == api.artifacts[created.id]["versions"][0]["id"]
    backend.unshare(created.id)
    assert api.artifacts[created.id]["visibility"] == "private"


@pytest.mark.integration
def test_backend_error_branches(backend, api):
    """The parts of the contract a server is most likely to leave out."""
    with pytest.raises(ArtifactNotFoundError):
        backend.get(str(uuid4()))

    created = backend.create(file_name="doc.md", name="Doc", kind=ArtifactKind.MARKDOWN, content="v1")
    stale = created.latest_version.id
    backend.update(created.id, content="v2 from a teammate")
    with pytest.raises(ArtifactConflictError, match="latest version is 2"):
        backend.update(created.id, content="v2 from the agent", if_match=stale)
    assert api.latest_text(created.id) == "v2 from a teammate", "A conflicting write must change nothing."

    with pytest.raises(ArtifactError, match="file_name"):
        backend.create(file_name="", name="Doc", kind=ArtifactKind.MARKDOWN, content="x")


def _zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def _tool_run(tool: ArtifactTool, **input_data) -> dict:
    return tool.execute(ArtifactToolInputSchema(**input_data))


SPEC = '{"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "mark": "%s"}'
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
DOCX = b"PK\x03\x04docx-bytes"
XLSX = b"PK\x03\x04xlsx-bytes"

# One case per kind: (file, v1, explicit kind, file published as v2 or None for the loaded copy, v2).
EVERY_KIND = [
    pytest.param("q3.html", "<!doctype html><p>v1</p>", None, None, "<!doctype html><p>v2</p>", id="html"),
    pytest.param("notes.md", "# Notes v1", None, None, "# Notes v2", id="markdown"),
    pytest.param("etl.py", "print('v1')\n", None, None, "print('v2')\n", id="code"),
    pytest.param("bars.svg", "<svg xmlns='http://www.w3.org/2000/svg'>v1</svg>", None, None, "<svg>v2</svg>", id="svg"),
    pytest.param("deploy.mmd", "flowchart LR\n  a --> b", None, None, "flowchart LR\n  a --> c", id="mermaid"),
    pytest.param("metrics.json", '{"version": 1}', None, None, '{"version": 2}', id="json"),
    pytest.param("data.csv", "quarter,revenue\nQ1,1.2\n", None, None, "quarter,revenue\nQ2,1.5\n", id="csv"),
    pytest.param("spec.json", SPEC % "bar", "chart", None, SPEC % "line", id="chart"),
    pytest.param("chart.png", PNG, None, "out/chart.jpg", JPEG, id="image-png-then-jpeg"),
    pytest.param("review.pdf", b"%PDF-1.7\nv1", None, None, b"%PDF-1.7\nv2", id="pdf"),
    pytest.param("deck.docx", DOCX, None, "out/numbers.xlsx", XLSX, id="file-docx-then-xlsx"),
    pytest.param(
        "site.zip",
        _zip({"report.html": "<p>v1</p>", "style.css": "p {}"}),
        "bundle",
        None,
        _zip({"report.html": "<p>v2</p>", "style.css": "p {}"}),
        id="bundle",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("file_name, v1, kind, v2_path, v2", EVERY_KIND)
def test_every_kind_is_created_loaded_and_updated_through_the_tool(api, backend, file_name, v1, kind, v2_path, v2):
    """The tool's requests for each kind are ones the platform accepts, and its bytes come back intact.

    No model: this pins the file-to-API path per kind, so a failure is the tool's or the contract's.
    """
    v1_bytes = v1.encode() if isinstance(v1, str) else v1
    v2_bytes = v2.encode() if isinstance(v2, str) else v2
    workspace = InMemoryFileStore()
    workspace.store(file_name, v1_bytes, overwrite=True)

    entry_path = "report.html" if kind == "bundle" else None
    created = _tool_run(
        ArtifactTool(backend=backend, workspace=workspace),
        action="create",
        path=file_name,
        name=f"{file_name} sample",
        kind=kind,
        entry_path=entry_path,
    )

    artifact_id = created["artifact"]["id"]
    record = api.artifacts[artifact_id]
    first = record["versions"][0]
    is_text = record["kind"] in TEXT_KIND_VALUES
    assert ("POST", "" if is_text else "/upload") in api.calls, "text kinds go as JSON, the rest as an upload"
    assert first["content"] == v1_bytes
    assert first["entry_path"] == entry_path

    # A later conversation: a fresh tool loads the artifact and publishes a changed file as v2.
    later = ArtifactTool(backend=backend, workspace=workspace)
    loaded = _tool_run(later, action="get", artifact_id=artifact_id)
    assert workspace.retrieve(loaded["path"]) == v1_bytes, "the loaded copy is the published bytes"

    target = v2_path or loaded["path"]
    workspace.store(target, v2_bytes, overwrite=True)
    updated = _tool_run(later, action="update", path=target, artifact_id=artifact_id, description="v2")

    assert updated["artifact"]["version"] == 2
    second = record["versions"][-1]
    assert second["content"] == v2_bytes
    assert api.if_matches[-1] == f'"{first["id"]}"', "built on the version loaded"
    assert ("POST", f"/{artifact_id}/versions" + ("" if is_text else "/upload")) in api.calls
    if kind == "bundle":
        assert second["entry_path"] == "report.html", "a bundle keeps the page it opens on"


@pytest.mark.integration
@pytest.mark.parametrize(
    "file_name, content, kind, message",
    [
        pytest.param("bad.json", "{not json", None, "The content is not valid JSON.", id="json"),
        pytest.param("spec.json", '{"mark": "bar"}', "chart", "A chart must be a Vega-Lite spec", id="chart"),
        pytest.param("photo.png", b"not an image", None, "The content is not an image.", id="image"),
        pytest.param("doc.pdf", b"not a pdf", None, "The content is not a PDF.", id="pdf"),
        pytest.param("site.zip", b"not a zip", "bundle", "The bundle is not a valid zip archive.", id="bundle-zip"),
        pytest.param("site.zip", _zip({"about.html": "x"}), "bundle", "no file at entry_path", id="bundle-entry"),
    ],
)
def test_content_the_platform_rejects_reaches_the_model_as_a_recoverable_error(
    api, backend, file_name, content, kind, message
):
    workspace = InMemoryFileStore()
    workspace.store(file_name, content.encode() if isinstance(content, str) else content, overwrite=True)

    with pytest.raises(ToolExecutionException, match=message) as exc:
        _tool_run(
            ArtifactTool(backend=backend, workspace=workspace), action="create", path=file_name, name="X", kind=kind
        )

    assert exc.value.recoverable
    assert api.artifacts == {}, "nothing is stored"


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_a_deliverable_is_published_then_read_by_another_agent(openai_llm, run_config, backend, api):
    """One agent writes a performance record in its workspace and publishes it; a fresh one reads it.

    Nothing in the first request mentions artifacts: the agent has to recognise a shareable HTML
    page as one. The second agent gets no id and no shared conversation, only the same backend, so
    it has to find the record and read it rather than guess.
    """
    writer = Agent(
        name="ReviewWriter",
        llm=openai_llm,
        role="You are an engineering manager's assistant who prepares polished review documents.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=12,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        artifacts=ArtifactConfig(enabled=True, backend=backend),
    )
    written = writer.run(input_data={"input": REVIEW_REQUEST}, config=run_config)

    assert written.status == RunnableStatus.SUCCESS, written.error
    assert len(api.artifacts) == 1, f"Expected one artifact, the server holds {len(api.artifacts)}."
    ((artifact_id, record),) = api.artifacts.items()
    html = api.latest_text(artifact_id)
    assert record["kind"] == "html", f"Published as {record['kind']}, not html."
    for fact in (EMPLOYEE, "142", MTTR.split()[0], "211", RATING):
        # A model may title-case a rating; the facts are what matter, not their capitalization.
        assert fact.lower() in html.lower(), f"'{fact}' is missing from the published record: {html[:500]}"
    assert record["versions"][-1]["client"].startswith("dynamiq-python/"), "The platform records the client."

    refs = written.output.get("artifacts") or []
    assert [(ref["id"], ref["version_id"]) for ref in refs] == [
        (artifact_id, record["versions"][-1]["id"])
    ], f"Run output did not report the deliverable's latest version: {refs}"
    assert html not in written.output["content"], "The answer should link the record, not paste it."

    # A new agent and conversation; the only thing carried over is the backend.
    reader = Agent(
        name="ReviewReader",
        llm=openai_llm,
        role="You answer HR questions from the team's published documents.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        artifacts=ArtifactConfig(enabled=True, backend=backend),
    )
    calls_before = len(api.calls)
    answered = reader.run(input_data={"input": READ_REQUEST}, config=run_config)

    assert answered.status == RunnableStatus.SUCCESS, answered.error
    answer = answered.output["content"]
    reads = api.calls[calls_before:]
    content_reads = [p for verb, p in reads if verb == "GET" and p.startswith(f"/{artifact_id}/versions/")]
    assert content_reads, f"The record was never read. API calls: {reads}. Answer: {answer}"
    assert MTTR.split()[0] in answer, f"MTTR not taken from the record: {answer}"
    assert "exceed" in answer.lower(), f"Rating not taken from the record: {answer}"
    assert "artifacts" not in answered.output, "Reading an artifact must not report it as a deliverable."
    assert len(api.artifacts) == 1, "The reader must not publish anything."
