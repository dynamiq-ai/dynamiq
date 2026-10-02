"""Integration tests for agent artifacts (OPENAI_API_KEY and E2B_API_KEY required).

``ArtifactAPISimulator`` stands in for the platform, so everything above the socket is real: the
same client, URLs, multipart bodies, If-Match headers and status codes. Each test gets a fresh one.
"""

import hashlib
import json as json_lib
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
from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox

MODEL = "gpt-5.4"
API_URL = "https://artifacts.simulated"
KINDS = {kind.value for kind in ArtifactKind}
TEXT_KIND_VALUES = {kind.value for kind in TEXT_KINDS}

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


class ArtifactAPISimulator:
    """In-process implementation of the platform's /v1/artifacts routes.

    Deliberately strict where the platform is: bearer token, known kinds, text kinds only as JSON,
    immutable versions, a repeated upload returning the latest version, and If-Match on the latest
    version id.
    """

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []

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
        artifact_id = str(uuid4())
        self.artifacts[artifact_id] = {
            "id": artifact_id,
            "store_id": body.get("store_id"),
            "user_id": body.get("user_id"),
            "file_name": file_name,
            "name": body["name"],
            "kind": body["kind"],
            "mime_type": body.get("mime_type") or "application/octet-stream",
            "visibility": "private",
            "versions": [],
            "share": None,
        }
        content = body["content"].encode() if isinstance(body["content"], str) else body["content"]
        self._record(artifact_id, content, body, client)
        return _Response(201, {"data": self._as_json(artifact_id)})

    def _add(self, artifact_id, body, if_match, client, upload):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        latest = artifact["versions"][-1]
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
        if _checksum(content) == latest["checksum"] and body.get("name") is None and body.get("description") is None:
            return _Response(200, {"data": self._version_json(artifact_id, latest)})
        version = self._record(artifact_id, content, body, client)
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
    def _record(self, artifact_id, content: bytes, body: dict, client: str | None) -> dict:
        artifact = self.artifacts[artifact_id]
        versions = artifact["versions"]
        version = {
            "id": str(uuid4()),
            "version": len(versions) + 1,
            "name": body.get("name") or artifact["name"],
            "description": body.get("description"),
            "content": content,
            "checksum": _checksum(content),
            "client": client,
            "created_at": _now(),
        }
        versions.append(version)
        artifact["name"] = version["name"]
        return version

    def _version_json(self, artifact_id, version):
        artifact = self.artifacts[artifact_id]
        return {
            "id": version["id"],
            "artifact_id": artifact_id,
            "version": version["version"],
            "name": version["name"],
            "description": version["description"],
            "mime_type": artifact["mime_type"],
            "size": len(version["content"]),
            "checksum": version["checksum"],
            "entry_path": None,
            "source": {"client": version["client"]},
            "created_at": version["created_at"],
        }

    def _as_json(self, artifact_id):
        artifact = self.artifacts[artifact_id]
        latest = artifact["versions"][-1]
        return {
            **{k: artifact[k] for k in ("id", "store_id", "user_id", "file_name", "name", "kind", "mime_type")},
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

    image = backend.create(file_name="chart.png", name="Chart", kind=ArtifactKind.IMAGE, content=b"\x89PNG\r\n")
    _, raw = backend.get(image.id)
    assert raw == b"\x89PNG\r\n", "Binary content must round-trip as bytes."

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


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_a_deliverable_is_published_then_read_by_another_agent(openai_llm, run_config, backend, api):
    """One agent writes a performance record in its sandbox and publishes it; a fresh one reads it.

    Nothing in the first request mentions artifacts: the agent has to recognise a shareable HTML
    page as one. The second agent gets no id and no shared conversation, only the same backend, so
    it has to find the record and read it rather than guess.
    """
    sandbox = E2BSandbox(connection=E2BConnection())
    try:
        writer = Agent(
            name="ReviewWriter",
            llm=openai_llm,
            role="You are an engineering manager's assistant who prepares polished review documents.",
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=12,
            sandbox=SandboxConfig(enabled=True, backend=sandbox),
            artifacts=ArtifactConfig(enabled=True, backend=backend),
        )
        written = writer.run(input_data={"input": REVIEW_REQUEST}, config=run_config)
    finally:
        sandbox.close(kill=True)

    assert written.status == RunnableStatus.SUCCESS, written.error
    assert len(api.artifacts) == 1, f"Expected one artifact, the server holds {len(api.artifacts)}."
    ((artifact_id, record),) = api.artifacts.items()
    html = api.latest_text(artifact_id)
    assert record["kind"] == "html", f"Published as {record['kind']}, not html."
    for fact in (EMPLOYEE, "142", MTTR.split()[0], "211", RATING):
        assert fact in html, f"'{fact}' is missing from the published record: {html[:500]}"
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
