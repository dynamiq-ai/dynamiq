"""Integration tests for agent artifacts (OPENAI_API_KEY and E2B_API_KEY required).

``ArtifactAPISimulator`` stands in for the server, so everything above the socket is real: the
same client, URLs, multipart bodies, If-Match headers and status codes. Each test gets a fresh one.
"""

import hashlib
import json as json_lib
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from dynamiq.connections import E2B as E2BConnection
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.file_tools import EditOperation
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.storages.artifact import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactStoreConfig,
    ArtifactStoreError,
    DynamiqArtifactStore,
)

MODEL = "gpt-5.4"
API_URL = "https://artifacts.simulated"
KINDS = {kind.value for kind in ArtifactKind}

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
    """The subset of a `requests` response that DynamiqArtifactStore reads."""

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


def _checksum(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class ArtifactAPISimulator:
    """In-process implementation of /v1/artifacts, following the artifacts proposal.

    Deliberately strict where a real server must be: bearer token, known kinds, immutable versions,
    If-Match, and edits that must match exactly once unless `replace_all` is set.
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
            return _Response(401, {"message": "missing bearer token"})

        parts = [p for p in path.split("/") if p]
        if verb == "POST" and not parts:
            return self._create(self._body(json, data, files))
        if verb == "GET" and not parts:
            return self._list(params)
        if verb == "PUT" and len(parts) == 1:
            return self._update(parts[0], self._body(json, data, files), headers.get("If-Match"))
        if verb == "GET" and len(parts) == 1:
            return self._get(parts[0], params.get("version"))
        if verb == "GET" and len(parts) == 4 and parts[1] == "versions" and parts[3] == "content":
            return self._content(parts[0], int(parts[2]))
        return _Response(404, {"message": f"no route for {verb} {path}"})

    @staticmethod
    def _body(json, data, files) -> dict:
        """JSON bodies carry text; multipart bodies carry bytes plus JSON-encoded dict fields."""
        if files:
            body = {k: json_lib.loads(v) if k in ("source", "metadata") else v for k, v in (data or {}).items()}
            _, content, _ = files["file"]
            body["content"] = content if isinstance(content, bytes) else content.read()
            return body
        body = dict(json or {})
        if isinstance(body.get("content"), str):
            body["content"] = body["content"].encode()
        return body

    # -- endpoints ---------------------------------------------------------
    def _create(self, body):
        missing = [k for k in ("name", "title", "kind", "content") if not body.get(k)]
        if missing:
            return _Response(422, {"message": f"missing fields: {missing}"})
        if body["kind"] not in KINDS:
            return _Response(422, {"message": f"unknown kind: {body['kind']}"})
        artifact_id = uuid4().hex[:12]
        self.artifacts[artifact_id] = {
            "id": artifact_id,
            "name": body["name"],
            "title": body["title"],
            "kind": body["kind"],
            "media_type": body.get("media_type") or "application/octet-stream",
            "project_id": body.get("project_id"),
            "metadata": body.get("metadata") or {},
            "versions": [],
        }
        self._add_version(artifact_id, body["content"], body)
        return _Response(201, {"data": self._as_json(artifact_id)})

    def _update(self, artifact_id, body, if_match):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _Response(404, {"message": f"no such artifact: {artifact_id}"})
        latest = artifact["versions"][-1]
        if if_match and if_match not in (latest["checksum"], str(latest["version"])):
            return _Response(412, {"message": "latest version has changed"})

        content = body.get("content")
        if body.get("edits"):
            text = latest["content"].decode()
            for edit in body["edits"]:
                count = text.count(edit["find"])
                if count == 0:
                    return _Response(422, {"message": f"edit target not found: {edit['find']!r}"})
                if count > 1 and not edit.get("replace_all"):
                    return _Response(422, {"message": f"edit target is ambiguous: {edit['find']!r}"})
                text = text.replace(edit["find"], edit["replace"], -1 if edit.get("replace_all") else 1)
            content = text.encode()
        if content is None and not body.get("title"):
            return _Response(422, {"message": "nothing to update"})
        if body.get("title"):
            artifact["title"] = body["title"]
        self._add_version(artifact_id, content if content is not None else latest["content"], body)
        return _Response(200, {"data": self._as_json(artifact_id)})

    def _get(self, artifact_id, version):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or (version is not None and not 1 <= int(version) <= len(artifact["versions"])):
            return _Response(404, {"message": f"no such artifact or version: {artifact_id} v{version}"})
        return _Response(200, {"data": self._as_json(artifact_id)})

    def _content(self, artifact_id, version):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None or not 1 <= version <= len(artifact["versions"]):
            return _Response(404, {"message": f"no such version: {artifact_id} v{version}"})
        return _Response(200, raw=artifact["versions"][version - 1]["content"])

    def _list(self, params):
        found = list(self.artifacts)
        if params.get("kind"):
            found = [i for i in found if self.artifacts[i]["kind"] == params["kind"]]
        if params.get("query"):
            q = params["query"].lower()
            found = [i for i in found if q in self.artifacts[i]["title"].lower() or q in self.artifacts[i]["name"]]
        found.sort(key=lambda i: self.artifacts[i]["versions"][-1]["created_at"], reverse=True)
        return _Response(200, {"data": [self._as_json(i) for i in found[: int(params.get("page_size", 50))]]})

    # -- helpers -----------------------------------------------------------
    def _add_version(self, artifact_id, content: bytes, body: dict) -> None:
        versions = self.artifacts[artifact_id]["versions"]
        versions.append(
            {
                "id": uuid4().hex[:12],
                "version": len(versions) + 1,
                "content": content,
                "checksum": _checksum(content),
                "summary": body.get("summary"),
                "source": body.get("source"),
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
        )

    def _as_json(self, artifact_id):
        artifact = self.artifacts[artifact_id]
        latest = artifact["versions"][-1]
        return {
            **{k: artifact[k] for k in ("id", "name", "title", "kind", "media_type", "metadata")},
            "url": f"https://app.simulated/artifacts/{artifact_id}",
            "latest": {
                "id": latest["id"],
                "version": latest["version"],
                "title": artifact["title"],
                "size": len(latest["content"]),
                "checksum": latest["checksum"],
                "summary": latest["summary"],
                "created_at": latest["created_at"],
            },
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
def store(api):
    return DynamiqArtifactStore(connection=DynamiqConnection(url=API_URL, api_key="simulated-key"))


@pytest.mark.integration
def test_store_crud_against_the_api(store, api):
    """Every endpoint of the contract, without a model in the loop.

    Run this first when bringing a server up: it isolates the HTTP contract from agent behaviour,
    so a failure here is the server's, not the model's.
    """
    created = store.create(name="probe.md", title="Probe", kind=ArtifactKind.MARKDOWN, content="Revenue Q2")
    assert created.version == 1 and created.url

    updated = store.update(created.id, edits=[EditOperation(find="Q2", replace="Q3")], summary="fix quarter")
    assert updated.version == 2

    artifact, latest = store.get(created.id)
    assert (artifact.version, latest) == (2, "Revenue Q3")
    _, first = store.get(created.id, version=1)
    assert first == "Revenue Q2", "Earlier versions must stay readable."

    image = store.create(name="chart.png", title="Chart", kind=ArtifactKind.IMAGE, content=b"\x89PNG\r\n")
    _, raw = store.get(image.id)
    assert raw == b"\x89PNG\r\n", "Binary content must round-trip as bytes."

    assert [a.id for a in store.list(kind=ArtifactKind.MARKDOWN)] == [created.id]
    assert [a.id for a in store.list(query="chart")] == [image.id]


@pytest.mark.integration
def test_store_error_branches(store, api):
    """The parts of the contract a server is most likely to leave out."""
    with pytest.raises(ArtifactNotFoundError):
        store.get("never-created")

    created = store.create(name="doc.md", title="Doc", kind=ArtifactKind.MARKDOWN, content="v1")
    stale = created.latest.checksum
    store.update(created.id, content="v2 from a teammate")
    with pytest.raises(ArtifactConflictError):
        store.update(created.id, content="v2 from the agent", if_match=stale)
    assert api.latest_text(created.id) == "v2 from a teammate", "A conflicting write must change nothing."

    with pytest.raises(ArtifactStoreError, match="not found"):
        store.update(created.id, edits=[EditOperation(find="absent", replace="x")])


@pytest.mark.flaky(reruns=2)
@pytest.mark.integration
def test_a_deliverable_is_published_then_read_by_another_agent(openai_llm, run_config, store, api):
    """One agent writes a performance record in its sandbox and publishes it; a fresh one reads it.

    Nothing in the first request mentions artifacts: the agent has to recognise a shareable HTML
    page as one. The second agent gets no id and no shared conversation, only the same store, so it
    has to find the record and read it rather than guess.
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
            artifact_store=ArtifactStoreConfig(enabled=True, backend=store),
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
    assert record["versions"][-1]["source"].get("run_id"), "The version must record the run that made it."

    refs = written.output.get("artifacts") or []
    assert [ref["id"] for ref in refs] == [artifact_id], f"Run output did not report the deliverable: {refs}"
    assert html not in written.output["content"], "The answer should link the record, not paste it."

    # A new agent and conversation; the only thing carried over is the store.
    reader = Agent(
        name="ReviewReader",
        llm=openai_llm,
        role="You answer HR questions from the team's published documents.",
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
        artifact_store=ArtifactStoreConfig(enabled=True, backend=store),
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
