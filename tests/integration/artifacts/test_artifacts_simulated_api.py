"""Artifacts through the backend and the agent's artifact tool, against a simulated platform API.

Everything above the socket is real; ``ArtifactAPISimulator`` answers in place of the platform. Each test
gets a fresh one. No model is called.
"""

import io
import zipfile
from uuid import uuid4

import pytest

from dynamiq.artifacts import ArtifactConflictError, ArtifactError, ArtifactKind, ArtifactNotFoundError
from dynamiq.artifacts.backends import Dynamiq as DynamiqArtifacts
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools import ArtifactTool
from dynamiq.nodes.tools.artifact_tool import ArtifactToolInputSchema
from dynamiq.storages.file import InMemoryFileStore
from tests.helpers.artifact_api import API_URL, PNG, TEXT_KIND_VALUES, ArtifactAPISimulator


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
