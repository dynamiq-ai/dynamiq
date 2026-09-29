import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.artifact import artifact
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.config import Settings

ARTIFACT = {"id": "a1", "title": "Q3", "latest": {"version": 3}, "url": "https://x/a1"}


def _response(payload=None, content: bytes | None = None, status_code=200):
    body = json.dumps(payload) if payload is not None else ""
    return SimpleNamespace(
        status_code=status_code,
        text=body,
        content=content if content is not None else body.encode(),
        json=lambda: payload,
    )


class RecordingApi:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def _record(self, method, path, kwargs):
        files = kwargs.get("files")
        if files:
            # The handle is closed once the command returns; keep what was sent.
            name, handle, media_type = files["file"]
            kwargs = {**kwargs, "files": {"file": (name, handle.read(), media_type)}}
        self.calls.append((method, path, kwargs))
        return self.responses.pop(0) if self.responses else _response({"data": ARTIFACT})

    def get(self, path, **kwargs):
        return self._record("GET", path, kwargs)

    def post(self, path, **kwargs):
        return self._record("POST", path, kwargs)

    def put(self, path, **kwargs):
        return self._record("PUT", path, kwargs)


def invoke(args, api=None, env=None):
    api = api or RecordingApi()
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id="00000000-0000-4000-8000-000000000001")
    dctx.api = api
    result = CliRunner(env=env).invoke(artifact, args, obj=dctx)
    return result, api


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "q3-report.html"
    path.write_text("<html>Q3</html>")
    return str(path)


def test_publish_uploads_the_file_with_inferred_kind(report):
    result, api = invoke(["publish", report, "--title", "Q3 report"], env={"DYNAMIQ_CONVERSATION_ID": "c-9"})

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("POST", "/v1/artifacts")
    assert kwargs["files"]["file"] == ("q3-report.html", b"<html>Q3</html>", "text/html")
    assert kwargs["data"]["name"] == "q3-report.html"
    assert kwargs["data"]["kind"] == "html"
    assert json.loads(kwargs["data"]["source"]) == {"conversation_id": "c-9", "client": "cli"}
    assert "project_id" not in kwargs["data"], "user-owned unless a project is named"
    assert '"id": "a1"' in result.output


def test_publish_needs_a_title_for_a_new_artifact(report):
    result, api = invoke(["publish", report])

    assert result.exit_code != 0
    assert "--title is required" in result.output
    assert api.calls == []


def test_publish_with_artifact_id_adds_a_version(report):
    result, api = invoke(["publish", report, "--artifact-id", "a1", "--summary", "new numbers"])

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("PUT", "/v1/artifacts/a1")
    assert kwargs["data"]["summary"] == "new numbers"
    assert kwargs["headers"] is None


def test_update_sends_if_match(report):
    result, api = invoke(["update", "a1", report, "--if-match", "sha256:abc"])

    assert result.exit_code == 0, result.output
    assert api.calls[0][2]["headers"] == {"If-Match": "sha256:abc"}


def test_list_passes_filters(report):
    api = RecordingApi([_response({"data": [ARTIFACT], "pagination": {"total_count": 1, "page_count": 1}})])

    result, api = invoke(["list", "--kind", "html", "--query", "q3"], api=api)

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("GET", "/v1/artifacts")
    assert kwargs["params"]["kind"] == "html" and kwargs["params"]["query"] == "q3"


def test_get_out_resolves_the_latest_version_and_saves_bytes(tmp_path):
    out = tmp_path / "saved.html"
    api = RecordingApi([_response({"data": ARTIFACT}), _response(content=b"<html>v3</html>")])

    result, api = invoke(["get", "a1", "--out", str(out)], api=api)

    assert result.exit_code == 0, result.output
    assert [c[1] for c in api.calls] == ["/v1/artifacts/a1", "/v1/artifacts/a1/versions/3/content"]
    assert out.read_bytes() == b"<html>v3</html>"


def test_get_prints_metadata_without_out():
    result, api = invoke(["get", "a1", "--version", "2"])

    assert result.exit_code == 0, result.output
    assert api.calls[0][2]["params"] == {"version": 2}


def test_an_error_status_exits_non_zero(report):
    api = RecordingApi([_response({"message": "forbidden"}, status_code=403)])

    result, _ = invoke(["publish", report, "--title", "T"], api=api)

    assert result.exit_code != 0
    assert "HTTP 403" in result.output
