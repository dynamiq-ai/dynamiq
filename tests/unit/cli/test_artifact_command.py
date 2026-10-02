import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.artifact import artifact
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.config import Settings

ARTIFACT = {"id": "a1", "name": "Q3", "latest_version": {"id": "v3", "version": 3}, "url": "https://x/a1"}
VERSIONS = [{"id": "v3", "version": 3}, {"id": "v2", "version": 2}, {"id": "v1", "version": 1}]


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
            name, handle, mime_type = files["file"]
            kwargs = {**kwargs, "files": {"file": (name, handle.read(), mime_type)}}
        self.calls.append((method, path, kwargs))
        return self.responses.pop(0) if self.responses else _response({"data": ARTIFACT})

    def get(self, path, **kwargs):
        return self._record("GET", path, kwargs)

    def post(self, path, **kwargs):
        return self._record("POST", path, kwargs)

    def delete(self, path, **kwargs):
        return self._record("DELETE", path, kwargs)


def invoke(args, api=None):
    api = api or RecordingApi()
    dctx = DynamiqCtx()
    dctx.settings = Settings(org_id="00000000-0000-4000-8000-000000000001")
    dctx.api = api
    result = CliRunner().invoke(artifact, args, obj=dctx)
    return result, api


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "q3-report.html"
    path.write_text("<html>Q3</html>")
    return str(path)


def test_publish_uploads_the_file_with_inferred_kind(report):
    result, api = invoke(["publish", report, "--name", "Q3 report"])

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("POST", "/v1/artifacts/upload")
    assert kwargs["files"]["file"] == ("q3-report.html", b"<html>Q3</html>", "text/html")
    assert json.loads(kwargs["data"]["data"]) == {"file_name": "q3-report.html", "name": "Q3 report", "kind": "html"}
    assert kwargs["retry"] is False, "a retried create could publish a duplicate"
    assert '"id": "a1"' in result.output


def test_publish_into_a_store_for_an_end_user(report):
    result, api = invoke(["publish", report, "--name", "Q3", "--store-id", "s1", "--user-id", "customer-42"])

    assert result.exit_code == 0, result.output
    fields = json.loads(api.calls[0][2]["data"]["data"])
    assert (fields["store_id"], fields["user_id"]) == ("s1", "customer-42")


def test_publish_a_zipped_site(tmp_path):
    site = tmp_path / "site.zip"
    site.write_bytes(b"PK\x03\x04")

    result, api = invoke(["publish", str(site), "--name", "Site", "--kind", "bundle", "--entry", "home.html"])

    assert result.exit_code == 0, result.output
    fields = json.loads(api.calls[0][2]["data"]["data"])
    assert (fields["kind"], fields["entry_path"]) == ("bundle", "home.html")


@pytest.mark.parametrize(
    "args, message",
    [
        ([], "--name is required"),
        (["--name", "Q3", "--user-id", "customer-42"], "--user-id requires --store-id"),
        (["--name", "Q3", "--if-match", "v2"], "--if-match applies with --artifact-id only"),
        (["--artifact-id", "a1", "--store-id", "s1"], "apply to a new artifact only"),
    ],
)
def test_publish_refuses_options_that_do_not_go_together(report, args, message):
    result, api = invoke(["publish", report, *args])

    assert result.exit_code != 0
    assert message in result.output
    assert api.calls == []


def test_publish_with_artifact_id_adds_a_version(report):
    result, api = invoke(["publish", report, "--artifact-id", "a1", "--description", "new numbers"])

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("POST", "/v1/artifacts/a1/versions/upload")
    assert json.loads(kwargs["data"]["data"]) == {"description": "new numbers"}
    assert kwargs["headers"] is None
    assert kwargs["retry"] is True, "the same content returns the latest version, so a retry is harmless"


def test_update_sends_if_match(report):
    result, api = invoke(["update", "a1", report, "--if-match", "v2"])

    assert result.exit_code == 0, result.output
    assert api.calls[0][1] == "/v1/artifacts/a1/versions/upload"
    assert api.calls[0][2]["headers"] == {"If-Match": '"v2"'}


def test_list_passes_filters():
    api = RecordingApi([_response({"data": [ARTIFACT], "pagination": {"total_count": 1, "page_count": 1}})])

    result, api = invoke(["list", "--store-id", "s1", "--user-id", "customer-42", "--kind", "html"], api=api)

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[0]
    assert (method, path) == ("GET", "/v1/artifacts")
    assert {k: kwargs["params"][k] for k in ("store_id", "user_id", "kind")} == {
        "store_id": "s1",
        "user_id": "customer-42",
        "kind": "html",
    }


def test_get_prints_metadata():
    result, api = invoke(["get", "a1"])

    assert result.exit_code == 0, result.output
    assert api.calls[0][1] == "/v1/artifacts/a1"


def test_get_out_saves_the_latest_version(tmp_path):
    out = tmp_path / "saved.html"
    api = RecordingApi([_response(content=b"<html>v3</html>")])

    result, api = invoke(["get", "a1", "--out", str(out)], api=api)

    assert result.exit_code == 0, result.output
    assert [c[1] for c in api.calls] == ["/v1/artifacts/a1/versions/latest/download"]
    assert out.read_bytes() == b"<html>v3</html>"


def test_get_a_version_resolves_its_id(tmp_path):
    out = tmp_path / "saved.html"
    api = RecordingApi([_response({"data": VERSIONS}), _response(content=b"<html>v2</html>")])

    result, api = invoke(["get", "a1", "--version", "2", "--out", str(out)], api=api)

    assert result.exit_code == 0, result.output
    assert [c[1] for c in api.calls] == ["/v1/artifacts/a1/versions", "/v1/artifacts/a1/versions/v2/download"]
    assert "Saved v2 of a1" in result.output


def test_get_a_version_that_is_gone():
    api = RecordingApi([_response({"data": VERSIONS})])

    result, _ = invoke(["get", "a1", "--version", "9"], api=api)

    assert result.exit_code != 0
    assert "has no version 9" in result.output


def test_share_pins_a_version_and_an_expiry():
    api = RecordingApi([_response({"data": VERSIONS}), _response({"data": {"url": "https://x/a/s1"}})])

    result, api = invoke(["share", "a1", "--pin", "1", "--expires-in-days", "30"], api=api)

    assert result.exit_code == 0, result.output
    method, path, kwargs = api.calls[-1]
    assert (method, path) == ("POST", "/v1/artifacts/a1/share")
    assert kwargs["json"]["pinned_version_id"] == "v1"
    expires_at = datetime.fromisoformat(kwargs["json"]["expires_at"])
    assert abs(expires_at - (datetime.now(timezone.utc) + timedelta(days=30))) < timedelta(minutes=1)
    assert "https://x/a/s1" in result.output


def test_share_revoke_deletes_the_link():
    result, api = invoke(["share", "a1", "--revoke"])

    assert result.exit_code == 0, result.output
    assert api.calls[0][:2] == ("DELETE", "/v1/artifacts/a1/share")


def test_an_error_status_exits_non_zero(report):
    api = RecordingApi([_response({"error": {"message": "forbidden"}}, status_code=403)])

    result, _ = invoke(["publish", report, "--name", "T"], api=api)

    assert result.exit_code != 0
    assert "HTTP 403" in result.output
