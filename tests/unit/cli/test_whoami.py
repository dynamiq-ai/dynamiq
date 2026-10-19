import json

import pytest
import requests
from click.testing import CliRunner

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.utils import cli

HOST = "https://api.example.test"
TOKEN = "tok-file-0123456789"
ENV_TOKEN = "env-token-0123456789"
ME = {"id": "u-1", "email": "jane@example.com", "first_name": "Jane", "last_name": "Doe", "timezone": "UTC"}
ORG = {"id": "org-1", "name": "Acme"}
PROJECT = {"id": "proj-1", "name": "Support", "org_id": "org-1"}


@pytest.fixture
def configured(config_dir):
    config_dir.write(
        config={"org_id": "org-1", "project_id": "proj-1"},
        creds={"api_host": HOST, "api_key": TOKEN},
    )
    return config_dir


@pytest.fixture
def api(requests_mock):
    requests_mock.get(f"{HOST}/v1/me", json={"data": ME})
    requests_mock.get(f"{HOST}/v1/orgs/org-1", json={"data": ORG})
    requests_mock.get(f"{HOST}/v1/projects/proj-1", json={"data": PROJECT})
    return requests_mock


def run(*args):
    return CliRunner(mix_stderr=False).invoke(cli, ["whoami", *args], obj=DynamiqCtx())


def test_whoami_prints_identity_and_sources(configured, api):
    result = run()

    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "User:     Jane Doe <jane@example.com> (id u-1)" in out
    assert "Token:    tok-... (credentials file)" in out
    assert f"Host:     {HOST} (credentials file)" in out
    assert "Org:      Acme [org-1] (config file)" in out
    assert "Project:  Support [proj-1] (config file)" in out
    assert TOKEN not in out + result.stderr
    assert result.stderr == ""
    assert api.request_history[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_whoami_names_env_sources(configured, api, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", ENV_TOKEN)

    result = run()

    assert result.exit_code == 0, result.stderr
    assert "Token:    env-... (env DYNAMIQ_API_TOKEN)" in result.stdout
    assert ENV_TOKEN not in result.stdout


def test_whoami_default_host(config_dir, requests_mock):
    config_dir.write(creds={"api_key": TOKEN})
    requests_mock.get("https://api.getdynamiq.ai/v1/me", json={"data": ME})

    result = run()

    assert result.exit_code == 0, result.stderr
    assert "Host:     https://api.getdynamiq.ai (default)" in result.stdout
    assert "Org:      <not set>" in result.stdout
    assert "Project:  <not set>" in result.stdout


def test_whoami_json(configured, api):
    result = run("--json")

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["user"] == {"id": "u-1", "email": "jane@example.com", "first_name": "Jane", "last_name": "Doe"}
    assert report["token"] == {"value": "tok-...", "source": "credentials file", "status": 200, "accepted": True}
    assert report["org"] == {
        "id": "org-1",
        "source": "config file",
        "accessible": True,
        "status": 200,
        "name": "Acme",
    }
    assert report["project"]["accessible"] is True
    assert report["project"]["org_id"] == "org-1"
    assert report["problems"] == []


def test_whoami_json_stdout_stays_parseable_with_override_warning(configured, api, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "proj-1")  # equal to disk: no warning
    monkeypatch.setenv("DYNAMIQ_ORG_ID", "org-1")
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", ENV_TOKEN)  # differs: one warning

    result = run("--json")

    report = json.loads(result.stdout)
    assert report["token"]["source"] == "env DYNAMIQ_API_TOKEN"
    assert result.stderr.count("overrides") == 1
    assert "DYNAMIQ_API_TOKEN overrides api_key from credentials file" in result.stderr


def test_whoami_token_rejected(configured, requests_mock):
    requests_mock.get(f"{HOST}/v1/me", status_code=401, json={"error": "unauthorized"})

    result = run()

    assert result.exit_code == 1
    assert "REJECTED (401)" in result.stdout
    assert "Token rejected (401)" in result.stderr
    assert TOKEN not in result.stdout + result.stderr
    # Nothing else is worth asking with a rejected token.
    assert [r.path for r in requests_mock.request_history] == ["/v1/me"]


def test_whoami_token_rejected_json(configured, requests_mock):
    requests_mock.get(f"{HOST}/v1/me", status_code=401)

    result = run("--json")

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["token"]["accepted"] is False


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (403, "Project proj-1 is not accessible with this token (403)."),
        (404, "Project proj-1 was not found (404)."),
    ],
)
def test_whoami_project_not_accessible(configured, api, status, message):
    api.get(f"{HOST}/v1/projects/proj-1", status_code=status)

    result = run()

    assert result.exit_code == 1
    assert f"Project:  proj-1 (config file) - NOT ACCESSIBLE (HTTP {status})" in result.stdout
    assert message in result.stderr
    assert "Org:      Acme [org-1] (config file)" in result.stdout


def test_whoami_org_not_accessible(configured, api):
    api.get(f"{HOST}/v1/orgs/org-1", status_code=403)

    result = run("--json")

    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert report["org"]["accessible"] is False
    assert report["org"]["status"] == 403
    assert report["problems"] == ["Organization org-1 is not accessible with this token (403)."]


def test_whoami_project_in_another_org(configured, api):
    api.get(f"{HOST}/v1/projects/proj-1", json={"data": {**PROJECT, "org_id": "org-9"}})

    result = run()

    assert result.exit_code == 1
    assert "belongs to organization org-9, not the configured organization org-1" in result.stderr


def test_whoami_service_account_token(configured, api):
    api.get(f"{HOST}/v1/me", status_code=403)

    result = run()

    assert result.exit_code == 0, result.stderr
    assert "User:     token is not tied to a user (403 from /v1/me)" in result.stdout
    assert "Org:      Acme [org-1]" in result.stdout


def test_whoami_without_token(config_dir, requests_mock):
    result = run()

    assert result.exit_code == 1
    assert "No API token configured" in result.stderr
    assert not requests_mock.request_history


def test_whoami_unreachable_host(configured, monkeypatch):
    def refuse(self, path, params=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(ApiClient, "get", refuse)

    result = run()

    assert result.exit_code == 1
    assert f"Could not reach {HOST}" in result.stderr
