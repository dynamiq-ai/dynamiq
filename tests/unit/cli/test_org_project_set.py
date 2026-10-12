import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.org import org
from dynamiq.cli.commands.project import project

HOST = "https://api.example.test"


@pytest.fixture
def configured(config_dir):
    config_dir.write(config={"org_id": "org-1"}, creds={"api_host": HOST, "api_key": "tok-file-0123456789"})
    return config_dir


def run(group, *args):
    return CliRunner(mix_stderr=False).invoke(group, list(args), obj=DynamiqCtx())


def test_org_set_saves_on_success(configured, requests_mock):
    requests_mock.get(f"{HOST}/v1/orgs/org-2", json={"data": {"id": "org-2"}})

    result = run(org, "set", "--id", "org-2")

    assert result.exit_code == 0, result.stderr
    assert configured.config() == {"org_id": "org-2"}


def test_org_set_with_env_token_does_not_persist_it(configured, requests_mock, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", "env-token-0123456789")
    requests_mock.get(f"{HOST}/v1/orgs/org-2", json={"data": {"id": "org-2"}})

    result = run(org, "set", "--id", "org-2")

    assert result.exit_code == 0, result.stderr
    assert requests_mock.last_request.headers["Authorization"] == "Bearer env-token-0123456789"
    assert configured.creds() == {"api_host": HOST, "api_key": "tok-file-0123456789"}
    # The override warning goes to stderr, once.
    assert result.stderr.count("DYNAMIQ_API_TOKEN overrides api_key from credentials file") == 1


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "Token rejected (401)"),
        (403, "Organization org-2 is not accessible with this token (403)."),
        (404, "Organization org-2 was not found (404)."),
        (500, "Could not read Organization org-2: HTTP 500."),
    ],
)
def test_org_set_failure_explains_status(configured, requests_mock, status, message):
    requests_mock.get(f"{HOST}/v1/orgs/org-2", status_code=status, json={"error": "x"})

    result = run(org, "set", "--id", "org-2")

    assert result.exit_code == 1
    assert message in result.stderr
    assert configured.config() == {"org_id": "org-1"}


def test_token_rejected_names_the_source_without_the_token(configured, requests_mock):
    requests_mock.get(f"{HOST}/v1/orgs/org-2", status_code=401)

    result = run(org, "set", "--id", "org-2")

    assert "comes from credentials file" in result.stderr
    assert "tok-file-0123456789" not in result.stderr


def test_project_set_saves_on_success(configured, requests_mock):
    requests_mock.get(f"{HOST}/v1/projects/proj-2", json={"data": {"id": "proj-2", "org_id": "org-1"}})

    result = run(project, "set", "--id", "proj-2")

    assert result.exit_code == 0, result.stderr
    assert configured.config() == {"org_id": "org-1", "project_id": "proj-2"}


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "Token rejected (401)"),
        (403, "Project proj-2 is not accessible with this token (403)."),
        (404, "Project proj-2 was not found (404)."),
    ],
)
def test_project_set_failure_explains_status(configured, requests_mock, status, message):
    requests_mock.get(f"{HOST}/v1/projects/proj-2", status_code=status)

    result = run(project, "set", "--id", "proj-2")

    assert result.exit_code == 1
    assert message in result.stderr
    assert "project_id" not in configured.config()
