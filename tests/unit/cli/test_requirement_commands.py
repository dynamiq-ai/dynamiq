import pytest
from click.testing import CliRunner

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.app import app
from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.config import Settings

API_HOST = "https://api.example.test"
APP_URL = "https://support-bot.apps.example.test"
APP_ID = "5f0c9a52-8d4e-4b1f-9a7e-3c2d1b0a9f88"
GMAIL = "c0ffee00-1234-4abc-8def-0123456789ab"
NOTION = "d15ea5e0-5678-4cde-9abc-fedcba987654"
USER_ID = "jane+qa@example.com"


@pytest.fixture
def platform(requests_mock, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_ACCESS_KEY", "app-access-key")
    requests_mock.get(
        f"{API_HOST}/v1/apps/{APP_ID}", json={"data": {"id": APP_ID, "hostname": "support-bot.apps.example.test"}}
    )
    return requests_mock


def run(*args: str):
    settings = Settings(api_host=API_HOST, api_key="management-pat")
    dctx = DynamiqCtx()
    dctx.settings, dctx.api = settings, ApiClient(settings)
    return CliRunner().invoke(app, list(args), obj=dctx)


@pytest.mark.parametrize(
    ("flags", "body"),
    [
        ([], {"user_id": USER_ID}),
        (
            ["--requirement", GMAIL, "--requirement", NOTION],
            {"user_id": USER_ID, "requirement_ids": [GMAIL, NOTION]},
        ),
    ],
    ids=["whole app", "scoped"],
)
def test_connect_token(platform, flags, body):
    platform.post(f"{APP_URL}/v1/connect/tokens", json={"url": "https://connect.example.test/connect?token=ct_1"})

    result = run("connect-token", APP_ID, USER_ID, *flags)

    assert result.exit_code == 0, result.output
    assert platform.last_request.json() == body


def test_disconnect(platform):
    platform.delete(f"{APP_URL}/v1/requirements/{GMAIL}/connections", status_code=204)

    result = run("disconnect", APP_ID, GMAIL, USER_ID, "--yes")

    assert result.exit_code == 0, result.output
    request = platform.last_request
    assert request.method == "DELETE"
    assert request.url == f"{APP_URL}/v1/requirements/{GMAIL}/connections?user_id=jane%2Bqa%40example.com"
    assert request.headers["Authorization"] == "Bearer app-access-key"
