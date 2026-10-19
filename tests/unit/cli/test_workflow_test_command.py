from types import SimpleNamespace

from click.testing import CliRunner

from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.workflow import workflow
from dynamiq.cli.config import Settings


class RecordingApi:
    def __init__(self):
        self.calls = []

    def post(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(status_code=200, text='{"output": {}}', json=lambda: {"output": {}})


def form_sent_by(*extra: str) -> dict:
    api = RecordingApi()
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id="00000000-0000-4000-8000-000000000001")
    dctx.api = api
    result = CliRunner().invoke(workflow, ["test", '{"nodes": []}', "{}", *extra], obj=dctx)
    assert result.exit_code == 0, result.output
    return api.calls[0][1]["files"]


def test_a_dry_run_is_sent_on_by_default_and_off_under_no_dry_run():
    assert form_sent_by()["dry_run"] == (None, "true")
    # The runtime's own default is on, so a field left out would keep the dry run on.
    assert form_sent_by("--no-dry-run")["dry_run"] == (None, "false")


def test_the_current_project_is_sent_as_project_id():
    assert form_sent_by()["project_id"] == (None, "00000000-0000-4000-8000-000000000001")


def test_a_missing_project_fails_before_any_request():
    api = RecordingApi()
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id=None)
    dctx.api = api
    result = CliRunner().invoke(workflow, ["test", '{"nodes": []}', "{}"], obj=dctx)
    assert result.exit_code != 0
    assert "No project set" in result.output
    assert api.calls == []
