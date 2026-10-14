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


def test_streaming_turned_on_without_an_event_is_sent_with_the_runtime_default():
    from dynamiq.cli.commands.workflow import normalize_flow

    flow = {
        "nodes": [
            {
                "id": "helper",
                "type": "dynamiq.nodes.agents.Agent",
                "streaming": {"enabled": True},
                "llm": {"id": "llm", "streaming": {"enabled": False}},
                "tools": [{"id": "search", "streaming": {"enabled": True, "event": None}}],
            },
            {"id": "named", "type": "dynamiq.nodes.agents.Agent", "streaming": {"enabled": True, "event": "data"}},
        ]
    }

    normalize_flow(flow)

    helper, named = flow["nodes"]
    assert helper["streaming"]["event"] == "streaming"
    assert helper["tools"][0]["streaming"]["event"] == "streaming"
    assert "event" not in helper["llm"]["streaming"]
    assert named["streaming"]["event"] == "data"
