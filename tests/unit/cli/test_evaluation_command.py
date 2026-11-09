import json

import pytest
from click.testing import CliRunner

from dynamiq.cli.commands.evaluation import evaluation

REQUIRED = {"name": "helpfulness", "metric_id": "m-1", "metric_version_id": "mv-1"}


def test_app_attach_sends_enabled_and_sample_rate_defaults(cli_ctx, recording_api):
    result = CliRunner().invoke(evaluation, ["app-attach", "app-1", json.dumps(REQUIRED)], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[0]
    assert (method, path) == ("POST", "/v1/apps/app-1/evaluations")
    assert kwargs["json"] == REQUIRED | {"enabled": True, "sample_rate": 0.1}


@pytest.mark.parametrize("overrides", [{"enabled": False}, {"sample_rate": 0.5}, {"enabled": False, "sample_rate": 0}])
def test_app_attach_keeps_explicit_values(cli_ctx, recording_api, overrides):
    payload = REQUIRED | overrides

    result = CliRunner().invoke(evaluation, ["app-attach", "app-1", json.dumps(payload)], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    sent = recording_api.calls[0][2]["json"]
    for key, value in overrides.items():
        assert sent[key] == value


def _cp1252_default_open(monkeypatch, module):
    """Make open() without an explicit encoding behave like a cp1252 locale (e.g. Windows)."""
    real_open = open

    def fake_open(file, mode="r", *args, encoding=None, **kwargs):
        return real_open(file, mode, *args, encoding=encoding or "cp1252", **kwargs)

    monkeypatch.setattr(module, "open", fake_open, raising=False)


def test_results_download_writes_utf8_file(cli_ctx, recording_api, tmp_path, monkeypatch):
    import dynamiq.cli.commands.evaluation as module

    _cp1252_default_open(monkeypatch, module)
    text = '{"output": "日本語 – résumé"}'
    recording_api.get = lambda path, **kw: type("R", (), {"status_code": 200, "text": text})()
    out = tmp_path / "results.json"

    result = CliRunner().invoke(evaluation, ["results-download", "e1", "--out", str(out)], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == text
