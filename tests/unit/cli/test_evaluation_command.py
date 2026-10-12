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
