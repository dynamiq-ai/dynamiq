import json

from click.testing import CliRunner

from dynamiq.cli.commands.voice import voice


def test_update_sends_patch_with_the_payload_as_given(cli_ctx, recording_api):
    payload = {"name": "support-line", "config": {"instructions": "Be brief.", "mode": "pipeline"}}

    result = CliRunner().invoke(voice, ["update", "agent-1", json.dumps(payload)], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[0]
    assert (method, path) == ("PATCH", "/v1/agents/voice/agents/agent-1")
    assert kwargs["json"] == payload
