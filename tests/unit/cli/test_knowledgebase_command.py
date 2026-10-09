import json

from click.testing import CliRunner

from dynamiq.cli.commands.knowledgebase import knowledgebase

KB_PATH = "/v1/knowledgebases/kb-1"
CURRENT = {
    "id": "kb-1",
    "name": "docs",
    "description": "product docs",
    "runtime_id": "rt-1",
    "workflow_id": "wf-1",
    "workflow_version_id": "wfv-1",
}


def test_update_keeps_current_values_for_omitted_fields(cli_ctx, recording_api):
    recording_api.responses[("GET", KB_PATH)] = {"data": CURRENT}

    result = CliRunner().invoke(
        knowledgebase, ["update", "kb-1", json.dumps({"workflow_version_id": "wfv-2"})], obj=cli_ctx
    )

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[-1]
    assert (method, path) == ("PUT", KB_PATH)
    assert kwargs["json"] == {"description": "product docs", "runtime_id": "rt-1", "workflow_version_id": "wfv-2"}


def test_update_rejects_fields_the_api_does_not_accept(cli_ctx, recording_api):
    result = CliRunner().invoke(knowledgebase, ["update", "kb-1", json.dumps({"name": "renamed"})], obj=cli_ctx)

    assert result.exit_code != 0
    assert "cannot update name" in result.output
    assert recording_api.calls == []


def test_source_add_help_says_when_connection_id_is_required():
    result = CliRunner().invoke(knowledgebase, ["source-add", "--help"])

    assert result.exit_code == 0, result.output
    assert "connection_id" in result.output
    assert "except `website`" in result.output
