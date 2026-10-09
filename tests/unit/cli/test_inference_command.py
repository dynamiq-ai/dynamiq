import json

from click.testing import CliRunner

from dynamiq.cli.commands.deployment import inference

INF_PATH = "/v1/inferences/inf-1"
CURRENT = {
    "id": "inf-1",
    "name": "llama",
    "description": "chat model",
    "model_id": "m-1",
    "resource_profile_id": "rp-1",
    "inference_runtime_id": "irt-1",
    "engine": "vllm",
    "task": "text_generation",
    "status": "running",
    "autoscaling": {"min_replicas": 1, "max_replicas": 2},
    "parameters": {"max_model_len": 8192, "dtype": "auto"},
}


def test_update_sends_the_full_record_with_changes_merged(cli_ctx, recording_api):
    recording_api.responses[("GET", INF_PATH)] = {"data": CURRENT}

    result = CliRunner().invoke(
        inference, ["update", "inf-1", json.dumps({"autoscaling": {"max_replicas": 4}})], obj=cli_ctx
    )

    assert result.exit_code == 0, result.output
    method, path, kwargs = recording_api.calls[-1]
    assert (method, path) == ("PUT", INF_PATH)
    assert kwargs["json"] == {
        "name": "llama",
        "description": "chat model",
        "model_id": "m-1",
        "resource_profile_id": "rp-1",
        "inference_runtime_id": "irt-1",
        "engine": "vllm",
        "autoscaling": {"min_replicas": 1, "max_replicas": 4},
        "parameters": {"max_model_len": 8192, "dtype": "auto"},
    }


def test_update_rejects_fields_the_api_does_not_accept(cli_ctx, recording_api):
    result = CliRunner().invoke(inference, ["update", "inf-1", json.dumps({"task": "embedding"})], obj=cli_ctx)

    assert result.exit_code != 0
    assert "cannot update task" in result.output
    assert recording_api.calls == []
