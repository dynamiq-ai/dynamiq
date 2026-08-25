import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

evaluation = click.Group(name="evaluation", help="Run and inspect evaluations and metrics")


@evaluation.command("metrics")
@pagination_options
@with_api_and_settings
def list_metrics(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List metrics in the current project (project_id is sent automatically).

    Reuse an existing `predefined` metric where possible; note its `id` and
    `latest_version_id` for `evaluation start`.
    """
    echo_list(api, "/v1/metrics", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@evaluation.command("metric")
@click.argument("metric_id")
@with_api_and_settings
def get_metric(*, api: ApiClient, settings: Settings, metric_id: str):
    """Fetch one metric, including its config - the shape to copy when creating another."""
    echo_response(api.get(f"/v1/metrics/{metric_id}"))


@evaluation.command("metric-create")
@click.argument("payload")
@with_api_and_settings
def create_metric(*, api: ApiClient, settings: Settings, payload: str):
    """Create a metric. REQUIRED: `name`, `project_id` (auto), `type`, `config`.

    `type` is one of: predefined | llm_as_a_judge | custom.
      predefined     -> config {"type": "dynamiq.evaluations.metrics.<X>Evaluator",
                        "config": {"llm": {...}}} where <X> is one of AnswerCorrectness,
                        ContextPrecision, ContextRecall, FactualCorrectness, Faithfulness.
      llm_as_a_judge -> config REQUIRES `instructions` and `llm` {type, model, connection_id}.
      custom         -> config REQUIRES `code`.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/metrics", json=body))


@evaluation.command("metric-test")
@click.argument("payload")
@with_api_and_settings
def test_metric(*, api: ApiClient, settings: Settings, payload: str):
    """Score sample values with a metric config WITHOUT persisting anything.

    Body: {"project_id": auto, "metrics": [{"id": "1", "metric": {...config...},
    "input": {...}, "input_transformer": {"selector": {...}}}]}. Cheap way to prove a
    metric config and its selectors before wiring a full evaluation.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/metrics/test", json=body))


@evaluation.command("start")
@click.argument("payload")
@with_api_and_settings
def start_evaluation(*, api: ApiClient, settings: Settings, payload: str):
    """Start an evaluation. REQUIRED: `name`, `project_id` (auto), `dataset_id`,
    `dataset_version_id`, `config` (non-empty list).

    The dataset version MUST be released - a draft version fails with
    "Dataset version is not released." Each config entry needs at least one metric:

      config: [{
        "workflow": {"id": <uuid>, "version_id": <uuid>,          # both REQUIRED if present
                     "input_transformer": {"selector": {"input": "$.dataset.<col>"}}},
        "metrics": [{"id": <uuid>, "version_id": <uuid, optional - latest is pinned>,
                     "input_transformer": {"selector": {
                        "answers": "$.workflow.<your Output-node field>",
                        "questions": "$.dataset.<col>",
                        "ground_truth_answers": "$.dataset.<col>"}}}]
      }]

    Omit `workflow` entirely to score stored dataset/trace data without running a workflow.
    Runs asynchronously: poll `evaluation get <id>` until status is succeeded/failed.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/evaluations", json=body))


@evaluation.command("list")
@pagination_options
@with_api_and_settings
def list_evaluations(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List evaluations in the current project (25 per page; --all for every one)."""
    echo_list(api, "/v1/evaluations", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@evaluation.command("get")
@click.argument("evaluation_id")
@with_api_and_settings
def get_evaluation(*, api: ApiClient, settings: Settings, evaluation_id: str):
    """Fetch one evaluation. Poll this every 5-10s until `status` leaves pending/running."""
    echo_response(api.get(f"/v1/evaluations/{evaluation_id}"))


@evaluation.command("results")
@click.argument("evaluation_id")
@pagination_options
@with_api_and_settings
def get_results(*, api: ApiClient, settings: Settings, evaluation_id: str, page, page_size, fetch_all, compact):
    """Per-item result rows: each row's workflow output and metric scores (--all for every row)."""
    echo_list(api, f"/v1/evaluations/{evaluation_id}/results", None, page, page_size, fetch_all, compact)


@evaluation.command("results-download")
@click.argument("evaluation_id")
@click.option("--out", "out_path", default=None, help="Write to this file instead of stdout.")
@with_api_and_settings
def download_results(*, api: ApiClient, settings: Settings, evaluation_id: str, out_path: str | None):
    """Download the ENTIRE result set in one call - no pagination to walk."""
    response = api.get(f"/v1/evaluations/{evaluation_id}/results/download")
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    if out_path:
        with open(out_path, "w") as f:
            f.write(response.text)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(response.text)


@evaluation.command("summary")
@click.argument("evaluation_id")
@with_api_and_settings
def get_evaluation_metrics(*, api: ApiClient, settings: Settings, evaluation_id: str):
    """Pre-aggregated avg/min/max per metric - use when you don't need per-row detail."""
    echo_response(api.get(f"/v1/evaluations/{evaluation_id}/metrics"))


@evaluation.command("rerun")
@click.argument("evaluation_id")
@with_api_and_settings
def rerun_evaluation(*, api: ApiClient, settings: Settings, evaluation_id: str):
    """Re-run ONLY the failed tasks, reusing the stored config (no body accepted).

    400s with "Evaluation is still running." while pending/running, and with
    "No failed tasks to rerun." when nothing failed.
    """
    echo_response(api.post(f"/v1/evaluations/{evaluation_id}/rerun"))


@evaluation.command("delete")
@click.argument("evaluation_id")
@click.confirmation_option(prompt="Delete this evaluation?")
@with_api_and_settings
def delete_evaluation(*, api: ApiClient, settings: Settings, evaluation_id: str):
    """Delete an evaluation permanently."""
    echo_response(api.delete(f"/v1/evaluations/{evaluation_id}"))
