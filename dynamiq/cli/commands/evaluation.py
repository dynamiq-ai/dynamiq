import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_response, read_json_arg
from dynamiq.cli.config import Settings

evaluation = click.Group(name="evaluation", help="Run and inspect evaluations and metrics")


@evaluation.command("metrics")
@with_api_and_settings
def list_metrics(*, api: ApiClient, settings: Settings):
    echo_response(api.get("/v1/metrics"))


@evaluation.command("start")
@click.argument("payload")
@with_api_and_settings
def start_evaluation(*, api: ApiClient, settings: Settings, payload: str):
    """Start an evaluation. PAYLOAD is inline JSON or @file; copy the shape from an
    existing evaluation (`dynamiq evaluation get <id>`) rather than inventing it."""
    echo_response(api.post("/v1/evaluations", json=read_json_arg(payload)))


@evaluation.command("list")
@with_api_and_settings
def list_evaluations(*, api: ApiClient, settings: Settings):
    echo_response(api.get("/v1/evaluations"))


@evaluation.command("get")
@click.argument("evaluation_id")
@with_api_and_settings
def get_evaluation(*, api: ApiClient, settings: Settings, evaluation_id: str):
    echo_response(api.get(f"/v1/evaluations/{evaluation_id}"))


@evaluation.command("results")
@click.argument("evaluation_id")
@with_api_and_settings
def get_results(*, api: ApiClient, settings: Settings, evaluation_id: str):
    echo_response(api.get(f"/v1/evaluations/{evaluation_id}/results"))


@evaluation.command("rerun")
@click.argument("evaluation_id")
@with_api_and_settings
def rerun_evaluation(*, api: ApiClient, settings: Settings, evaluation_id: str):
    echo_response(api.post(f"/v1/evaluations/{evaluation_id}/rerun", json={}))
