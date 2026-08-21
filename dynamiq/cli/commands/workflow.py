import json

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

workflow = click.Group(name="workflow", help="Manage workflows: create, save the DAG, test, release")


def read_json_arg(value: str):
    """Accept inline JSON or @path/to/file.json."""
    try:
        if value.startswith("@"):
            with open(value[1:]) as f:
                return json.load(f)
        return json.loads(value)
    except FileNotFoundError:
        raise click.ClickException(f"file not found: {value[1:]}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"invalid JSON in {value[:60]}: {e}")


def echo_response(response, success_message: str | None = None) -> None:
    """Print the JSON body; non-200 exits with the body as the error."""
    body = response.text.strip()
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {body[:2000]}")
    if success_message:
        click.echo(success_message)
    if body:
        click.echo(json.dumps(response.json(), indent=2, ensure_ascii=False))


@workflow.command("list")
@with_api_and_settings
def list_workflows(*, api: ApiClient, settings: Settings):
    echo_response(api.get("/v1/workflows"))


@workflow.command("get")
@click.argument("workflow_id")
@with_api_and_settings
def get_workflow(*, api: ApiClient, settings: Settings, workflow_id: str):
    echo_response(api.get(f"/v1/workflows/{workflow_id}"))


@workflow.command("create")
@click.argument("payload")
@with_api_and_settings
def create_workflow(*, api: ApiClient, settings: Settings, payload: str):
    echo_response(api.post("/v1/workflows", json=read_json_arg(payload)))


@workflow.command("save")
@click.argument("workflow_id")
@click.argument("flow")
@click.option("--flow-ui", default=None, help="Canvas layout JSON (inline or @file).")
@with_api_and_settings
def save_workflow(*, api: ApiClient, settings: Settings, workflow_id: str, flow: str, flow_ui: str | None):
    """Save the workflow DAG. This is the ONLY endpoint that persists nodes -
    PUT /v1/workflows/{id} updates name/description only and silently ignores a flow.
    Always `workflow get` afterwards to verify the change persisted."""
    body = {"flow": read_json_arg(flow)}
    if flow_ui:
        body["flow_ui"] = read_json_arg(flow_ui)
    echo_response(api.post(f"/v1/workflows/{workflow_id}/save", json=body))


@workflow.command("test")
@click.argument("flow")
@click.argument("input_data")
@with_api_and_settings
def test_workflow(*, api: ApiClient, settings: Settings, flow: str, input_data: str):
    """Dry-run a flow with the given input, without saving or releasing."""
    echo_response(api.post("/v1/workflows/test", json={"flow": read_json_arg(flow), "input": read_json_arg(input_data)}))


@workflow.command("release")
@click.argument("workflow_id")
@with_api_and_settings
def release_workflow(*, api: ApiClient, settings: Settings, workflow_id: str):
    echo_response(api.post(f"/v1/workflows/{workflow_id}/release", json={}))


@workflow.command("versions")
@click.argument("workflow_id")
@with_api_and_settings
def list_workflow_versions(*, api: ApiClient, settings: Settings, workflow_id: str):
    echo_response(api.get(f"/v1/workflows/{workflow_id}/versions"))
