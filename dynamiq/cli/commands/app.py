import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

app = click.Group(name="app", help="Deploy workflows as apps, invoke them, inspect deployments")

ACCESS_TYPES = ("private", "public")
DEPLOYMENT_TYPES = ("serverless", "server_based")


@app.command("list")
@click.option("--workflow", "workflow_id", default=None, help="Only apps deployed from this workflow.")
@click.option("--include-archived", is_flag=True, help="Include archived apps.")
@pagination_options
@with_api_and_settings
def list_apps(
    *,
    api: ApiClient,
    settings: Settings,
    workflow_id: str | None,
    include_archived: bool,
    page,
    page_size,
    fetch_all,
    compact,
):
    """List apps in the current project (25 per page; --all for every one)."""
    params: dict = {"project_id": require_project(settings)}
    if workflow_id:
        params["workflow_id"] = workflow_id
    if include_archived:
        params["include_archived"] = "true"
    echo_list(api, "/v1/apps", params, page, page_size, fetch_all, compact)


@app.command("get")
@click.argument("app_id")
@with_api_and_settings
def get_app(*, api: ApiClient, settings: Settings, app_id: str):
    """Fetch one app, including its `hostname` - the endpoint you call to run it."""
    echo_response(api.get(f"/v1/apps/{app_id}"))


@app.command("deploy")
@click.option("--name", required=True, help="App name (letters, digits, spaces, . / _ ' -).")
@click.option("--workflow", "workflow_id", required=True, help="Workflow id to deploy.")
@click.option("--version", "workflow_version_id", default=None, help="Workflow version id (defaults to latest).")
@click.option("--description", default=None, help="Optional description (max 512 chars).")
@click.option(
    "--access",
    type=click.Choice(ACCESS_TYPES),
    default="private",
    show_default=True,
    help="private needs an Access Key to call; public is open.",
)
@click.option(
    "--type",
    "deployment_type",
    type=click.Choice(DEPLOYMENT_TYPES),
    default="serverless",
    show_default=True,
    help="server_based needs --autoscaling.",
)
@click.option("--runtime", "runtime_id", default=None, help="Runtime id (optional).")
@click.option("--autoscaling", default=None, help="Autoscaling JSON for server_based, e.g. '{\"min_replicas\": 1}'.")
@with_api_and_settings
def deploy_app(
    *,
    api: ApiClient,
    settings: Settings,
    name: str,
    workflow_id: str,
    workflow_version_id: str | None,
    description: str | None,
    access: str,
    deployment_type: str,
    runtime_id: str | None,
    autoscaling: str | None,
):
    """Create AND deploy an app from a workflow - this is how a workflow becomes callable.

    REQUIRED by the API: `name`, `project_id` (auto), `workflow_id`. The call is
    synchronous: the response already carries the app `id` and its `hostname`.

        dynamiq app deploy --name qna --workflow <workflow_id>

    A trigger needs an app id, so deploy before `dynamiq trigger create`.
    """
    if deployment_type == "server_based" and not autoscaling:
        raise click.ClickException("--type server_based requires --autoscaling, e.g. '{\"min_replicas\": 1}'.")

    config = read_json_arg(autoscaling) if autoscaling else {}
    body: dict = {
        "name": name,
        "project_id": require_project(settings),
        "workflow_id": workflow_id,
        "access_control": {"access_type": access},
        "deployment_config": {
            "deployment_type": deployment_type,
            "config": {"autoscaling": config} if deployment_type == "server_based" else {},
        },
    }
    if workflow_version_id:
        body["workflow_version_id"] = workflow_version_id
    if description:
        body["description"] = description
    if runtime_id:
        body["runtime_id"] = runtime_id

    response = api.post("/v1/apps", json=body)
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    data = response.json().get("data", {})
    click.echo(f"deployed app {data.get('id')} at {data.get('hostname')}", err=True)
    echo_response(response)


@app.command("redeploy")
@click.argument("app_id")
@click.option("--workflow", "workflow_id", required=True, help="Workflow id to deploy onto this app.")
@click.option("--version", "workflow_version_id", default=None, help="Workflow version id (defaults to latest).")
@click.option("--runtime", "runtime_id", default=None, help="Runtime id (optional).")
@with_api_and_settings
def redeploy_app(
    *,
    api: ApiClient,
    settings: Settings,
    app_id: str,
    workflow_id: str,
    workflow_version_id: str | None,
    runtime_id: str | None,
):
    """Push a new workflow version onto an EXISTING app (keeps its id and hostname).

    Use this instead of `deploy` when the app already exists - `workflow_id` is required.
    """
    body: dict = {"workflow_id": workflow_id}
    if workflow_version_id:
        body["workflow_version_id"] = workflow_version_id
    if runtime_id:
        body["runtime_id"] = runtime_id
    echo_response(api.post(f"/v1/apps/{app_id}/deploy", json=body))


@app.command("invoke")
@click.argument("app_id")
@click.argument("input_data")
@with_api_and_settings
def invoke_app(*, api: ApiClient, settings: Settings, app_id: str, input_data: str):
    """Run a deployed app through the management API, using only the PAT.

    INPUT_DATA is inline JSON or @file - your Input node's fields, e.g. '{"input": "hi"}'.
    It is wrapped as {"input": {...}} unless you already pass that shape. This avoids
    needing an Access Key for the app's own hostname.
    """
    payload = read_json_arg(input_data)
    body = payload if isinstance(payload, dict) and "input" in payload else {"input": payload}
    echo_response(api.post(f"/v1/apps/{app_id}/invoke", json=body))


@app.command("deployments")
@click.argument("app_id")
@with_api_and_settings
def list_deployments(*, api: ApiClient, settings: Settings, app_id: str):
    """Deployment history for an app - check here when a deploy looks stuck."""
    echo_response(api.get(f"/v1/apps/{app_id}/deployments"))


@app.command("delete")
@click.argument("app_id")
@click.confirmation_option(prompt="Delete this app?")
@with_api_and_settings
def delete_app(*, api: ApiClient, settings: Settings, app_id: str):
    """Delete an app permanently (its triggers go with it)."""
    echo_response(api.delete(f"/v1/apps/{app_id}"))
