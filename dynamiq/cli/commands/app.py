import json
import os
from urllib.parse import quote

import click
import requests

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

app = click.Group(name="app", help="Deploy workflows as apps, invoke them, inspect deployments")

ACCESS_TYPES = ("private", "public")

# The app's own hostname is a different service from the management API, with a different
# credential: an Access Key (created in the UI), never the PAT. Everything below that talks to
# a deployed app goes through these two helpers so the rules live in one place.


def access_key() -> str:
    """The Access Key from the environment, or an error explaining where to get one."""
    key = os.getenv("DYNAMIQ_ACCESS_KEY")
    if not key:
        raise click.ClickException(
            "DYNAMIQ_ACCESS_KEY is not set. Create an Access Key in the UI and export it in "
            "your shell - the app hostname does not accept the management PAT."
        )
    return key


def app_endpoint(api: ApiClient, app_id: str) -> str:
    """Base URL of the deployed app, asked for rather than guessed."""
    response = api.get(f"/v1/apps/{app_id}")
    if not ok(response):
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")

    hostname = (response.json().get("data") or {}).get("hostname")
    if not hostname:
        raise click.ClickException(f"App {app_id} has no hostname yet - deploy it first.")

    return hostname if hostname.startswith("http") else f"https://{hostname}"


def call_app(url: str, key: str, *, method: str = "POST", json_body: dict | None = None) -> None:
    """Call the app hostname and print the body; non-2xx becomes a readable error."""
    click.echo(f"{method} {url}", err=True)
    result = requests.request(
        method,
        url,
        json=json_body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=300,
    )
    if not ok(result):
        raise click.ClickException(f"HTTP {result.status_code}: {result.text.strip()[:2000]}")
    try:
        click.echo(json.dumps(result.json(), indent=2, ensure_ascii=False))
    except ValueError:
        click.echo(result.text)


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
    if not ok(response):
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
    """Run a deployed app by POSTing to its own hostname - the way an integration calls it.

    INPUT_DATA is inline JSON or @file - your Input node's fields, e.g. '{"input": "hi"}'
    for an Input node with one field named `input`. It is wrapped as {"input": {...}}
    unless you already pass the envelope, i.e. an object whose `input` is itself an object.

    The hostname is read from `GET /v1/apps/{app_id}` and authenticated with an Access Key
    from the DYNAMIQ_ACCESS_KEY environment variable (create one in the UI; the management
    PAT is not accepted there). This is the endpoint the app actually serves - the
    management API's /v1/apps/{id}/invoke reverse-proxies through synapse and can fail
    while the app itself is healthy, so it is not used.
    """
    payload = read_json_arg(input_data)
    # The app expects {"input": <node fields>}. `{"input": "hi"}` is the common case of an
    # Input node with one field called `input`, so only an object-valued `input` counts as
    # the envelope already being there - otherwise that call would arrive a level too high.
    enveloped = isinstance(payload, dict) and isinstance(payload.get("input"), dict)
    body = payload if enveloped else {"input": payload}

    key = access_key()
    call_app(app_endpoint(api, app_id), key, json_body=body)


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


@app.command("requirements")
@click.argument("app_id")
@with_api_and_settings
def app_requirements(*, api: ApiClient, settings: Settings, app_id: str):
    """What this app needs each end user to connect, with the titles they will read."""
    call_app(f"{app_endpoint(api, app_id)}/v1/requirements", access_key(), method="GET")


@app.command("requirements-status")
@click.argument("app_id")
@click.argument("user_id")
@with_api_and_settings
def app_requirements_status(*, api: ApiClient, settings: Settings, app_id: str, user_id: str):
    """Whether ONE end user has connected everything, and what is still `unsatisfied`.

    Poll this after sending them a connect link. A run for a user with anything unsatisfied
    fails before the workflow is even built.
    """
    url = f"{app_endpoint(api, app_id)}/v1/requirements/status?user_id={quote(user_id)}"
    call_app(url, access_key(), method="GET")


@app.command("connect-token")
@click.argument("app_id")
@click.argument("user_id")
@with_api_and_settings
def app_connect_token(*, api: ApiClient, settings: Settings, app_id: str, user_id: str):
    """Mint a CONNECT LINK for ONE end user so they attach their own accounts.

    Give the returned `url` to that person and nobody else: it is scoped to them and this app,
    is single-use and expires within minutes. Never open or reuse it yourself.

    This is NOT `integration connect-project`, which connects YOUR project's shared account.
    Sending that one to an end user looks like it worked and leaves the run unsatisfied,
    because nothing ties the account to them or to the requirement.
    """
    call_app(
        f"{app_endpoint(api, app_id)}/v1/connect/tokens",
        access_key(),
        json_body={"user_id": user_id},
    )
