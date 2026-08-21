import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_response
from dynamiq.cli.config import Settings

integration = click.Group(
    name="integration",
    help="Connect third-party apps (Pipedream/MCP/Composio) and inspect connected accounts",
)

PIPEDREAM_CONNECT_PAGE = "https://pipedream.com/_static/connect.html"


@integration.command("list")
@with_api_and_settings
def list_connectors(*, api: ApiClient, settings: Settings):
    """List the connector catalog with your connected instances (note each entry's provider)."""
    echo_response(api.get("/v1/user/connectors"))


@integration.command("connect")
@click.argument("connector_id")
@with_api_and_settings
def connect(*, api: ApiClient, settings: Settings, connector_id: str):
    """Mint a connect URL for a catalog connector. Give the URL to the user to open -
    it is one-time and expires in minutes; mint a fresh one per attempt."""
    echo_response(api.post(f"/v1/user/connectors/{connector_id}/connect", json={}))


@integration.command("accounts")
@with_api_and_settings
def list_accounts(*, api: ApiClient, settings: Settings):
    """List project-scoped Pipedream connected accounts (account_id = authProvisionId
    for workflow nodes; external_user_id = the project id)."""
    if not settings.project_id:
        raise click.ClickException("No project set. Run `dynamiq project set` first.")
    echo_response(api.get(f"/v1/pipedream/connect/accounts?project_id={settings.project_id}"))


@integration.command("connect-project")
@click.argument("app_slug")
@with_api_and_settings
def connect_project(*, api: ApiClient, settings: Settings, app_slug: str):
    """Mint a project-scoped Pipedream connect link for APP_SLUG (for workflow nodes).
    The API returns only a token; the hosted page URL is assembled from it."""
    if not settings.project_id:
        raise click.ClickException("No project set. Run `dynamiq project set` first.")
    response = api.post("/v1/pipedream/connect/tokens", json={"project_id": settings.project_id})
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    data = payload.get("data", payload)
    click.echo(f"{PIPEDREAM_CONNECT_PAGE}?token={data['token']}&connectLink=true&app={app_slug}")
    if data.get("expires_at"):
        click.echo(f"expires_at: {data['expires_at']}", err=True)
