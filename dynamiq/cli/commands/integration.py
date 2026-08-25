import json

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, require_project
from dynamiq.cli.config import Settings

integration = click.Group(
    name="integration",
    help="Connect third-party apps (Pipedream/MCP/Composio) and inspect connected accounts",
)

# Pipedream's hosted connect page. `POST /v1/pipedream/connect/tokens` returns only
# {token, expires_at} - the backend drops Pipedream's own connect_link_url - so the link
# is assembled here exactly the way the platform does it (connect link + ?app=<slug>).
PIPEDREAM_CONNECT_PAGE = "https://pipedream.com/_static/connect.html"

# Fields a connector's connect response may carry the user-facing URL in.
URL_FIELDS = ("connect_link_url", "url", "connect_url", "authorization_url", "redirect_url", "link")


def find_url(payload) -> str | None:
    """First URL-looking value anywhere in a connect response."""
    if isinstance(payload, dict):
        for key in URL_FIELDS:
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for value in payload.values():
            found = find_url(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = find_url(value)
            if found:
                return found
    return None


def echo_connect_link(url: str, expires_at: str | None = None) -> None:
    """Print the connect link the way the user needs to receive it."""
    click.echo("")
    click.echo("CONNECT LINK (send this to the user to open in a browser):")
    click.echo(url)
    if expires_at:
        click.echo(f"expires_at: {expires_at}")
    click.echo("Single-use and expires within minutes - mint a fresh one per attempt.")


@integration.command("list")
@with_api_and_settings
def list_connectors(*, api: ApiClient, settings: Settings):
    """List the connector catalog with your connected instances.

    Note each entry's `provider` (pipedream / mcp / composio / custom) - the connect flow
    differs per provider, and only pipedream accounts can be bound into workflow nodes.
    """
    echo_response(api.get("/v1/user/connectors"))


@integration.command("connect")
@click.argument("connector_id")
@with_api_and_settings
def connect(*, api: ApiClient, settings: Settings, connector_id: str):
    """Mint a connect URL for a catalog connector (user-scoped).

    POST /v1/user/connectors/{connector_id}/connect. The URL found in the response is
    printed as a CONNECT LINK - give it to the user; it is single-use and short-lived.
    For a connection a WORKFLOW node can use, use `connect-project` instead.
    """
    response = api.post(f"/v1/user/connectors/{connector_id}/connect", json={})
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    url = find_url(payload)
    if url:
        echo_connect_link(url)
    else:
        click.echo("No connect URL in the response - check the connector's provider.", err=True)


@integration.command("accounts")
@pagination_options
@with_api_and_settings
def list_accounts(*, api: ApiClient, settings: Settings, page, page_size, fetch_all):
    """List project-scoped Pipedream connected accounts.

    GET /v1/pipedream/connect/accounts?project_id=... (project REQUIRED).
    `account_id` (apn_...) is what a workflow node binds to; `external_user_id` is the
    project id, because connections are bound to the project rather than to one user.
    """
    echo_list(
        api, "/v1/pipedream/connect/accounts", {"project_id": require_project(settings)}, page, page_size, fetch_all
    )


@integration.command("connect-project")
@click.argument("app_slug")
@with_api_and_settings
def connect_project(*, api: ApiClient, settings: Settings, app_slug: str):
    """Mint a project-scoped Pipedream CONNECT LINK for APP_SLUG (e.g. notion, slack).

    POST /v1/pipedream/connect/tokens with {"project_id": ...} (the only body field, and
    it is REQUIRED). The response carries only {token, expires_at}, so the hosted page URL
    is assembled here as:

        https://pipedream.com/_static/connect.html?token=<token>&connectLink=true&app=<slug>

    Give the printed link to the user to open, then poll `dynamiq integration accounts`
    until the account shows up. Never open or reuse the link yourself.
    """
    response = api.post("/v1/pipedream/connect/tokens", json={"project_id": require_project(settings)})
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    data = payload.get("data", payload) if isinstance(payload, dict) else {}

    url = data.get("connect_link_url")
    if url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}app={app_slug}"
    else:
        token = data.get("token") or data.get("connect_token")
        if not token:
            raise click.ClickException(f"No connect token in the response: {json.dumps(payload)[:500]}")
        url = f"{PIPEDREAM_CONNECT_PAGE}?token={token}&connectLink=true&app={app_slug}"

    echo_connect_link(url, data.get("expires_at"))
