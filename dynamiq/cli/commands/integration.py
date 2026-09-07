import json
import uuid

import click
import requests

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import (
    echo_list,
    echo_response,
    pagination_options,
    read_json_arg,
    require_project,
)
from dynamiq.cli.config import Settings

integration = click.Group(
    name="integration",
    help="Connect third-party apps (Pipedream/MCP/Composio) and inspect connected accounts",
)

# Pipedream's hosted connect page. `POST /v1/pipedream/connect/tokens` returns only
# {token, expires_at} - the backend drops Pipedream's own connect_link_url - so the link
# is assembled here exactly the way the platform does it (connect link + ?app=<slug>).
PIPEDREAM_CONNECT_PAGE = "https://pipedream.com/_static/connect.html"

# Pipedream's own REST API. The platform UI talks to it DIRECTLY from the browser - Dynamiq
# only mints the token - so every lookup below is the same request the workflow builder makes,
# and the answers are therefore identical to what the builder would have stored.
#
# Component reads are connect-scoped, so they sit under /v1/connect and are authorized with a
# connect token. The APP record is the exception and does NOT come from here at all - the
# builder takes it from the platform's own catalogue (`integration app` below), because that
# entry carries the action and trigger lists Pipedream's /v1/apps/<slug> does not.
PIPEDREAM_CONNECT_API = "https://api.pipedream.com/v1/connect"
PIPEDREAM_NODE_TYPE = "dynamiq.nodes.tools.Pipedream"

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
    if not ok(response):
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
def list_accounts(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List project-scoped Pipedream connected accounts.

    GET /v1/pipedream/connect/accounts?project_id=... (project REQUIRED).
    `account_id` (apn_...) is what a workflow node binds to; `external_user_id` is the
    project id, because connections are bound to the project rather than to one user.
    """
    echo_list(
        api,
        "/v1/pipedream/connect/accounts",
        {"project_id": require_project(settings)},
        page,
        page_size,
        fetch_all,
        compact,
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
    if not ok(response):
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


def pipedream_token(api: ApiClient, settings: Settings) -> str:
    """Mint a project-scoped Pipedream connect token for one lookup.

    This is a short-lived credential. It is returned to the caller and used as a bearer for the
    Pipedream request, and it is never echoed - none of the commands below print it, and none
    accept one on the command line, where it would end up in shell history and in an agent's
    streamed transcript.
    """
    response = api.post("/v1/pipedream/connect/tokens", json={"project_id": require_project(settings)})
    if not ok(response):
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    token = data.get("token") or data.get("connect_token")
    if not token:
        raise click.ClickException("No connect token in the response from /v1/pipedream/connect/tokens.")
    return token


def pipedream_call(url: str, token: str, *, params: dict | None = None, json_body: dict | None = None) -> None:
    """Call Pipedream with a freshly minted token and print the JSON body."""
    method = "POST" if json_body is not None else "GET"
    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise click.ClickException(f"Pipedream request failed: {exc}") from exc

    if not ok(response):
        # The token is in the request header, never in the response, so this is safe to show.
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    try:
        click.echo(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except ValueError:
        click.echo(response.text)


@integration.command("app")
@click.argument("app_slug")
@with_api_and_settings
def pipedream_app(*, api: ApiClient, settings: Settings, app_slug: str):
    """The app record for APP_SLUG (notion, slack, github, ...) as the BUILDER stores it.

    GET /v1/pipedream/connect/apps, filtered to one entry. This is the platform's own
    catalogue, and it is deliberately not Pipedream's `/v1/apps/<slug>`: the catalogue entry
    carries `actions[]` and `triggers[]` alongside `id`, `name`, `name_slug`, `img_src` and
    `description`, and the whole entry is what the editor keeps as a node's `pipedreamApp`.
    It is also the only source of the logo the canvas draws.

    Verified against a builder-authored workflow: its stored `pipedreamApp` has this shape,
    down to the nested action and trigger lists.
    """
    response = api.get("/v1/pipedream/connect/apps", params={"page_size": 500})
    if not ok(response):
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    apps = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(apps, list):
        raise click.ClickException(f"unexpected /v1/pipedream/connect/apps response: {json.dumps(payload)[:500]}")

    wanted = app_slug.lower()
    match = next(
        (
            entry
            for entry in apps
            if isinstance(entry, dict)
            and wanted in {str(entry.get(field, "")).lower() for field in ("name_slug", "id", "name")}
        ),
        None,
    )
    if match is None:
        near = sorted(
            str(entry.get("name_slug") or entry.get("name"))
            for entry in apps
            if isinstance(entry, dict) and wanted in str(entry.get("name_slug") or entry.get("name") or "").lower()
        )[:10]
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise click.ClickException(f"no app matching {app_slug!r} in the catalogue.{hint}")

    click.echo(json.dumps(match, indent=2, ensure_ascii=False))


@integration.command("components")
@click.argument("app_slug")
@click.option("--triggers", is_flag=True, help="List the app's TRIGGERS (sources) instead of its actions.")
@click.option("--q", default=None, help="Filter by search term.")
@with_api_and_settings
def pipedream_components(*, api: ApiClient, settings: Settings, app_slug: str, triggers: bool, q: str | None):
    """List an app's components, so you can find an action's exact `key`.

    GET /v1/connect/components?app=<slug> (or /v1/connect/triggers with --triggers). The `key`
    printed here is what `integration component` and a tool node's `action_id` both want -
    never guess it from the action's display name.
    """
    path = "triggers" if triggers else "components"
    params = {"app": app_slug}
    if q:
        params["q"] = q
    pipedream_call(f"{PIPEDREAM_CONNECT_API}/{path}", pipedream_token(api, settings), params=params)


@integration.command("component")
@click.argument("key")
@with_api_and_settings
def pipedream_component(*, api: ApiClient, settings: Settings, key: str):
    """The FULL component record for KEY (e.g. notion-create-page).

    GET /v1/connect/components/<key>. This is the exact call the workflow builder makes the
    moment you pick an action, and its answer is what it stores as `pipedreamComponent`:
    `key`, `version`, `name`, `description`, and `configurable_props` with every label,
    description, `remoteOptions` and `reloadProps` flag resolved.

    Build the tool node from THIS, not from the component's source on GitHub. A node whose
    `input_props` were hand-written runs, but the editor has no component to render, so it
    shows no logo, no account picker and no configuration form.
    """
    pipedream_call(f"{PIPEDREAM_CONNECT_API}/components/{key}", pipedream_token(api, settings))


@integration.command("component-props")
@click.argument("key")
@click.argument("configured_props")
@click.option("--dynamic-props-id", default=None, help="Carry forward the id from a previous reload.")
@with_api_and_settings
def pipedream_component_props(
    *, api: ApiClient, settings: Settings, key: str, configured_props: str, dynamic_props_id: str | None
):
    """Re-resolve a component's props after setting one marked `reloadProps`.

    POST /v1/connect/components/props. CONFIGURED_PROPS is the JSON you have pinned so far
    (inline or @file) - at minimum the app prop, e.g.
    '{"notion": {"authProvisionId": "apn_..."}}'.

    Some props only exist once the account is known, and others only then learn their real
    option list. The response carries a NEW `configurable_props` array plus a `dynamic_props_id`:
    both belong on the node (`input_props` regenerated from the array, `dynamic_props_id` set),
    exactly as the builder does when you fill such a field.
    """
    body = {
        "external_user_id": require_project(settings),
        "id": key,
        "configured_props": read_json_arg(configured_props),
    }
    if dynamic_props_id:
        body["dynamic_props_id"] = dynamic_props_id
    pipedream_call(f"{PIPEDREAM_CONNECT_API}/components/props", pipedream_token(api, settings), json_body=body)


@integration.command("component-options")
@click.argument("key")
@click.argument("prop_name")
@click.argument("configured_props")
@click.option("--query", default=None, help="Search term to filter the options by.")
@click.option("--dynamic-props-id", default=None, help="From a prior `component-props` reload.")
@with_api_and_settings
def pipedream_component_options(
    *,
    api: ApiClient,
    settings: Settings,
    key: str,
    prop_name: str,
    configured_props: str,
    query: str | None,
    dynamic_props_id: str | None,
):
    """Resolve the real option list for one `remoteOptions` prop.

    POST /v1/connect/components/configure. This is how the builder fills a dropdown such as
    Notion's parent page or Slack's channel: the values live in the user's own account, so
    they cannot be known from the component schema alone.

    Use it to turn a name the user gave you ("the To-Do List page") into the id the prop
    actually takes, instead of pinning a guess that fails at run time.
    """
    body = {
        "external_user_id": require_project(settings),
        "id": key,
        "prop_name": prop_name,
        "configured_props": read_json_arg(configured_props),
    }
    if query:
        body["query"] = query
    if dynamic_props_id:
        body["dynamic_props_id"] = dynamic_props_id
    pipedream_call(f"{PIPEDREAM_CONNECT_API}/components/configure", pipedream_token(api, settings), json_body=body)


def pipedream_fetch(url: str, token: str, *, params: dict | None = None, json_body: dict | None = None):
    """Same call as `pipedream_call`, returning the payload instead of printing it."""
    method = "POST" if json_body is not None else "GET"
    try:
        response = requests.request(
            method,
            url,
            params=params,
            json=json_body,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise click.ClickException(f"Pipedream request failed: {exc}") from exc
    if not ok(response):
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    payload = response.json()
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def map_configurable_props(props: list) -> list:
    """`type` becomes `type_`, and `alert` props are dropped.

    Not cosmetic: the SDK renames it back on load, so a prop sent as `type` is not
    recognised and the tool is handed to the agent without it.
    """
    mapped = []
    for prop in props or []:
        if not isinstance(prop, dict):
            continue
        rest = {k: v for k, v in prop.items() if k != "type"}
        rest["type_"] = prop.get("type")
        if rest["type_"] == "alert":
            continue
        mapped.append(rest)
    return mapped
