import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options
from dynamiq.cli.config import Settings

connection = click.Group(name="connection", help="Inspect connections (LLM/tool credentials used by nodes)")


@connection.command("list")
@click.option("--type", "types", multiple=True, help="Filter by type, repeatable (e.g. dynamiq.connections.OpenAI).")
@click.option(
    "--include-system/--no-include-system", default=True, show_default=True, help="Include system connections."
)
@click.option("--all-projects", is_flag=True, help="Do not scope to the current project.")
@pagination_options
@with_api_and_settings
def list_connections(
    *,
    api: ApiClient,
    settings: Settings,
    types: tuple[str, ...],
    include_system: bool,
    all_projects: bool,
    page,
    page_size,
    fetch_all,
    compact,
):
    """List connections and their ids.

    A workflow node's `connection` field is one of these ids - never invent it. Look for
    `dynamiq.connections.OpenAI` (or Anthropic, etc.) for an LLM, and e.g.
    `dynamiq.connections.Exa` / `...Tavily` for a search tool.

    `project_id` is OPTIONAL here (unlike most endpoints); it is sent by default to scope
    the list, use --all-projects to omit it.
    """
    params: dict = {}
    if not all_projects and settings.project_id:
        params["project_id"] = settings.project_id
    if include_system:
        params["include_system"] = "true"
    if types:
        params["type"] = list(types)
    echo_list(api, "/v1/connections", params, page, page_size, fetch_all, compact)


@connection.command("get")
@click.argument("connection_id")
@with_api_and_settings
def get_connection(*, api: ApiClient, settings: Settings, connection_id: str):
    """Fetch one connection (credentials are never returned)."""
    echo_response(api.get(f"/v1/connections/{connection_id}"))
