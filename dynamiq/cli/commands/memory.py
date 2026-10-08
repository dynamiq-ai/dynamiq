import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

memory = click.Group(
    name="memory",
    help="Platform-hosted agent memory (the `dynamiq.memory.backends.Dynamiq` backend)",
)


@memory.command("list")
@pagination_options
@with_api_and_settings
def list_memories(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List memories in the current project.

    Check here before creating another: one memory per conversational agent is usually right,
    and a duplicate silently splits an agent's history across two stores.
    """
    echo_list(api, "/v1/memories", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@memory.command("create")
@click.argument("payload")
@with_api_and_settings
def create_memory(*, api: ApiClient, settings: Settings, payload: str):
    """Create a memory. REQUIRED: `name`; `project_id` is filled in automatically.

    The returned `id` is the `memory_id` for an agent node's memory backend:

        "memory": {"backend": {"type": "dynamiq.memory.backends.Dynamiq",
                               "memory_id": "<id>"}}

    Unlike the SDK's InMemory backend this survives redeploys, and the platform supplies the
    connection itself - the flow needs nothing but the id.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/memories", json=body))


@memory.command("get")
@click.argument("memory_id")
@with_api_and_settings
def get_memory(*, api: ApiClient, settings: Settings, memory_id: str):
    """Fetch one memory."""
    echo_response(api.get(f"/v1/memories/{memory_id}"))


def _scope_params(user_id: str, session_id: str | None) -> dict:
    params = {"user_id": user_id}
    if session_id:
        params["session_id"] = session_id
    return params


user_id_option = click.option(
    "--user-id", required=True, help="The `user_id` the agent run carried. Items are stored per user."
)
session_id_option = click.option(
    "--session-id", default=None, help="Narrow to one conversation (the run's `session_id`)."
)


@memory.command("items")
@click.argument("memory_id")
@user_id_option
@session_id_option
@pagination_options
@with_api_and_settings
def list_memory_items(
    *,
    api: ApiClient,
    settings: Settings,
    memory_id: str,
    user_id: str,
    session_id: str | None,
    page,
    page_size,
    fetch_all,
    compact,
):
    """The stored messages for one user - proof an agent is actually remembering.

    Items are stored per `user_id` (and optionally per `session_id`), so pass the same ids
    the run carried. Empty after a run usually means the run carried no `user_id`/`session_id`,
    so memory never switched on. That is silent at run time; this is where you see it.
    """
    echo_list(
        api,
        f"/v1/memories/{memory_id}/items",
        _scope_params(user_id, session_id),
        page,
        page_size,
        fetch_all,
        compact,
    )


@memory.command("clear")
@click.argument("memory_id")
@user_id_option
@session_id_option
@click.confirmation_option(prompt="Delete the stored messages for this user?")
@with_api_and_settings
def delete_memory_items(*, api: ApiClient, settings: Settings, memory_id: str, user_id: str, session_id: str | None):
    """Delete one user's stored messages (or one session's, with --session-id).

    The memory itself is kept and its id stays valid.
    """
    echo_response(api.delete(f"/v1/memories/{memory_id}/items", params=_scope_params(user_id, session_id)))


@memory.command("delete")
@click.argument("memory_id")
@click.confirmation_option(prompt="Delete this memory?")
@with_api_and_settings
def delete_memory(*, api: ApiClient, settings: Settings, memory_id: str):
    """Delete the memory permanently. Any flow pointing at this `memory_id` will break."""
    echo_response(api.delete(f"/v1/memories/{memory_id}"))
