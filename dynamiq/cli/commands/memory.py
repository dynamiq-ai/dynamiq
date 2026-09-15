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


@memory.command("items")
@click.argument("memory_id")
@pagination_options
@with_api_and_settings
def list_memory_items(*, api: ApiClient, settings: Settings, memory_id: str, page, page_size, fetch_all, compact):
    """The stored messages - proof an agent is actually remembering.

    Empty after a run usually means the run carried no `user_id`/`session_id`, so memory
    never switched on. That is silent at run time; this is where you see it.
    """
    echo_list(api, f"/v1/memories/{memory_id}/items", None, page, page_size, fetch_all, compact)


@memory.command("clear")
@click.argument("memory_id")
@click.confirmation_option(prompt="Delete every stored message in this memory?")
@with_api_and_settings
def delete_memory_items(*, api: ApiClient, settings: Settings, memory_id: str):
    """Delete the stored messages but keep the memory itself (its id stays valid)."""
    echo_response(api.delete(f"/v1/memories/{memory_id}/items"))


@memory.command("delete")
@click.argument("memory_id")
@click.confirmation_option(prompt="Delete this memory?")
@with_api_and_settings
def delete_memory(*, api: ApiClient, settings: Settings, memory_id: str):
    """Delete the memory permanently. Any flow pointing at this `memory_id` will break."""
    echo_response(api.delete(f"/v1/memories/{memory_id}"))
