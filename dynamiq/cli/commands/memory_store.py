import click

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.memory import _reject_blank
from dynamiq.cli.commands.workflow import (
    echo_list,
    echo_response,
    fetch_data,
    pagination_options,
    read_json_arg,
    require_project,
)
from dynamiq.cli.config import Settings

memory_store = click.Group(
    name="memory-store",
    help="Memory stores: path-keyed notes an agent keeps per user (the `DynamiqMemoryStore` backend)",
)

user_id_option = click.option(
    "--user-id",
    required=True,
    callback=_reject_blank,
    help="The end user whose files to address. Files are stored per user.",
)
path_argument = click.argument("path", callback=_reject_blank)


@memory_store.command("list")
@pagination_options
@with_api_and_settings
def list_memory_stores(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List memory stores in the current project."""
    echo_list(api, "/v1/memory-stores", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact)


@memory_store.command("create")
@click.argument("payload")
@with_api_and_settings
def create_memory_store(*, api: ApiClient, settings: Settings, payload: str):
    """Create a memory store. REQUIRED: `name`; optional `description`; `project_id` is filled in automatically.

    The returned `id` is the `memory_store_id` of a `DynamiqMemoryStore` backend.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/memory-stores", json=body))


@memory_store.command("get")
@click.argument("store_id")
@with_api_and_settings
def get_memory_store(*, api: ApiClient, settings: Settings, store_id: str):
    """Fetch one memory store."""
    echo_response(api.get(f"/v1/memory-stores/{store_id}"))


# The update endpoint replaces both fields: an omitted description is cleared, so the CLI
# always sends both.
MEMORY_STORE_UPDATE_FIELDS = {"name", "description"}


@memory_store.command("update")
@click.argument("store_id")
@click.argument("payload")
@with_api_and_settings
def update_memory_store(*, api: ApiClient, settings: Settings, store_id: str, payload: str):
    """Rename a memory store or change its description.

    Accepted fields: `name`, `description`. Any field left out keeps its current value (the
    store is fetched first and your fields are laid over it). Pass `"description": null` to
    clear the description.
    """
    changes = read_json_arg(payload)
    unknown = sorted(set(changes) - MEMORY_STORE_UPDATE_FIELDS)
    if unknown:
        raise click.ClickException(
            f"cannot update {', '.join(unknown)}; accepted fields: {', '.join(sorted(MEMORY_STORE_UPDATE_FIELDS))}"
        )
    current = fetch_data(api, f"/v1/memory-stores/{store_id}")
    body = {field: current.get(field) for field in MEMORY_STORE_UPDATE_FIELDS} | changes
    echo_response(api.put(f"/v1/memory-stores/{store_id}", json=body))


@memory_store.command("delete")
@click.argument("store_id")
@click.confirmation_option(prompt="Delete this memory store and every file in it?")
@with_api_and_settings
def delete_memory_store(*, api: ApiClient, settings: Settings, store_id: str):
    """Delete the memory store permanently. Any flow pointing at this store id will break."""
    echo_response(api.delete(f"/v1/memory-stores/{store_id}"))


@memory_store.command("files")
@click.argument("store_id")
@user_id_option
@click.option(
    "--path",
    "prefix",
    default=None,
    callback=_reject_blank,
    help="Only files whose path starts with this prefix (e.g. `notes/`).",
)
@with_api_and_settings
def list_memory_store_files(*, api: ApiClient, settings: Settings, store_id: str, user_id: str, prefix: str | None):
    """List one user's files (path, size, updated_at), sorted by path.

    The API returns every matching file in one response; there are no pages.
    """
    params = {"user_id": user_id}
    if prefix is not None:
        params["path"] = prefix
    echo_response(api.get(f"/v1/memory-stores/{store_id}/files", params=params))


@memory_store.command("read")
@click.argument("store_id")
@path_argument
@user_id_option
@click.option("--raw", is_flag=True, help="Print only the file content, not the JSON envelope.")
@with_api_and_settings
def read_memory_store_file(*, api: ApiClient, settings: Settings, store_id: str, path: str, user_id: str, raw: bool):
    """Print one file of a user, with its content."""
    response = api.get(f"/v1/memory-stores/{store_id}/files/content", params={"user_id": user_id, "path": path})
    if raw and ok(response):
        click.echo((response.json().get("data") or {}).get("content", ""), nl=False)
        return
    echo_response(response)


@memory_store.command("write")
@click.argument("store_id")
@path_argument
@user_id_option
@click.option("--content", default=None, help="The file content.")
@click.option(
    "--from-file",
    "source",
    type=click.File("r", encoding="utf-8"),
    default=None,
    help="Read the content from a local file (`-` for stdin).",
)
@with_api_and_settings
def write_memory_store_file(
    *, api: ApiClient, settings: Settings, store_id: str, path: str, user_id: str, content: str | None, source
):
    """Create or overwrite one file of a user. Give the content with --content or --from-file.

    Paths are relative (no leading `/`, no `.` or `..` segments). A file holds at most 8192
    characters and a user at most 100 files per store; the server rejects anything beyond.
    """
    if (content is None) == (source is None):
        raise click.UsageError("give exactly one of --content or --from-file.")
    if source is not None:
        content = source.read()
    if not content:
        raise click.UsageError("content must not be empty.")
    body = {"path": path, "content": content, "user_id": user_id}
    echo_response(api.put(f"/v1/memory-stores/{store_id}/files", json=body))


@memory_store.command("delete-file")
@click.argument("store_id")
@path_argument
@user_id_option
@click.confirmation_option(prompt="Delete this file?")
@with_api_and_settings
def delete_memory_store_file(*, api: ApiClient, settings: Settings, store_id: str, path: str, user_id: str):
    """Delete one file of a user. The store and the user's other files are kept."""
    echo_response(api.delete(f"/v1/memory-stores/{store_id}/files", params={"user_id": user_id, "path": path}))
