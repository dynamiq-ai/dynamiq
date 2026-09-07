import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

knowledgebase = click.Group(
    name="knowledge-base",
    help="Knowledge bases: create, upload files, inspect items, search",
)


@knowledgebase.command("list")
@pagination_options
@with_api_and_settings
def list_knowledgebases(*, api: ApiClient, settings: Settings, page, page_size, fetch_all, compact):
    """List knowledge bases in the current project."""
    echo_list(
        api, "/v1/knowledgebases", {"project_id": require_project(settings)}, page, page_size, fetch_all, compact
    )


@knowledgebase.command("create")
@click.argument("payload")
@with_api_and_settings
def create_knowledgebase(*, api: ApiClient, settings: Settings, payload: str):
    """Create a knowledge base. REQUIRED: `name`; `project_id` is filled in automatically.

    `flow` AND `flow_ui` are also REQUIRED - a knowledge base is backed by an ingestion
    workflow (convert -> split -> embed -> write) and there is no server-side default, so a
    name alone is rejected. Generate both rather than copying them from another knowledge
    base; the skill's `kb_flow` script emits the standard pipeline for a given embedder
    connection.

    Uploading files is a separate step - creating a knowledge base does not put anything in it.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/knowledgebases", json=body))


@knowledgebase.command("get")
@click.argument("knowledgebase_id")
@with_api_and_settings
def get_knowledgebase(*, api: ApiClient, settings: Settings, knowledgebase_id: str):
    """Fetch one knowledge base, including its ingestion flow."""
    echo_response(api.get(f"/v1/knowledgebases/{knowledgebase_id}"))


@knowledgebase.command("update")
@click.argument("knowledgebase_id")
@click.argument("payload")
@with_api_and_settings
def update_knowledgebase(*, api: ApiClient, settings: Settings, knowledgebase_id: str, payload: str):
    """Update a knowledge base (name, description, ingestion flow)."""
    echo_response(api.put(f"/v1/knowledgebases/{knowledgebase_id}", json=read_json_arg(payload)))


@knowledgebase.command("upload")
@click.argument("knowledgebase_id")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@with_api_and_settings
def upload_files(*, api: ApiClient, settings: Settings, knowledgebase_id: str, paths: tuple[str, ...]):
    """Upload one or more files into a knowledge base.

    The endpoint takes a multipart form with a repeated `files` field, so several files go up
    in one call. Ingestion is ASYNCHRONOUS: a 2xx here means the files were accepted, not that
    they are searchable. Poll `knowledge-base items <id>` until each item's status settles
    before telling anyone the content is available.
    """
    handles = [open(path, "rb") for path in paths]  # noqa: SIM115 - closed below
    try:
        files = [("files", (path.rsplit("/", 1)[-1], handle)) for path, handle in zip(paths, handles)]
        # Not retried: requests reads the handles to EOF, and nothing rewinds them, so a
        # retried upload sends empty parts and the API answers 2xx over an empty file.
        echo_response(api.post(f"/v1/knowledgebases/{knowledgebase_id}/upload",
                               files=files, retry=False))
    finally:
        for handle in handles:
            handle.close()


@knowledgebase.command("items")
@click.argument("knowledgebase_id")
@pagination_options
@with_api_and_settings
def list_items(*, api: ApiClient, settings: Settings, knowledgebase_id: str, page, page_size, fetch_all, compact):
    """List the items in a knowledge base - this is where ingestion status shows up."""
    echo_list(api, f"/v1/knowledgebases/{knowledgebase_id}/items", None, page, page_size, fetch_all, compact)


@knowledgebase.command("item")
@click.argument("item_id")
@with_api_and_settings
def get_item(*, api: ApiClient, settings: Settings, item_id: str):
    """Fetch one item by its own id (not the knowledge base id)."""
    echo_response(api.get(f"/v1/knowledgebase-items/{item_id}"))


@knowledgebase.command("item-reprocess")
@click.argument("item_id")
@with_api_and_settings
def reprocess_item(*, api: ApiClient, settings: Settings, item_id: str):
    """Re-run ingestion for one item - the fix for an item stuck in a failed state."""
    echo_response(api.post(f"/v1/knowledgebase-items/{item_id}/reprocess"))


@knowledgebase.command("item-delete")
@click.argument("item_id")
@click.confirmation_option(prompt="Delete this item?")
@with_api_and_settings
def delete_item(*, api: ApiClient, settings: Settings, item_id: str):
    """Delete one item and its indexed chunks."""
    echo_response(api.delete(f"/v1/knowledgebase-items/{item_id}"))


@knowledgebase.command("search")
@click.argument("knowledgebase_id")
@click.argument("query")
@click.option("--limit", type=int, default=None, help="Max documents to return.")
@click.option("--threshold", type=float, default=None, help="Minimum similarity score (0-1).")
@with_api_and_settings
def vector_search(
    *, api: ApiClient, settings: Settings, knowledgebase_id: str, query: str, limit: int | None, threshold: float | None
):
    """Vector-search a knowledge base - the fastest proof that ingestion actually worked.

    Empty results right after an upload usually mean ingestion has not finished, not that the
    content is missing. Check `knowledge-base items <id>` before concluding anything.
    """
    body: dict = {"query": query}
    if limit is not None:
        body["limit"] = limit
    if threshold is not None:
        body["similarity_threshold"] = threshold
    echo_response(api.post(f"/v1/knowledgebases/{knowledgebase_id}/vector-search", json=body))


@knowledgebase.command("sources")
@click.argument("knowledgebase_id")
@pagination_options
@with_api_and_settings
def list_sources(*, api: ApiClient, settings: Settings, knowledgebase_id: str, page, page_size, fetch_all, compact):
    """List connected sources (e.g. Google Drive) that sync into this knowledge base."""
    echo_list(api, f"/v1/knowledgebases/{knowledgebase_id}/sources", None, page, page_size, fetch_all, compact)


@knowledgebase.command("source-add")
@click.argument("knowledgebase_id")
@click.argument("payload")
@with_api_and_settings
def add_source(*, api: ApiClient, settings: Settings, knowledgebase_id: str, payload: str):
    """Connect a syncing source. REQUIRED: `name`, `provider`, `config`.

    A source keeps ingesting on its own; `upload` is a one-off. Use a source when the user
    wants a folder kept in sync, not when they hand you a file.
    """
    echo_response(api.post(f"/v1/knowledgebases/{knowledgebase_id}/sources", json=read_json_arg(payload)))


@knowledgebase.command("source-sync")
@click.argument("source_id")
@with_api_and_settings
def sync_source(*, api: ApiClient, settings: Settings, source_id: str):
    """Trigger a sync now instead of waiting for the schedule."""
    echo_response(api.post(f"/v1/knowledgebase-sources/{source_id}/sync"))


@knowledgebase.command("delete")
@click.argument("knowledgebase_id")
@click.confirmation_option(prompt="Delete this knowledge base and everything in it?")
@with_api_and_settings
def delete_knowledgebase(*, api: ApiClient, settings: Settings, knowledgebase_id: str):
    """Delete a knowledge base. Any workflow retrieving from it will break."""
    echo_response(api.delete(f"/v1/knowledgebases/{knowledgebase_id}"))
