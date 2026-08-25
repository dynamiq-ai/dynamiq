import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

dataset = click.Group(
    name="dataset",
    help="Evaluation datasets: create, add items, release a version",
)


@dataset.command("list")
@pagination_options
@with_api_and_settings
def list_datasets(*, api: ApiClient, settings: Settings, page, page_size, fetch_all):
    """List datasets in the current project (25 per page; --all for every one)."""
    echo_list(api, "/v1/datasets", {"project_id": require_project(settings)}, page, page_size, fetch_all)


@dataset.command("create")
@click.argument("payload")
@with_api_and_settings
def create_dataset(*, api: ApiClient, settings: Settings, payload: str):
    """Create a dataset. REQUIRED: `name`, `project_id` (auto); optional `description`.

    A first draft version is created automatically - see `latest_version_id`.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/datasets", json=body))


@dataset.command("get")
@click.argument("dataset_id")
@with_api_and_settings
def get_dataset(*, api: ApiClient, settings: Settings, dataset_id: str):
    """Fetch one dataset."""
    echo_response(api.get(f"/v1/datasets/{dataset_id}"))


@dataset.command("versions")
@click.argument("dataset_id")
@pagination_options
@with_api_and_settings
def list_versions(*, api: ApiClient, settings: Settings, dataset_id: str, page, page_size, fetch_all):
    """List a dataset's versions. NOTE: this is `/v1/dataset-versions?dataset_id=...`
    (keyed by dataset_id, NOT project_id)."""
    echo_list(api, "/v1/dataset-versions", {"dataset_id": dataset_id}, page, page_size, fetch_all)


@dataset.command("version-create")
@click.argument("dataset_id")
@click.option("--schema", "schema", default=None, help="Schema JSON (inline or @file); inferred from items if omitted.")
@with_api_and_settings
def create_version(*, api: ApiClient, settings: Settings, dataset_id: str, schema: str | None):
    """Create a new DRAFT version under a dataset (the only nested dataset route).

    Optional schema, e.g.
      {"type": "object", "required": true, "properties": {
         "input": {"type": "any", "required": true},
         "ground_truth_answer": {"type": "any", "required": false}}}
    Column names chosen here are what evaluation selectors reference as $.dataset.<col>.
    """
    body = {"schema": read_json_arg(schema)} if schema else {}
    echo_response(api.post(f"/v1/datasets/{dataset_id}/versions", json=body))


@dataset.command("items-add")
@click.argument("dataset_version_id")
@click.argument("items")
@with_api_and_settings
def add_items(*, api: ApiClient, settings: Settings, dataset_version_id: str, items: str):
    """Add items to a DRAFT version - flat path `/v1/dataset-versions/{id}/items`.

    ITEMS is inline JSON or @file, either a bare list of row objects or {"items": [...]}.
    This is one batch call; there is no per-item endpoint. Released versions are immutable
    (fork them first).
    """
    payload = read_json_arg(items)
    body = payload if isinstance(payload, dict) and "items" in payload else {"items": payload}
    echo_response(api.post(f"/v1/dataset-versions/{dataset_version_id}/items", json=body))


@dataset.command("items")
@click.argument("dataset_version_id")
@pagination_options
@with_api_and_settings
def list_items(*, api: ApiClient, settings: Settings, dataset_version_id: str, page, page_size, fetch_all):
    """List a version's items (25 per page; --all for every item)."""
    echo_list(api, "/v1/dataset-items", {"dataset_version_id": dataset_version_id}, page, page_size, fetch_all)


@dataset.command("release")
@click.argument("dataset_version_id")
@with_api_and_settings
def release_version(*, api: ApiClient, settings: Settings, dataset_version_id: str):
    """Release a draft version (no body). REQUIRED before an evaluation can use it -
    otherwise `evaluation start` fails with "Dataset version is not released."."""
    echo_response(api.post(f"/v1/dataset-versions/{dataset_version_id}/release"))


@dataset.command("fork")
@click.argument("dataset_version_id")
@with_api_and_settings
def fork_version(*, api: ApiClient, settings: Settings, dataset_version_id: str):
    """Fork a released (immutable) version into a new draft you can edit."""
    echo_response(api.post(f"/v1/dataset-versions/{dataset_version_id}/fork"))


@dataset.command("download")
@click.argument("dataset_version_id")
@click.option("--format", "fmt", type=click.Choice(["json", "jsonl"]), default="json", show_default=True)
@click.option("--out", "out_path", default=None, help="Write to this file instead of stdout.")
@with_api_and_settings
def download_version(*, api: ApiClient, settings: Settings, dataset_version_id: str, fmt: str, out_path: str | None):
    """Download a whole version in one call (json or jsonl)."""
    response = api.get(f"/v1/dataset-versions/{dataset_version_id}/download", params={"format": fmt})
    if response.status_code != 200:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text.strip()[:2000]}")
    if out_path:
        with open(out_path, "w") as f:
            f.write(response.text)
        click.echo(f"wrote {out_path}")
    else:
        click.echo(response.text)
