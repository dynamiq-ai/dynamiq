import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from dynamiq.artifacts import ArtifactKind, default_mime_type, infer_kind
from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options
from dynamiq.cli.config import Settings

KIND_CHOICE = click.Choice([k.value for k in ArtifactKind])

# The platform keeps an artifact's latest 50 versions, so one page holds every version it has.
MAX_VERSIONS = 50

artifact = click.Group(
    name="artifact",
    help="Artifacts: versioned deliverables with a link (publish, update, list, get, share)",
)


def _upload(api: ApiClient, path: str, file_path: str, fields: dict, *, headers: dict | None, retry: bool):
    """Send one file with its fields as the JSON `data` part; None-valued fields are left out."""
    name = os.path.basename(file_path)
    mime_type = default_mime_type(ArtifactKind(fields.get("kind") or infer_kind(name)), name)
    data = {"data": json.dumps({k: v for k, v in fields.items() if v is not None})}
    with open(file_path, "rb") as handle:
        return api.post(path, headers=headers, data=data, files={"file": (name, handle, mime_type)}, retry=retry)


def _version_id(api: ApiClient, artifact_id: str, version: int) -> str:
    """The id of an artifact's version by its number."""
    response = api.get(f"/v1/artifacts/{artifact_id}/versions", params={"page_size": MAX_VERSIONS})
    if not ok(response):
        echo_response(response)
    for item in response.json().get("data") or []:
        if item.get("version") == version:
            return item["id"]
    raise click.ClickException(f"Artifact {artifact_id} has no version {version}.")


@artifact.command("publish")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", help="Display name. REQUIRED for a new artifact.")
@click.option(
    "--kind",
    type=KIND_CHOICE,
    help="Artifact kind; inferred from the file extension when omitted. Use 'bundle' for a zipped site.",
)
@click.option("--description", help="One line: what this version is or what changed.")
@click.option("--entry", "entry_path", help="Page a bundle opens on, relative to the zip's root (default index.html).")
@click.option("--artifact-id", help="Add a new version to this artifact instead of creating one.")
@click.option("--if-match", help="With --artifact-id: the version id you last saw; fails if a newer one exists.")
@click.option("--store-id", help="Artifact store that owns a new artifact. Omitted: you own it.")
@click.option("--user-id", help="End user of an app that a new artifact belongs to, within --store-id.")
@with_api_and_settings
def publish_artifact(
    *,
    api: ApiClient,
    settings: Settings,
    file_path: str,
    name: str | None,
    kind: str | None,
    description: str | None,
    entry_path: str | None,
    artifact_id: str | None,
    if_match: str | None,
    store_id: str | None,
    user_id: str | None,
):
    """Publish a file as an artifact and print it, including its `id` and `url`.

    Use this for deliverables someone will open and share: an HTML report, a Markdown doc, an
    SVG/Mermaid diagram, a CSV, or a zipped site with `--kind bundle`. Re-publishing the same
    deliverable? Pass `--artifact-id` so it becomes v2 of the same artifact rather than a new one.

    Inside a platform sandbox the credentials come from the environment
    (DYNAMIQ_API_TOKEN, DYNAMIQ_API_BASE_URL) and the artifact is yours. With a personal
    access token, publish into an artifact store with `--store-id`.

    For an agent, the same backend is `artifacts` on the agent node:

        "artifacts": {"enabled": true,
                      "backend": {"type": "dynamiq.artifacts.backends.Dynamiq"}}
    """
    if artifact_id:
        if store_id or user_id:
            raise click.UsageError("--store-id and --user-id apply to a new artifact only.")
        _publish_version(
            api, artifact_id, file_path, name=name, description=description, entry_path=entry_path, if_match=if_match
        )
        return
    if not name:
        raise click.UsageError("--name is required when creating a new artifact.")
    if if_match:
        raise click.UsageError("--if-match applies with --artifact-id only.")
    if user_id and not store_id:
        raise click.UsageError("--user-id requires --store-id.")
    file_name = Path(file_path).name
    fields = {
        "store_id": store_id,
        "user_id": user_id,
        "file_name": file_name,
        "name": name,
        "description": description,
        "kind": kind or infer_kind(file_name).value,
        "entry_path": entry_path,
    }
    # No retry: a create that reached the platform before the response was lost would be duplicated.
    echo_response(_upload(api, "/v1/artifacts/upload", file_path, fields, headers=None, retry=False))


@artifact.command("update")
@click.argument("artifact_id")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", help="New display name.")
@click.option("--description", help="One line: what changed.")
@click.option("--entry", "entry_path", help="Page a bundle opens on, relative to the zip's root.")
@click.option("--if-match", help="The version id you last saw; the update fails if a newer one exists.")
@with_api_and_settings
def update_artifact(
    *,
    api: ApiClient,
    settings: Settings,
    artifact_id: str,
    file_path: str,
    name: str | None,
    description: str | None,
    entry_path: str | None,
    if_match: str | None,
):
    """Add a new version to an artifact from a file. Earlier versions stay readable."""
    _publish_version(
        api, artifact_id, file_path, name=name, description=description, entry_path=entry_path, if_match=if_match
    )


def _publish_version(api: ApiClient, artifact_id: str, file_path: str, *, name, description, entry_path, if_match):
    fields = {"name": name, "description": description, "entry_path": entry_path}
    headers = {"If-Match": f'"{if_match}"'} if if_match else None
    # Retried: the platform returns the latest version instead of adding one for the same content.
    echo_response(
        _upload(api, f"/v1/artifacts/{artifact_id}/versions/upload", file_path, fields, headers=headers, retry=True)
    )


@artifact.command("list")
@click.option("--store-id", help="List this artifact store's artifacts. Omitted: your own.")
@click.option("--user-id", help="Only the artifacts of this end user of an app, within --store-id.")
@click.option("--kind", type=KIND_CHOICE, help="Only artifacts of this kind.")
@pagination_options
@with_api_and_settings
def list_artifacts(*, api: ApiClient, settings: Settings, store_id, user_id, kind, page, page_size, fetch_all, compact):
    """List artifacts, most recently updated first. Check here before publishing a duplicate."""
    if user_id and not store_id:
        raise click.UsageError("--user-id requires --store-id.")
    params = {"store_id": store_id, "user_id": user_id, "kind": kind}
    echo_list(api, "/v1/artifacts", {k: v for k, v in params.items() if v}, page, page_size, fetch_all, compact)


@artifact.command("get")
@click.argument("artifact_id")
@click.option("--version", "version", type=int, help="Version number. Defaults to the latest.")
@click.option("--out", type=click.Path(dir_okay=False, writable=True), help="Write the content to this file.")
@with_api_and_settings
def get_artifact(*, api: ApiClient, settings: Settings, artifact_id: str, version: int | None, out: str | None):
    """Print an artifact's metadata, or a version's with `--version`, or save its content with `--out`."""
    if version is None and not out:
        echo_response(api.get(f"/v1/artifacts/{artifact_id}"))
        return

    version_id = "latest" if version is None else _version_id(api, artifact_id, version)
    if not out:
        echo_response(api.get(f"/v1/artifacts/{artifact_id}/versions/{version_id}"))
        return

    response = api.get(f"/v1/artifacts/{artifact_id}/versions/{version_id}/download")
    if not ok(response):
        echo_response(response)
    Path(out).write_bytes(response.content)
    label = f"v{version}" if version is not None else "the latest version"
    click.echo(f"Saved {label} of {artifact_id} to {out} ({len(response.content)} bytes)")


@artifact.command("share")
@click.argument("artifact_id")
@click.option("--pin", "pinned_version", type=int, help="Version the link shows. Omitted: it follows the latest.")
@click.option("--expires-in-days", type=click.IntRange(min=1), help="Days until the link stops working.")
@click.option("--revoke", is_flag=True, help="Revoke the link instead; the artifact becomes private.")
@with_api_and_settings
def share_artifact(
    *,
    api: ApiClient,
    settings: Settings,
    artifact_id: str,
    pinned_version: int | None,
    expires_in_days: int | None,
    revoke: bool,
):
    """Print a link anyone can open, or revoke it with `--revoke`. Sharing after a revoke makes a new link."""
    if revoke:
        if pinned_version is not None or expires_in_days:
            raise click.UsageError("--pin and --expires-in-days do not apply with --revoke.")
        echo_response(api.delete(f"/v1/artifacts/{artifact_id}/share"))
        return

    body = {}
    if pinned_version is not None:
        body["pinned_version_id"] = _version_id(api, artifact_id, pinned_version)
    if expires_in_days:
        body["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()
    # Retried: sharing again updates the same link.
    echo_response(api.post(f"/v1/artifacts/{artifact_id}/share", json=body, retry=True))
