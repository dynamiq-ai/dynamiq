import json
import mimetypes
import os
from pathlib import Path

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options
from dynamiq.cli.config import Settings
from dynamiq.storages.artifact.base import ArtifactKind, infer_kind

KIND_CHOICE = click.Choice([k.value for k in ArtifactKind])

artifact = click.Group(
    name="artifact",
    help="Artifacts: versioned deliverables with a link (publish, update, list, get)",
)


def _upload(api: ApiClient, method: str, path: str, file_path: str, fields: dict, headers: dict | None = None):
    """Send one file as multipart with its metadata fields; None-valued fields are left out."""
    name = os.path.basename(file_path)
    media_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    data = {k: json.dumps(v) if isinstance(v, dict) else v for k, v in fields.items() if v is not None}
    with open(file_path, "rb") as handle:
        files = {"file": (name, handle, media_type)}
        if method == "POST":
            return api.post(path, data=data, files=files, headers=headers, retry=True)
        return api.put(path, data=data, files=files, headers=headers)


def _source() -> dict | None:
    """Provenance from the sandbox environment the platform injects, when running inside one."""
    conversation_id = os.environ.get("DYNAMIQ_CONVERSATION_ID")
    return {"conversation_id": conversation_id, "client": "cli"} if conversation_id else {"client": "cli"}


@artifact.command("publish")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--title", help="Display title. REQUIRED for a new artifact.")
@click.option("--kind", type=KIND_CHOICE, help="Artifact kind; inferred from the file extension when omitted.")
@click.option("--summary", help="One line: what this version is or what changed.")
@click.option("--artifact-id", help="Add a new version to this artifact instead of creating one.")
@click.option("--project-id", help="Project to own a new artifact. Omitted: you own it.")
@with_api_and_settings
def publish_artifact(
    *,
    api: ApiClient,
    settings: Settings,
    file_path: str,
    title: str | None,
    kind: str | None,
    summary: str | None,
    artifact_id: str | None,
    project_id: str | None,
):
    """Publish a file as an artifact and print it, including its `id` and `url`.

    Use this for deliverables someone will open and share: an HTML report, a Markdown doc, an
    SVG/Mermaid diagram, a CSV. Re-publishing the same deliverable? Pass `--artifact-id` so it
    becomes v2 of the same artifact rather than a new one.

    Inside a platform sandbox the credentials come from the environment
    (DYNAMIQ_API_TOKEN, DYNAMIQ_API_BASE_URL) and nothing needs configuring.

    For an agent, the same store is `artifact_store` on the agent node:

        "artifact_store": {"enabled": true,
                           "backend": {"type": "dynamiq.storages.artifact.DynamiqArtifactStore"}}
    """
    if artifact_id:
        _publish_version(api, artifact_id, file_path, title=title, summary=summary, if_match=None)
        return
    if not title:
        raise click.UsageError("--title is required when creating a new artifact.")
    name = Path(file_path).name
    fields = {
        "name": name,
        "title": title,
        "kind": kind or infer_kind(name).value,
        "summary": summary,
        "project_id": project_id,
        "source": _source(),
    }
    echo_response(_upload(api, "POST", "/v1/artifacts", file_path, fields))


@artifact.command("update")
@click.argument("artifact_id")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--title", help="New display title.")
@click.option("--summary", help="One line: what changed.")
@click.option("--if-match", help="Checksum or version you last saw; the update fails if it has moved on.")
@with_api_and_settings
def update_artifact(
    *,
    api: ApiClient,
    settings: Settings,
    artifact_id: str,
    file_path: str,
    title: str | None,
    summary: str | None,
    if_match: str | None,
):
    """Add a new version to an artifact from a file. Earlier versions stay readable."""
    _publish_version(api, artifact_id, file_path, title=title, summary=summary, if_match=if_match)


def _publish_version(api: ApiClient, artifact_id: str, file_path: str, *, title, summary, if_match) -> None:
    fields = {"title": title, "summary": summary, "source": _source()}
    headers = {"If-Match": if_match} if if_match else None
    echo_response(_upload(api, "PUT", f"/v1/artifacts/{artifact_id}", file_path, fields, headers=headers))


@artifact.command("list")
@click.option("--kind", type=KIND_CHOICE, help="Only artifacts of this kind.")
@click.option("--query", help="Filter by title or name.")
@click.option("--project-id", help="Only artifacts owned by this project.")
@pagination_options
@with_api_and_settings
def list_artifacts(*, api: ApiClient, settings: Settings, kind, query, project_id, page, page_size, fetch_all, compact):
    """List artifacts you can see, newest first. Check here before publishing a duplicate."""
    params = {"kind": kind, "query": query, "project_id": project_id}
    echo_list(api, "/v1/artifacts", {k: v for k, v in params.items() if v}, page, page_size, fetch_all, compact)


@artifact.command("get")
@click.argument("artifact_id")
@click.option("--version", "version", type=int, help="Version number. Defaults to the latest.")
@click.option("--out", type=click.Path(dir_okay=False, writable=True), help="Write the content to this file.")
@with_api_and_settings
def get_artifact(*, api: ApiClient, settings: Settings, artifact_id: str, version: int | None, out: str | None):
    """Print an artifact's metadata, or save its content with `--out`."""
    params = {"version": version} if version is not None else None
    if not out:
        echo_response(api.get(f"/v1/artifacts/{artifact_id}", params=params))
        return

    if version is None:
        meta = api.get(f"/v1/artifacts/{artifact_id}")
        if not 200 <= meta.status_code < 300:
            echo_response(meta)
            return
        body = meta.json()
        data = body.get("data", body) if isinstance(body, dict) else {}
        latest = data.get("latest") or data.get("latest_version") or {}
        version = latest.get("version")
        if version is None:
            raise click.ClickException("The API did not report a latest version for this artifact.")

    response = api.get(f"/v1/artifacts/{artifact_id}/versions/{version}/content")
    if not 200 <= response.status_code < 300:
        echo_response(response)
        return
    Path(out).write_bytes(response.content)
    click.echo(f"Saved v{version} of {artifact_id} to {out} ({len(response.content)} bytes)")
