import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api
from dynamiq.cli.config import Settings, load_settings

service = click.Group(name="service", help="Manage services")


@service.command("list")
@with_api
def list_services(*, api: ApiClient, settings: Settings):
    project_id = settings.project_id
    if not project_id:
        click.echo("No project ID found. Please set the current project ID.")
        return

    response = api.get(f"/v1/services?project_id={project_id}")
    if response.status_code == 200:
        services = response.json().get("data", [])
        click.echo(f"{'ID':<40} {'Name'}")
        for service in services:
            click.echo(f"{service['id']:<40} {service['name']}")
    else:
        click.echo("Failed to list services.")


@service.command("get")
@click.option("--id", "service_id", prompt=True)
@with_api
def get_service(*, api: ApiClient, settings: Settings, service_id: str):
    response = api.get(f"/v1/services/{service_id}")
    if response.status_code == 200:
        service_info = response.json().get("data", [])
        for key, value in service_info.items():
            if not isinstance(value, dict):
                click.echo(f"{key}: {value}")
    else:
        click.echo(f"Failed to find service ID {service_id}.")


@service.command("create")
@click.option("--name", prompt=True)
@with_api
def create_service(*, api: ApiClient, settings: Settings, name: str):
    project_id = settings.project_id

    if not project_id:
        click.echo("No project ID found. Please set the current project ID.")
        return

    payload = {"name": name, "project_id": project_id}
    try:
        _require_project()
        response = api.post(
            "/v1/services",
            json=payload,
        )
        click.echo(f"Service '{name}' created successfully: {response.json()}")
    except Exception as e:
        click.echo(f"Failed to create service. Details:{str(e)}")


def _archive_directory(path: Path) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
        archive_path = temp_file.name
    shutil.make_archive(archive_path[:-7], "gztar", root_dir=path)
    return archive_path


@service.command("deploy")
@click.option("--id", "service_id", prompt=True)
@click.option("--source", prompt=True)
@click.option("--docker-context", default=".", show_default=True)
@click.option("--docker-file", default="Dockerfile", show_default=True)
@click.option("--cpu-requests", default="100m", show_default=True)
@click.option("--memory-requests", default="100Mi", show_default=True)
@click.option("--cpu-limits", default="500m", show_default=True)
@click.option("--memory-limits", default="200Mi", show_default=True)
@click.option(
    "--env",
    multiple=True,
    type=(str, str),
    metavar="NAME VALUE",
    help="Environment variables (repeatable)",
)
@with_api
def deploy_service(
    *,
    api: ApiClient,
    settings: Settings,
    service_id: str,
    source: Path,
    docker_context: str,
    docker_file: str,
    cpu_requests: str,
    memory_requests: str,
    cpu_limits: str,
    memory_limits: str,
    env: tuple[tuple[str, str], ...],
):
    archive_path = ""
    try:
        click.echo("• Archiving source directory …")
        archive_path = _archive_directory(source)
        project_id = settings.project_id
        if not project_id:
            click.echo("No project ID found. Please set the current project ID.")
            return False

        service_id = service_id
        api_endpoint = f"/v1/services/{service_id}/deploy"

        data = {
                "docker": {"file": str(docker_file), "context": str(docker_context)},
                "resources": {
                    "requests": {
                        "cpu": str(cpu_requests),
                        "memory": str(memory_requests),
                    },
                    "limits": {"cpu": str(cpu_limits), "memory": str(memory_limits)},
                },
            }
        if env:
            for i, (name, value) in enumerate(env):
                data[f"env.{i}.name"] = name
                data[f"env.{i}.value"] = value

        try:
            with open(archive_path, "rb") as archive_file:
                files = {"source": ("archive.tar.gz", archive_file, "application/x-tar")}
                response = api.post(api_endpoint, data={"data": json.dumps(data)}, files=files)
                click.echo("Deployment successfully started.")
                return response.status_code == 200
        except Exception:
            logging.error("Deployment failed.")
            return False

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)


def _require_project() -> None:
    settings = load_settings()
    if not settings.project_id:
        click.echo("❌ No project selected. Run `dynamiq project set` first.", err=True)
        click.get_current_context().exit(1)
