import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

service = click.Group(name="service", help="Manage services")


@service.command("list")
@with_api_and_settings
def list_services(*, api: ApiClient, settings: Settings):
    project_id = settings.project_id
    if not project_id:
        click.echo("No project ID found. Please set the current project ID.")
        return

    response = api.get(f"/v1/services?project_id={project_id}")
    if response.status_code == 200:
        services = response.json().get("data", [])
        max_name_len = max(len(s["name"]) for s in services) + 2 if services else 40
        click.echo(f"{'ID':<40} {'Name':<{max_name_len}} {'Host'}")
        for service in services:
            click.echo(f"{service['id']:<40} {service['name']:<{max_name_len}} {service['hostname']}")
    else:
        click.echo("Failed to list services.")


@service.command("get")
@click.option("--id", "service_id", prompt=True)
@with_api_and_settings
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
@with_api_and_settings
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
        if response.status_code == 200:
            click.echo(f"Service '{name}' created successfully: {response.json()}")
        else:
            click.echo("Failed to create service.")
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
@click.option("--memory-requests", default="256Mi", show_default=True)
@click.option("--cpu-limits", default="200m", show_default=True)
@click.option("--memory-limits", default="512Mi", show_default=True)
@click.option(
    "--env",
    multiple=True,
    type=(str, str),
    metavar="NAME VALUE",
    help="Environment variables (repeatable)",
)
@with_api_and_settings
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
            data["env"] = []
            for i, (name, value) in enumerate(env):
                data["env"].append({"name": name, "value": value})

        try:
            with open(archive_path, "rb") as archive_file:
                files = {"source": ("archive.tar.gz", archive_file, "application/x-tar")}
                response = api.post(api_endpoint, data={"data": json.dumps(data)}, files=files)
                if response.status_code == 200:
                    click.echo("Deployment successfully started.")
                else:
                    click.echo("Failed to deploy service.")
        except Exception:
            logging.error("Deployment failed.")
            return False

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        if archive_path and os.path.exists(archive_path):
            os.remove(archive_path)


def _require_project() -> None:
    settings = Settings.load_settings()
    if not settings.project_id:
        click.echo("❌ No project selected. Run `dynamiq project set` first.", err=True)
        click.get_current_context().exit(1)


@service.command("status")
@click.option("--id", "service_id", prompt=True)
@with_api_and_settings
def get_service_status(*, api: ApiClient, settings: Settings, service_id: str):
    response = api.get(f"/v1/services/{service_id}/deployments")
    if response.status_code == 200:
        service_info = response.json().get("data", [])
        for key, value in service_info[0].items() if service_info else []:
            if not isinstance(value, dict):
                click.echo(f"{key}: {value}")
    else:
        click.echo(f"Failed to find service ID {service_id}.")
