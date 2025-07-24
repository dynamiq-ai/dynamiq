import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

project = click.Group(name="project", help="Manage projects")


@project.command("list")
@with_api_and_settings
def list_projects(*, api: ApiClient, settings: Settings):
    org_id = settings.org_id
    if not org_id:
        click.echo("No organization ID found. Please set the current organization ID.")
        return
    response = api.get(f"/v1/projects?org_id={org_id}")
    if response.status_code == 200:
        projects = response.json().get("data", [])
        click.echo(f"{'ID':<40} {'Name'}")
        for project in projects:
            click.echo(f"{project['id']:<40} {project['name']}")
    else:
        click.echo("Failed to list projects")


@project.command("set")
@click.option("--id", "proj_id", prompt=True, help="Project ID")
@with_api_and_settings
def set_project(*, api: ApiClient, settings: Settings, proj_id: str):
    org_id = settings.org_id
    if not org_id:
        click.echo("No organization ID found. Please set the current organization ID.")
        return

    response = api.get(f"/v1/projects/{proj_id}?org_id={org_id}")
    if response.status_code == 200:
        settings.project_id = proj_id
        settings.save_settings()
        click.echo(f"Current project set to: {proj_id}")
    else:
        click.echo(f"Project ID {proj_id} was not found.")
