import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api
from dynamiq.cli.config import Settings, save_settings


@click.group()
def project() -> None:
    """Manage projects"""


@project.command("list")
@with_api
def list_projects(*, api: ApiClient, settings: Settings):
    org_id = settings.org_id
    if not org_id:
        print("No organization ID found. Please set the current organization ID.")
        return
    response = api.get(f"/v1/projects?org_id={org_id}")
    if response.status_code == 200:
        projects = response.json().get("data", [])
        for project in projects:
            print(f"ID: {project['id']}, Name: {project['name']}")
    else:
        print("Failed to list projects")


@project.command("set")
@click.option("--id", "proj_id", prompt=True, help="Project ID")
@with_api
def set_project(*, api: ApiClient, settings: Settings, proj_id: str):
    project_id = proj_id or input("Enter the project ID: ")
    org_id = settings.org_id
    if not org_id:
        print("No organization ID found. Please set the current organization ID.")
        return

    response = api.get(f"/v1/projects/{project_id}?org_id={org_id}")
    if response.status_code == 200:
        settings.project_id = project_id
        save_settings(settings)
        print(f"Current project set to: {project_id}")
    else:
        print(f"Project ID {project_id} was not found.")
