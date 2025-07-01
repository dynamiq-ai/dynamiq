import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api
from dynamiq.cli.config import Settings, save_settings


@click.group()
def org() -> click.Group:
    """Manage organisations"""
    pass


@org.command("list")
@with_api
def list_orgs(*, api: ApiClient, **__):
    response = api.get("/v1/orgs")
    if response.status_code == 200:
        orgs = response.json().get("data", [])
        for org in orgs:
            print(f"ID: {org['id']}, Name: {org['name']}")
    else:
        print("Failed to list organizations.")


@org.command("set")
@click.option("--id", "org_id", prompt=True, help="Organisation ID")
@with_api
def set_org(*, api: ApiClient, settings: Settings, org_id: str):
    org_id = org_id or input("Enter the organization ID: ")
    response = api.get(f"/v1/orgs/{org_id}")
    if response.status_code == 200:
        settings.org_id = org_id
        save_settings(settings)
        print(f"Current organization set to: {org_id}")
    else:
        print(f"Organization ID {org_id} was not found.")
