import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api
from dynamiq.cli.config import Settings, save_settings


org = click.Group(name="org", help="Manage organizations")


@org.command("list")
@with_api
def list_orgs(*, api: ApiClient, **__):
    response = api.get("/v1/orgs")
    if response.status_code == 200:
        orgs = response.json().get("data", [])
        click.echo(f"{'ID':<40} {'Name'}")
        for organization in orgs:
            click.echo(f"{organization['id']:<40} {organization['name']}")
    else:
        click.echo("Failed to list organizations.")


@org.command("set")
@click.option("--id", "org_id", prompt=True, help="Organisation ID")
@with_api
def set_org(*, api: ApiClient, settings: Settings, org_id: str):
    response = api.get(f"/v1/orgs/{org_id}")
    if response.status_code == 200:
        settings.org_id = org_id
        save_settings(settings)
        click.echo(f"Current organization set to: {org_id}")
    else:
        click.echo(f"Organization ID {org_id} was not found.")
