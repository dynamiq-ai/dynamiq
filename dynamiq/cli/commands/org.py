import click

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.access import access_failure
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

org = click.Group(name="org", help="Manage organizations")


@org.command("list")
@with_api_and_settings
def list_orgs(*, api: ApiClient, **__):
    response = api.get("/v1/orgs")
    if ok(response):
        orgs = response.json().get("data", [])
        click.echo(f"{'ID':<40} {'Name'}")
        for organization in orgs:
            click.echo(f"{organization['id']:<40} {organization['name']}")
    else:
        click.echo("Failed to list organizations.")


@org.command("set")
@click.option("--id", "org_id", prompt=True, help="Organisation ID")
@with_api_and_settings
def set_org(*, api: ApiClient, settings: Settings, org_id: str):
    response = api.get(f"/v1/orgs/{org_id}")
    if not ok(response):
        raise click.ClickException(access_failure(settings, "Organization", org_id, response.status_code))
    settings.org_id = org_id
    settings.save_settings()
    click.echo(f"Current organization set to: {org_id}")
