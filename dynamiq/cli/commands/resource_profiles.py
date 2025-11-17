import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

profile = click.Group(name="resource-profiles", help="Manage profiles")


@profile.command("list")
@click.option(
    "--purpose",
    default="service",
    required=False,
    type=click.Choice(["inference", "service", "fine_tuning"], case_sensitive=True),
)
@with_api_and_settings
def list_resource_profiles(*, api: ApiClient, settings: Settings, purpose: str):
    response = api.get(f"/v1/resource-profiles?purpose={purpose}&page_size=100&sort=sort_order")
    if response.status_code == 200:
        profiles = response.json().get("data", [])
        click.echo(f"{len(profiles)} resource(s) found.")
        max_name_len = max(len(p["name"]) for p in profiles) + 2 if profiles else 40
        max_description_len = max(len(p["description"]) for p in profiles) + 2 if profiles else 40
        click.echo(f"{'ID':<40} {'Name':<{max_name_len}} {'Description':<{max_description_len}}")
        for profile in profiles:
            click.echo(
                f"{profile['id']:<40} {profile['name']:<{max_name_len}} "
                f"{profile['description']:<{max_description_len}}"
            )

    else:
        click.echo("Failed to list resource profiles.")
