import click

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

profile = click.Group(name="resource-profiles", help="Manage profiles")


@profile.command("list")
@click.option(
    "--purpose",
    default="service",
    required=False,
    type=click.Choice(["inference", "service", "fine_tuning", "database"], case_sensitive=True),
)
@click.option(
    "--sort-by",
    default="sort_order",
    required=False,
    type=click.Choice(["name", "sort_order", "description"], case_sensitive=True),
)
@click.option("--page-size", default=100, show_default=True, type=int)
@with_api_and_settings
def list_resource_profiles(*, api: ApiClient, settings: Settings, purpose: str, sort_by: str, page_size: int):
    response = api.get(f"/v1/resource-profiles?purpose={purpose}&page_size={page_size}&sort={sort_by}")
    if ok(response):
        profiles = response.json().get("data", [])
        click.echo(f"{len(profiles)} resource(s) found.")
        # The API leaves an unset description out of the response.
        descriptions = [p.get("description") or "" for p in profiles]
        max_name_len = max(len(p["name"]) for p in profiles) + 2 if profiles else 40
        max_description_len = max(len(d) for d in descriptions) + 2 if profiles else 40
        click.echo(f"{'ID':<40} {'Name':<{max_name_len}} {'Description':<{max_description_len}}")
        for profile, description in zip(profiles, descriptions):
            click.echo(f"{profile['id']:<40} {profile['name']:<{max_name_len}} {description:<{max_description_len}}")

    else:
        click.echo("Failed to list resource profiles.")
