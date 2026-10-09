import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.access import mask_token
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import DYNAMIQ_BASE_URL, SOURCE_NOT_SET, Settings


@click.group(help="Manage configuration", invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        settings = Settings.load_settings()
        # Offer what is on disk, not the effective value: an env-supplied host accepted with
        # Enter would otherwise be written to the credentials file.
        default_host = settings.stored_value("api_host") or DYNAMIQ_BASE_URL
        host = click.prompt("Enter API host (press Enter to use default)", default=default_host, show_default=True)
        api_key = click.prompt("Enter API key", default="", show_default=False)

        if host:
            settings.api_host = host
        # Empty input keeps the stored key; assigning the current value would persist a token
        # that came only from the environment.
        if api_key:
            settings.api_key = api_key
        settings.save_settings()

        click.echo("\n✅ Configuration saved to .dynamiq/config.json")
        click.echo("These values will be used automatically when you run Dynamiq commands.")


@config.command("show")
@with_api_and_settings
def show_config(api: ApiClient, settings: Settings):
    rows = (
        ("DYNAMIQ API HOST", settings.api_host or "<not set>", "api_host"),
        ("DYNAMIQ API KEY", mask_token(settings.api_key), "api_key"),
        ("DYNAMIQ ORG ID", settings.org_id or "<not set>", "org_id"),
        ("DYNAMIQ PROJECT ID", settings.project_id or "<not set>", "project_id"),
    )

    click.echo("\nCurrent Dynamiq CLI configuration:")
    for label, value, field in rows:
        source = settings.source_of(field)
        click.echo(f"{label}: {value}" + ("" if source == SOURCE_NOT_SET else f" ({source})"))
