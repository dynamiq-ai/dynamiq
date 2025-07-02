import click

from dynamiq.cli.config import DYNAMIQ_BASE_URL, load_settings, save_settings


@click.group(help="Manage configuration", invoke_without_command=True)
@click.pass_context
def config(ctx: click.Context):
    if ctx.invoked_subcommand is None:
        host = click.prompt("Enter API host (press Enter to use default)", default=DYNAMIQ_BASE_URL, show_default=True)
        token = click.prompt("Enter API key")

        settings = load_settings()
        settings.api_host = host
        settings.api_key = token
        save_settings(settings)

        click.echo("\n✅ Configuration saved to .dynamiq/config.json")
        click.echo("These values will be used automatically when you run Dynamiq commands.")


@config.command("show")
def show_config():
    settings = load_settings()
    host = settings.api_host or "<not set>"
    api_key = settings.api_key or "<not set>"
    org_id = settings.org_id or "<not set>"
    project_id = settings.project_id or "<not set>"
    masked_key = api_key[:4] + "..." if api_key != "<not set>" else api_key  # nosec B105

    click.echo("\nCurrent Dynamiq CLI configuration:")
    click.echo(f"DYNAMIQ API HOST: {host}")
    click.echo(f"DYNAMIQ API KEY: {masked_key}")
    click.echo(f"DYNAMIQ ORG ID: {org_id}")
    click.echo(f"DYNAMIQ PROJECT ID: {project_id}")
