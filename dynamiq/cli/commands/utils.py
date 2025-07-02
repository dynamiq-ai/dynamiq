import click

from dynamiq.cli.client import HTTPError
from dynamiq.cli.commands.context import DynamiqCtx, pass_dctx

from .config import config
from .org import org
from .project import project
from .service import service


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(None, "--version", prog_name="dynamiq")
@click.option("--verbose", "-v", is_flag=True, help="Enable chatty output (for debugging / CI).")
@pass_dctx
def cli(dctx: DynamiqCtx, verbose: bool) -> None:
    pass


cli.add_command(org, name="org")
cli.add_command(org, name="orgs")
cli.add_command(project, name="project")
cli.add_command(project, name="projects")
cli.add_command(service, name="service")
cli.add_command(service, name="services")
cli.add_command(config)


def main() -> None:
    try:
        cli(obj=DynamiqCtx())
    except HTTPError as exc:
        click.echo(f"❌ {exc}", err=True)
        raise SystemExit(1) from exc
