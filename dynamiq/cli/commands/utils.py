import click

from dynamiq.cli.client import HTTPError
from dynamiq.cli.commands.context import DynamiqCtx, pass_dctx

from .app import app
from .config import config
from .connection import connection
from .dataset import dataset
from .deployment import database, finetuning, inference
from .evaluation import evaluation
from .integration import integration
from .knowledgebase import knowledgebase
from .memory import memory
from .org import org
from .project import project
from .resource_profiles import profile
from .service import service
from .skill import skill
from .trigger import trigger
from .voice import voice
from .workflow import workflow


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
cli.add_command(profile, name="resource-profiles")
cli.add_command(profile, name="resource-profile")
cli.add_command(workflow, name="workflow")
cli.add_command(workflow, name="workflows")
cli.add_command(integration, name="integration")
cli.add_command(integration, name="integrations")
cli.add_command(evaluation, name="evaluation")
cli.add_command(evaluation, name="evaluations")
cli.add_command(dataset, name="dataset")
cli.add_command(dataset, name="datasets")
cli.add_command(connection, name="connection")
cli.add_command(connection, name="connections")
cli.add_command(memory, name="memory")
cli.add_command(memory, name="memories")
cli.add_command(knowledgebase, name="knowledge-base")
cli.add_command(knowledgebase, name="knowledge-bases")
cli.add_command(skill, name="skill")
cli.add_command(skill, name="skills")
cli.add_command(finetuning, name="fine-tuning")
cli.add_command(inference, name="inference")
cli.add_command(inference, name="inferences")
cli.add_command(database, name="database")
cli.add_command(database, name="databases")
cli.add_command(voice, name="voice")
cli.add_command(voice, name="voice-agents")
cli.add_command(app, name="app")
cli.add_command(app, name="apps")
cli.add_command(trigger, name="trigger")
cli.add_command(trigger, name="triggers")
cli.add_command(config)


def main() -> None:
    try:
        cli(obj=DynamiqCtx())
    except HTTPError as exc:
        click.echo(f"❌ {exc}", err=True)
        raise SystemExit(1) from exc
