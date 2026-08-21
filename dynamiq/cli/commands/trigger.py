import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_response, read_json_arg
from dynamiq.cli.config import Settings

trigger = click.Group(name="trigger", help="Manage app triggers (schedule/webhook)")


@trigger.command("list")
@click.argument("app_id")
@with_api_and_settings
def list_triggers(*, api: ApiClient, settings: Settings, app_id: str):
    echo_response(api.get(f"/v1/apps/{app_id}/triggers"))


@trigger.command("create")
@click.argument("app_id")
@click.argument("payload")
@with_api_and_settings
def create_trigger(*, api: ApiClient, settings: Settings, app_id: str, payload: str):
    """Create a trigger. Copy the config shape from an app that already has one.
    New triggers are inactive until `dynamiq trigger activate`."""
    echo_response(api.post(f"/v1/apps/{app_id}/triggers", json=read_json_arg(payload)))


@trigger.command("activate")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def activate_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/activate", json={}))


@trigger.command("deactivate")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def deactivate_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/deactivate", json={}))


@trigger.command("run")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def run_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Fire the trigger once, immediately - use to verify a new trigger end to end."""
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/run", json={}))


@trigger.command("delete")
@click.argument("app_id")
@click.argument("trigger_id")
@click.confirmation_option(prompt="Delete this trigger?")
@with_api_and_settings
def delete_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    echo_response(api.delete(f"/v1/apps/{app_id}/triggers/{trigger_id}"))
