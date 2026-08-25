import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_response, read_json_arg
from dynamiq.cli.config import Settings

trigger = click.Group(name="trigger", help="Manage app triggers (pipedream events / cron schedules)")

# The registry accepts exactly these two providers - there is no "webhook" provider.
PROVIDERS = ("pipedream", "schedule")


@trigger.command("list")
@click.argument("app_id")
@with_api_and_settings
def list_triggers(*, api: ApiClient, settings: Settings, app_id: str):
    """List an app's triggers. Copy a working `config` shape from here before creating one."""
    echo_response(api.get(f"/v1/apps/{app_id}/triggers"))


@trigger.command("get")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def get_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Fetch one trigger (status: draft | active | inactive | expired)."""
    echo_response(api.get(f"/v1/apps/{app_id}/triggers/{trigger_id}"))


@trigger.command("create")
@click.argument("app_id")
@click.argument("payload")
@with_api_and_settings
def create_trigger(*, api: ApiClient, settings: Settings, app_id: str, payload: str):
    """Create a trigger. REQUIRED body fields: `name`, `config`; `provider` defaults to
    "pipedream" (the only other value is "schedule" - there is NO webhook provider).

      provider "pipedream" -> config REQUIRES `trigger_id`; optional `configured_props`.
                              Never send `deployed_trigger_id`; the backend sets it.
      provider "schedule"  -> config REQUIRES `timezone` (IANA, not "local") and EXACTLY
                              ONE of `schedule` (5-field cron, no seconds) or `run_at`
                              (future ISO-8601). Optional `expires_at` (future, and not
                              allowed together with `run_at`).

    Optional `input_transformer`: {"path": ..., "selector": {...}}.
    New triggers are created as `draft` - activate them to arm them.
    """
    body = read_json_arg(payload)
    provider = body.get("provider", "pipedream")
    if provider not in PROVIDERS:
        raise click.ClickException(f"Unknown provider {provider!r}. Allowed: {', '.join(PROVIDERS)}.")
    echo_response(api.post(f"/v1/apps/{app_id}/triggers", json=body))


@trigger.command("create-schedule")
@click.argument("app_id")
@click.option("--name", required=True, help="Trigger name.")
@click.option("--cron", default=None, help="5-field cron expression, e.g. '0 9 * * 1' (no seconds field).")
@click.option("--run-at", default=None, help="One-off ISO-8601 time in the future, e.g. 2026-09-01T09:00:00Z.")
@click.option("--timezone", "tz", default="UTC", show_default=True, help="IANA timezone, e.g. Europe/Kyiv.")
@click.option("--expires-at", default=None, help="Future ISO-8601 expiry (cron schedules only).")
@with_api_and_settings
def create_schedule_trigger(
    *,
    api: ApiClient,
    settings: Settings,
    app_id: str,
    name: str,
    cron: str | None,
    run_at: str | None,
    tz: str,
    expires_at: str | None,
):
    """Create a schedule trigger without hand-writing the payload.

    Pass EXACTLY ONE of --cron or --run-at (the API rejects both or neither).
    """
    if bool(cron) == bool(run_at):
        raise click.ClickException("Pass exactly one of --cron or --run-at.")
    if expires_at and run_at:
        raise click.ClickException("One-off triggers (--run-at) cannot have --expires-at.")

    config: dict = {"timezone": tz}
    if cron:
        config["schedule"] = cron
    else:
        config["run_at"] = run_at
    if expires_at:
        config["expires_at"] = expires_at

    body = {"name": name, "provider": "schedule", "config": config}
    echo_response(api.post(f"/v1/apps/{app_id}/triggers", json=body))


@trigger.command("update")
@click.argument("app_id")
@click.argument("trigger_id")
@click.argument("payload")
@with_api_and_settings
def update_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str, payload: str):
    """Update a trigger's `config`/`input_transformer`. The provider CANNOT be changed."""
    echo_response(api.put(f"/v1/apps/{app_id}/triggers/{trigger_id}", json=read_json_arg(payload)))


@trigger.command("activate")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def activate_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Arm a trigger (no body). Expired triggers cannot be activated."""
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/activate"))


@trigger.command("deactivate")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def deactivate_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Disarm a trigger (no body). Expired triggers cannot be deactivated."""
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/deactivate"))


@trigger.command("run")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def run_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Fire a trigger once, immediately (no body).

    Only works for provider "schedule" AND status "active" - the API rejects manual runs
    of pipedream triggers and of inactive/draft ones.
    """
    echo_response(api.post(f"/v1/apps/{app_id}/triggers/{trigger_id}/run"))


@trigger.command("events")
@click.argument("app_id")
@click.argument("trigger_id")
@with_api_and_settings
def list_trigger_events(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """List the events a trigger has fired - use this to verify it actually ran."""
    echo_response(api.get(f"/v1/apps/{app_id}/triggers/{trigger_id}/events"))


@trigger.command("delete")
@click.argument("app_id")
@click.argument("trigger_id")
@click.confirmation_option(prompt="Delete this trigger?")
@with_api_and_settings
def delete_trigger(*, api: ApiClient, settings: Settings, app_id: str, trigger_id: str):
    """Delete a trigger permanently (no body)."""
    echo_response(api.delete(f"/v1/apps/{app_id}/triggers/{trigger_id}"))
