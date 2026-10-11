import json
from typing import Any

import click
import requests

from dynamiq.cli.client import ApiClient, ok
from dynamiq.cli.commands.access import access_failure, mask_token, token_rejected
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.config import Settings

NOT_SET = "<not set>"


def _body(response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else {}


def _check(api: ApiClient, settings: Settings, kind: str, path: str, resource_id: str) -> dict[str, Any]:
    response = api.get(path)
    if ok(response):
        data = _body(response)
        return {"accessible": True, "status": response.status_code, "name": data.get("name"), "data": data}
    return {
        "accessible": False,
        "status": response.status_code,
        "message": access_failure(settings, kind, resource_id, response.status_code),
    }


def build_report(api: ApiClient, settings: Settings) -> dict[str, Any]:
    """Ask the API who the token is and whether the configured org and project are reachable."""
    problems: list[str] = []
    report: dict[str, Any] = {
        "host": {"value": settings.api_host, "source": settings.source_of("api_host")},
        "token": {"value": mask_token(settings.api_key), "source": settings.source_of("api_key")},
        "user": None,
        "org": {"id": settings.org_id, "source": settings.source_of("org_id")},
        "project": {"id": settings.project_id, "source": settings.source_of("project_id")},
        "problems": problems,
    }

    me = api.get("/v1/me")
    report["token"]["status"] = me.status_code
    if me.status_code == 401:
        report["token"]["accepted"] = False
        problems.append(token_rejected(settings))
        report["ok"] = False
        return report

    report["token"]["accepted"] = True
    if ok(me):
        user = _body(me)
        report["user"] = {key: user.get(key) for key in ("id", "email", "first_name", "last_name")}
    elif me.status_code == 403:
        # /v1/me only answers for user principals; a service-account token is valid but has no
        # user behind it. That is not a misconfiguration, so it is noted, not reported.
        report["user_note"] = "token is not tied to a user (403 from /v1/me), e.g. a service-account token"
    else:
        problems.append(f"Could not read the token's user from /v1/me: HTTP {me.status_code}.")

    if settings.org_id:
        org = _check(api, settings, "Organization", f"/v1/orgs/{settings.org_id}", settings.org_id)
        report["org"].update({key: value for key, value in org.items() if key != "data"})
        if not org["accessible"]:
            problems.append(org["message"])

    if settings.project_id:
        project = _check(api, settings, "Project", f"/v1/projects/{settings.project_id}", settings.project_id)
        report["project"].update({key: value for key, value in project.items() if key != "data"})
        if not project["accessible"]:
            problems.append(project["message"])
        else:
            owner = project["data"].get("org_id")
            report["project"]["org_id"] = owner
            if owner and settings.org_id and owner != settings.org_id:
                problems.append(
                    f"Project {settings.project_id} belongs to organization {owner}, "
                    f"not the configured organization {settings.org_id}."
                )

    report["ok"] = not problems
    return report


def _user_line(report: dict[str, Any]) -> str:
    user = report["user"]
    if user:
        name = " ".join(part for part in (user.get("first_name"), user.get("last_name")) if part)
        identity = f"{name} <{user.get('email')}>" if name else str(user.get("email"))
        return f"{identity} (id {user.get('id')})"
    if report.get("user_note"):
        return report["user_note"]
    return "unknown"


def _resource_line(entry: dict[str, Any]) -> str:
    if not entry["id"]:
        return NOT_SET
    label = f"{entry['name']} [{entry['id']}]" if entry.get("name") else entry["id"]
    line = f"{label} ({entry['source']})"
    if entry.get("accessible") is False:
        line += f" - NOT ACCESSIBLE (HTTP {entry['status']})"
    elif "accessible" not in entry:
        line += " - not checked"
    return line


def render_text(report: dict[str, Any]) -> None:
    token = report["token"]
    token_line = f"{token['value']} ({token['source']})"
    if token.get("accepted") is False:
        token_line += " - REJECTED (401)"

    click.echo(f"User:     {_user_line(report)}")
    click.echo(f"Token:    {token_line}")
    click.echo(f"Host:     {report['host']['value']} ({report['host']['source']})")
    click.echo(f"Org:      {_resource_line(report['org'])}")
    click.echo(f"Project:  {_resource_line(report['project'])}")
    for problem in report["problems"]:
        click.echo(f"❌ {problem}", err=True)


@click.command("whoami")
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
@with_api_and_settings
def whoami(*, api: ApiClient, settings: Settings, as_json: bool):
    """Show who the token belongs to, and whether the configured org and project are reachable.

    Every value is followed by where it came from: an env var, the config or credentials
    file, or the default. Exits 1 when the token is rejected or the org or project cannot
    be read with it.
    """
    if not settings.api_key:
        raise click.ClickException("No API token configured. Set DYNAMIQ_API_TOKEN or run `dynamiq config`.")
    try:
        report = build_report(api, settings)
    except requests.RequestException as exc:
        raise click.ClickException(f"Could not reach {settings.api_host}: {exc}") from exc

    if as_json:
        click.echo(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        render_text(report)
    if not report["ok"]:
        click.get_current_context().exit(1)
