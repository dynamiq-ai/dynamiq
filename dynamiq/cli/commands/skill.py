import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.commands.context import with_api_and_settings
from dynamiq.cli.commands.workflow import echo_list, echo_response, pagination_options, read_json_arg, require_project
from dynamiq.cli.config import Settings

skill = click.Group(name="skill", help="Agent skills: create, upload, version, inspect")


@skill.command("list")
@click.option("--include-official", is_flag=True, help="Also list the platform's official skills.")
@pagination_options
@with_api_and_settings
def list_skills(*, api: ApiClient, settings: Settings, include_official: bool, page, page_size, fetch_all, compact):
    """List skills available in the current project."""
    params: dict = {"project_id": require_project(settings)}
    if include_official:
        params["include_official"] = "true"
    echo_list(api, "/v1/skills", params, page, page_size, fetch_all, compact)


@skill.command("get")
@click.argument("skill_id")
@with_api_and_settings
def get_skill(*, api: ApiClient, settings: Settings, skill_id: str):
    """Fetch one skill and its metadata."""
    echo_response(api.get(f"/v1/skills/{skill_id}"))


@skill.command("create")
@click.argument("payload")
@with_api_and_settings
def create_skill(*, api: ApiClient, settings: Settings, payload: str):
    """Create a skill from inline instructions. REQUIRED: `name`, `description`, `instructions`.

    `project_id` is filled in automatically. `instructions` is the whole SKILL.md body as a
    string - pass `@file.json` rather than trying to escape a long document on the command
    line. An agent reads `description` when deciding whether to open the skill at all, so write
    it as a trigger ("use when the user asks to ..."), not as a title.

    Use `skill upload` instead when the skill has scripts or other files: this route stores
    instructions only.
    """
    body = read_json_arg(payload)
    body.setdefault("project_id", require_project(settings))
    echo_response(api.post("/v1/skills", json=body))


@skill.command("upload")
@click.argument("zip_path", type=click.Path(exists=True, dir_okay=False))
@with_api_and_settings
def upload_skill(*, api: ApiClient, settings: Settings, zip_path: str):
    """Upload a skill as a ZIP - the way to ship one that has scripts alongside SKILL.md.

    `SKILL.md` must sit at the ROOT of the archive, not inside a wrapping folder, or the
    platform will not find it. Build it from inside the skill directory:

        cd my-skill && zip -r ../my-skill.zip .

    Max 1 MB.
    """
    with open(zip_path, "rb") as handle:
        echo_response(
            api.post(
                "/v1/skills/upload",
                data={"project_id": require_project(settings)},
                files={"file": (zip_path.rsplit("/", 1)[-1], handle)},
                retry=False,   # the handle is at EOF after the first attempt
            )
        )


@skill.command("import-github")
@click.argument("github_url")
@with_api_and_settings
def import_skill(*, api: ApiClient, settings: Settings, github_url: str):
    """Import a skill straight from a public GitHub URL."""
    echo_response(
        api.post(
            "/v1/skills/import/github",
            json={"github_url": github_url, "project_id": require_project(settings)},
        )
    )


@skill.command("versions")
@click.argument("skill_id")
@pagination_options
@with_api_and_settings
def list_versions(*, api: ApiClient, settings: Settings, skill_id: str, page, page_size, fetch_all, compact):
    """List a skill's versions, newest first."""
    echo_list(api, f"/v1/skills/{skill_id}/versions", None, page, page_size, fetch_all, compact)


@skill.command("instructions")
@click.argument("skill_id")
@click.argument("version_id")
@with_api_and_settings
def get_instructions(*, api: ApiClient, settings: Settings, skill_id: str, version_id: str):
    """Print a version's instructions - what the agent actually reads."""
    echo_response(api.get(f"/v1/skills/{skill_id}/versions/{version_id}/instructions"))


@skill.command("version-add")
@click.argument("skill_id")
@click.argument("payload")
@with_api_and_settings
def add_version(*, api: ApiClient, settings: Settings, skill_id: str, payload: str):
    """Publish a new version. REQUIRED: `description`, `instructions`.

    Skills are versioned rather than edited in place, so this is how an update ships. Agents
    already running keep the version they loaded.
    """
    echo_response(api.post(f"/v1/skills/{skill_id}/versions", json=read_json_arg(payload)))


@skill.command("version-upload")
@click.argument("skill_id")
@click.argument("zip_path", type=click.Path(exists=True, dir_okay=False))
@with_api_and_settings
def upload_version(*, api: ApiClient, settings: Settings, skill_id: str, zip_path: str):
    """Publish a new version from a ZIP, same layout rules as `skill upload`."""
    with open(zip_path, "rb") as handle:
        echo_response(
            api.post(
                f"/v1/skills/{skill_id}/versions/upload",
                files={"file": (zip_path.rsplit("/", 1)[-1], handle)},
                retry=False,   # the handle is at EOF after the first attempt
            )
        )


@skill.command("delete")
@click.argument("skill_id")
@click.confirmation_option(prompt="Delete this skill?")
@with_api_and_settings
def delete_skill(*, api: ApiClient, settings: Settings, skill_id: str):
    """Delete a skill and all of its versions."""
    echo_response(api.delete(f"/v1/skills/{skill_id}"))
