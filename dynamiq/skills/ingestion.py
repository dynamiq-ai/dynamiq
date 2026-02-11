import zipfile
from io import BytesIO
from typing import TYPE_CHECKING

from dynamiq.skills.registries.dynamiq import Dynamiq
from dynamiq.skills.types import SkillRegistryError
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox


def ingest_skills_into_sandbox(
    sandbox: "Sandbox",
    registry: Dynamiq,
    skill_names: list[str] | None = None,
    sandbox_skills_base_path: str | None = None,
) -> list[str]:
    """Download skill archives from the Dynamiq registry, unzip, and upload into the sandbox.

    For each skill (all configured skills, or only those in skill_names), calls
    registry.download_skill_archive(skill_id, version_id), unzips the zip, and
    uploads every file to sandbox at {base}/{skill_name}/{relative_path}.
    Default base is sandbox.base_path + "/skills" (e.g. /home/user/skills).

    Args:
        sandbox: Sandbox with upload_file(file_name, content, destination_path).
        registry: Dynamiq registry (must have download_skill_archive).
        skill_names: If set, only ingest these skill names; otherwise all registry skills.
        sandbox_skills_base_path: Base path in sandbox for skills (default: {sandbox.base_path}/skills).

    Returns:
        List of skill names that were ingested.

    Raises:
        SkillRegistryError: If download or registry fails.
    """
    base = (sandbox_skills_base_path or f"{sandbox.base_path.rstrip('/')}/skills").rstrip("/")
    skills_to_ingest = skill_names if skill_names is not None else [e.name for e in registry.skills]
    ingested: list[str] = []

    for entry in registry.skills:
        if entry.name not in skills_to_ingest:
            continue
        try:
            zip_bytes = registry.download_skill_archive(entry.id, entry.version_id)
        except Exception as e:
            logger.warning("Failed to download skill archive %s: %s", entry.name, e)
            raise SkillRegistryError(
                f"Failed to download skill '{entry.name}': {e}",
                details={"skill_id": entry.id, "version_id": entry.version_id},
            ) from e

        try:
            _upload_zip_to_sandbox(sandbox, zip_bytes, base, entry.name)
        except Exception as e:
            logger.warning("Failed to upload skill %s to sandbox: %s", entry.name, e)
            raise SkillRegistryError(
                f"Failed to upload skill '{entry.name}' to sandbox: {e}",
                details={"skill_name": entry.name},
            ) from e

        ingested.append(entry.name)
        logger.info("Ingested skill '%s' into sandbox under %s/%s", entry.name, base, entry.name)

    return ingested


def _upload_zip_to_sandbox(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
) -> None:
    """Unzip zip_bytes and upload each file to sandbox under base_path/skill_name/."""
    import shlex

    prefix = f"{base_path}/{skill_name}/"
    skill_dir = prefix.rstrip("/")
    try:
        sandbox.run_command_shell(f"mkdir -p {shlex.quote(skill_dir)}")
    except Exception as e:
        logger.warning("mkdir -p for skill dir %s failed: %s", skill_dir, e)
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if ".." in name or name.startswith("/"):
                continue
            relative = name.lstrip("/")
            destination = f"{prefix}{relative}"
            content = zf.read(info)
            sandbox.upload_file(relative.split("/")[-1], content, destination_path=destination)
