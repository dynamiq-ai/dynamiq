import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import TYPE_CHECKING

from dynamiq.skills.types import SkillRegistryError
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox
    from dynamiq.skills.registries.dynamiq import Dynamiq, DynamiqSkillEntry


def extract_skill_content_slice(
    instructions: str,
    section: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> tuple[str, str | None]:
    """Extract a slice of skill instructions by section header or line range.

    Args:
        instructions: Full skill instructions (e.g. from SkillInstructions.instructions).
        section: Markdown header to extract (e.g. "Welcome messages"); first # or ## match.
        line_start: 1-based start line (inclusive).
        line_end: 1-based end line (inclusive).

    Returns:
        Tuple of (sliced_instructions, section_used). section_used is the section name if
        section was requested and found, else None.
    """
    lines = instructions.splitlines()
    section_used: str | None = None

    if section:
        section_lower = section.strip().lower()
        start_i: int | None = None
        end_i = len(lines)
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("#"):
                header_level = len(s) - len(s.lstrip("#"))
                header_text = s.lstrip("#").strip().lower()
                if header_text == section_lower:
                    start_i = i
                    for j in range(i + 1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line.startswith("#"):
                            next_level = len(next_line) - len(next_line.lstrip("#"))
                            if next_level <= header_level:
                                end_i = j
                                break
                    section_used = section
                    break
        if start_i is not None:
            instructions = "\n".join(lines[start_i:end_i])
        else:
            instructions = ""
            section_used = None
    elif line_start is not None or line_end is not None:
        start = max(0, (line_start or 1) - 1)
        end = line_end if line_end is not None else len(lines)
        end = min(end, len(lines))
        instructions = "\n".join(lines[start:end])

    return instructions, section_used


def _ingest_one_skill(
    sandbox: "Sandbox",
    registry: "Dynamiq",
    base: str,
    entry: "DynamiqSkillEntry",
) -> str:
    """Download one skill archive and upload it to the sandbox. Returns skill name or raises."""
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
    return entry.name


def ingest_skills_into_sandbox(
    sandbox: "Sandbox",
    registry: "Dynamiq",
    skill_names: list[str] | None = None,
    sandbox_skills_base_path: str | None = None,
    max_workers: int | None = None,
) -> list[str]:
    """Download skill archives from the Dynamiq registry, unzip, and upload into the sandbox.

    Skills are ingested in parallel (download + upload per skill). For each skill
    (all configured skills, or only those in skill_names), calls
    registry.download_skill_archive(skill_id, version_id), unzips the zip, and
    uploads every file to sandbox at {base}/{skill_name}/{relative_path}.
    Default base is sandbox.base_path + "/skills" (e.g. /home/user/skills).

    Args:
        sandbox: Sandbox with upload_file(file_name, content, destination_path).
        registry: Dynamiq registry (must have download_skill_archive).
        skill_names: If set, only ingest these skill names; otherwise all registry skills.
        sandbox_skills_base_path: Base path in sandbox for skills (default: {sandbox.base_path}/skills).
        max_workers: Max concurrent skills to ingest (default: number of skills, capped at 8).

    Returns:
        List of skill names that were ingested (order may differ from registry).

    Raises:
        SkillRegistryError: If download or upload fails for any skill.
    """
    base = (sandbox_skills_base_path or f"{sandbox.base_path.rstrip('/')}/skills").rstrip("/")
    skills_to_ingest = skill_names if skill_names is not None else [e.name for e in registry.skills]
    entries = [e for e in registry.skills if e.name in skills_to_ingest]
    if not entries:
        return []

    workers = max_workers
    if workers is None:
        workers = min(len(entries), 8)

    ingested: list[str] = []
    first_exception: BaseException | None = None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_entry = {
            executor.submit(_ingest_one_skill, sandbox, registry, base, entry): entry for entry in entries
        }
        for future in as_completed(future_to_entry):
            try:
                name = future.result()
                ingested.append(name)
                logger.info("Ingested skill '%s' into sandbox under %s/%s", name, base, name)
            except BaseException as e:
                if first_exception is None:
                    first_exception = e

    if first_exception is not None:
        raise first_exception

    return sorted(ingested)


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
