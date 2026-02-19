import shlex
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import TYPE_CHECKING, Any

from dynamiq.skills.types import SkillRegistryError
from dynamiq.utils.logger import logger

MAX_INGEST_WORKERS = 8

if TYPE_CHECKING:
    from dynamiq.sandboxes.base import Sandbox
    from dynamiq.skills.registries.dynamiq import Dynamiq


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


def _download_and_upload_one(
    registry: "Dynamiq",
    sandbox: "Sandbox",
    base: str,
    entry: Any,
) -> tuple[str, str, str, int]:
    """Download one skill archive and upload its zip to the sandbox.
    Returns (skill_name, skill_dir, zip_path, file_count)."""
    zip_bytes = registry.download_skill_archive(entry.id, entry.version_id)
    logger.debug(f"Ingestion: downloaded {entry.name} ({len(zip_bytes)} bytes), uploading to sandbox")
    file_count, skill_dir, zip_path = _upload_zip_only(sandbox, zip_bytes, base, entry.name)
    return (entry.name, skill_dir, zip_path, file_count)


def ingest_skills_into_sandbox(
    sandbox: "Sandbox",
    registry: "Dynamiq",
    skill_names: list[str] | None = None,
    sandbox_skills_base_path: str | None = None,
) -> list[str]:
    """Download skill archives from the Dynamiq registry and ingest into the sandbox.

    Each worker thread downloads one skill and uploads its zip to the sandbox (no need to wait
    for all downloads before starting uploads). Then a single batch unzip runs in the sandbox.
    Default base is sandbox.base_path + "/skills" (e.g. /home/user/skills).
    Requires `unzip` in the sandbox.

    Args:
        sandbox: Sandbox with upload_file(file_name, content, destination_path) and run_command_shell.
        registry: Dynamiq registry (must have download_skill_archive).
        skill_names: If set, only ingest these skill names; otherwise all registry skills.
        sandbox_skills_base_path: Base path in sandbox for skills (default: {sandbox.base_path}/skills).

    Returns:
        List of skill names that were ingested.

    Raises:
        SkillRegistryError: If download, upload, or unzip fails.
    """
    base = (sandbox_skills_base_path or f"{sandbox.base_path.rstrip('/')}/skills").rstrip("/")
    skills_to_ingest = skill_names if skill_names is not None else [e.name for e in registry.skills]
    entries_to_ingest = [e for e in registry.skills if e.name in skills_to_ingest]
    if not entries_to_ingest:
        return []

    workers = min(MAX_INGEST_WORKERS, len(entries_to_ingest))
    logger.debug(f"Ingestion: download+upload {len(entries_to_ingest)} skills with {workers} workers")
    uploaded: list[tuple[str, str, str, int]] = []
    ingested: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_entry = {
            executor.submit(_download_and_upload_one, registry, sandbox, base, e): e for e in entries_to_ingest
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                uploaded.append(future.result())
                ingested.append(entry.name)
            except Exception as e:
                logger.warning(f"Failed to ingest skill {entry.name}: {e}")
                raise SkillRegistryError(
                    f"Failed to ingest skill '{entry.name}': {e}",
                    details={"skill_id": entry.id, "version_id": entry.version_id, "skill_name": entry.name},
                ) from e

    if uploaded:
        try:
            _batch_unzip_in_sandbox(sandbox, uploaded)
        except Exception as e:
            logger.warning(f"Batch unzip failed: {e}")
            raise SkillRegistryError(
                "Failed to unzip skills in sandbox: " + str(e),
                details={"uploaded_skills": [u[0] for u in uploaded]},
            ) from e

    # Return in same order as entries_to_ingest for deterministic behavior.
    order = {name: i for i, e in enumerate(entries_to_ingest) for name in [e.name]}
    return sorted(ingested, key=order.__getitem__)


def _upload_zip_only(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
) -> tuple[int, str, str]:
    """Upload skill zip to sandbox as a single file; do not unzip.

    Sandbox must create parent dirs for destination_path if needed.
    Returns (file_count, skill_dir, zip_path). Caller should run unzip in sandbox (e.g. via _batch_unzip_in_sandbox).
    """
    prefix = f"{base_path}/{skill_name}/"
    skill_dir = prefix.rstrip("/")
    zip_path = f"{prefix}skill.zip"

    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf_read:
        all_infos = zf_read.infolist()
        to_include = [
            info
            for info in all_infos
            if not info.is_dir() and ".." not in info.filename and not info.filename.startswith("/")
        ]
        file_count = len(to_include)
        num_dirs = sum(1 for info in all_infos if info.is_dir())
        logger.debug(
            f"Ingestion: skill {skill_name!r} — archive_size={len(zip_bytes)} bytes, files={file_count}, "
            f"total_entries={len(all_infos)}, dirs_skipped={num_dirs}"
        )
        buf_out = BytesIO()
        with zipfile.ZipFile(buf_out, "w", zipfile.ZIP_DEFLATED) as zf_write:
            for info in to_include:
                zf_write.writestr(info.filename, zf_read.read(info))
        zip_to_upload = buf_out.getvalue()

    sandbox.upload_file("skill.zip", zip_to_upload, destination_path=zip_path)
    return file_count, skill_dir, zip_path


def _batch_unzip_in_sandbox(
    sandbox: "Sandbox",
    uploaded: list[tuple[str, str, str, int]],
) -> None:
    """Run a single shell in the sandbox to unzip all uploaded skill zips and remove the zip files.

    uploaded: list of (skill_name, skill_dir, zip_path, file_count).
    Requires `unzip` in the sandbox.

    Raises:
        SkillRegistryError: If the unzip shell command fails (e.g. exit_code != 0).
    """

    if not uploaded:
        return
    # One shell: for each (skill_dir, zip_path), unzip into skill_dir and rm zip.
    parts = []
    for _name, skill_dir, zip_path, _count in uploaded:
        parts.append(
            f"unzip -o -q {shlex.quote(zip_path)} -d {shlex.quote(skill_dir)} && rm -f {shlex.quote(zip_path)}"
        )
    result = sandbox.run_command_shell(" && ".join(parts))
    exit_code = getattr(result, "exit_code", None)
    if isinstance(exit_code, int) and exit_code != 0:
        stderr = getattr(result, "stderr", "") or ""
        stdout = getattr(result, "stdout", "") or ""
        msg = f"Batch unzip failed (exit_code={exit_code})"
        if stderr:
            msg += f". stderr: {stderr.strip()!r}"
        if stdout and not stderr:
            msg += f". stdout: {stdout.strip()!r}"
        raise SkillRegistryError(msg, details={"uploaded_skills": [u[0] for u in uploaded]})


def _upload_zip_to_sandbox(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
) -> int:
    """Upload zip as one file to sandbox, then unzip there (1 round-trip per skill instead of N).

    Returns the number of files in the zip (directory entries and path-traversal names excluded).
    Requires `unzip` in the sandbox; falls back to per-file upload if unzip returns non-zero exit code.
    """
    import shlex

    file_count, skill_dir, zip_path = _upload_zip_only(sandbox, zip_bytes, base_path, skill_name)
    unzip_cmd = f"unzip -o -q {shlex.quote(zip_path)} -d {shlex.quote(skill_dir)} && rm -f {shlex.quote(zip_path)}"
    result = sandbox.run_command_shell(unzip_cmd)
    exit_code = getattr(result, "exit_code", None)
    if not (isinstance(exit_code, int) and exit_code != 0):
        logger.debug(f"Ingestion: upload+unzip skill {skill_name!r} — 1 zip, {file_count} files")
        return file_count
    logger.warning(
        f"Ingestion: upload+unzip failed for {skill_name!r} (exit_code={exit_code}), falling back to per-file upload"
    )
    try:
        sandbox.run_command_shell(f"rm -f {shlex.quote(zip_path)}")
    except Exception as cleanup_err:
        logger.debug(f"Cleanup of leftover zip {zip_path} failed: {cleanup_err}")
    return _upload_zip_to_sandbox_per_file(sandbox, zip_bytes, base_path, skill_name)


def _upload_zip_to_sandbox_per_file(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
) -> int:
    """Fallback: unzip in memory and upload each file (N round-trips). Used when upload+unzip in sandbox fails."""
    prefix = f"{base_path}/{skill_name}/"
    count = 0
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
            count += 1
    logger.debug(f"Ingestion: per-file upload skill {skill_name!r} — {count} files")
    return count
