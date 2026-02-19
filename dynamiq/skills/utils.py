import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import TYPE_CHECKING, Any

from dynamiq.skills.types import SkillRegistryError
from dynamiq.utils.logger import logger

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


def ingest_skills_into_sandbox(
    sandbox: "Sandbox",
    registry: "Dynamiq",
    skill_names: list[str] | None = None,
    sandbox_skills_base_path: str | None = None,
) -> list[str]:
    """Download skill archives from the Dynamiq registry and ingest into the sandbox.

    Downloads all skill zips in parallel (ThreadPoolExecutor, up to 8 workers),
    then uploads one zip per skill to the sandbox, then runs one batch unzip
    shell in the sandbox. Default base is sandbox.base_path + "/skills" (e.g. /home/user/skills).
    Parent directories for each skill are created by the sandbox on upload. Requires `unzip` in the sandbox.

    Args:
        sandbox: Sandbox with upload_file(file_name, content, destination_path) and run_command_shell.
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

    entries_to_ingest = [e for e in registry.skills if e.name in skills_to_ingest]
    if not entries_to_ingest:
        return ingested

    t_ingestion_start = time.perf_counter()

    logger.info("Ingestion: downloading %d skills in parallel", len(entries_to_ingest))
    download_results: dict[str, tuple[Any, bytes]] = {}  # name -> (entry, zip_bytes)
    max_workers = min(8, len(entries_to_ingest))
    t_download_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(registry.download_skill_archive, e.id, e.version_id): e for e in entries_to_ingest
        }
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                zip_bytes = future.result()
                download_results[entry.name] = (entry, zip_bytes)
            except Exception as e:
                logger.warning("Failed to download skill archive %s: %s", entry.name, e)
                raise SkillRegistryError(
                    f"Failed to download skill '{entry.name}': {e}",
                    details={"skill_id": entry.id, "version_id": entry.version_id},
                ) from e
    download_duration = time.perf_counter() - t_download_start
    total_bytes = sum(len(z) for _e, z in download_results.values())
    logger.info(
        "Ingestion: downloaded %d skills in parallel — total_size=%d bytes, duration=%.2fs",
        len(download_results),
        total_bytes,
        download_duration,
    )

    uploaded: list[tuple[str, str, str, int]] = []  # (skill_name, skill_dir, zip_path, file_count)
    for entry in entries_to_ingest:
        _e, zip_bytes = download_results[entry.name]
        logger.info("Ingestion: uploading skill '%s' to sandbox as zip (%d bytes)", entry.name, len(zip_bytes))
        t_upload_start = time.perf_counter()
        try:
            file_count, skill_dir, zip_path = _upload_zip_only(sandbox, zip_bytes, base, entry.name)
        except Exception as e:
            logger.warning("Failed to upload skill %s to sandbox: %s", entry.name, e)
            raise SkillRegistryError(
                f"Failed to upload skill '{entry.name}' to sandbox: {e}",
                details={"skill_name": entry.name},
            ) from e
        upload_duration = time.perf_counter() - t_upload_start
        ingested.append(entry.name)
        uploaded.append((entry.name, skill_dir, zip_path, file_count))
        logger.info(
            "Ingestion: uploaded zip for '%s' under %s — files=%d, duration=%.2fs",
            entry.name,
            skill_dir,
            file_count,
            upload_duration,
        )

    if uploaded:
        t_unzip_start = time.perf_counter()
        try:
            _batch_unzip_in_sandbox(sandbox, uploaded)
        except Exception as e:
            logger.warning("Batch unzip failed: %s", e)
            raise SkillRegistryError(
                "Failed to unzip skills in sandbox: " + str(e),
                details={"uploaded_skills": [u[0] for u in uploaded]},
            ) from e
        unzip_duration = time.perf_counter() - t_unzip_start
        logger.info(
            "Ingestion: batch unzip %d skills in sandbox — duration=%.2fs",
            len(uploaded),
            unzip_duration,
        )

    total_duration = time.perf_counter() - t_ingestion_start
    logger.info("Ingestion: completed in %.2fs total (%d skills)", total_duration, len(ingested))
    return ingested


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
            "Ingestion: skill '%s' — archive_size=%d bytes, files=%d, total_entries=%d, dirs_skipped=%d",
            skill_name,
            len(zip_bytes),
            file_count,
            len(all_infos),
            num_dirs,
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
    """
    import shlex

    if not uploaded:
        return
    # One shell: for each (skill_dir, zip_path), unzip into skill_dir and rm zip.
    parts = []
    for _name, skill_dir, zip_path, _count in uploaded:
        parts.append(
            f"unzip -o -q {shlex.quote(zip_path)} -d {shlex.quote(skill_dir)} && rm -f {shlex.quote(zip_path)}"
        )
    sandbox.run_command_shell(" && ".join(parts))


def _upload_zip_to_sandbox(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
) -> int:
    """Upload zip as one file to sandbox, then unzip there (1 round-trip per skill instead of N).

    Returns the number of files in the zip (directory entries and path-traversal names excluded).
    Requires `unzip` in the sandbox; falls back to per-file upload if unzip fails.
    """
    import shlex

    file_count, skill_dir, zip_path = _upload_zip_only(sandbox, zip_bytes, base_path, skill_name)
    t_start = time.perf_counter()
    try:
        unzip_cmd = (
            f"unzip -o -q {shlex.quote(zip_path)} -d {shlex.quote(skill_dir)} " f"&& rm -f {shlex.quote(zip_path)}"
        )
        sandbox.run_command_shell(unzip_cmd)
        duration = time.perf_counter() - t_start
        logger.info(
            "Ingestion: upload+unzip skill '%s' — 1 zip, %d files, duration=%.2fs",
            skill_name,
            file_count,
            duration,
        )
        return file_count
    except Exception as e:
        logger.warning(
            "Ingestion: upload+unzip failed for '%s' (%s), falling back to per-file upload",
            skill_name,
            e,
        )
        try:
            sandbox.run_command_shell(f"rm -f {shlex.quote(zip_path)}")
        except Exception as cleanup_err:
            logger.debug("Cleanup of leftover zip %s failed: %s", zip_path, cleanup_err)
        return _upload_zip_to_sandbox_per_file(sandbox, zip_bytes, base_path, skill_name, t_start)


def _upload_zip_to_sandbox_per_file(
    sandbox: "Sandbox",
    zip_bytes: bytes,
    base_path: str,
    skill_name: str,
    t_start: float | None = None,
) -> int:
    """Fallback: unzip in memory and upload each file (N round-trips). Used when upload+unzip in sandbox fails."""
    if t_start is None:
        t_start = time.perf_counter()
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
    duration = time.perf_counter() - t_start
    logger.info(
        "Ingestion: per-file upload skill '%s' — files=%d, duration=%.2fs",
        skill_name,
        count,
        duration,
    )
    return count
