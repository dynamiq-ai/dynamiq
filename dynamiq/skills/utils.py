from pathlib import Path
from typing import Any

from dynamiq.utils.logger import logger


def sync_local_skills_to_filestore(
    file_store: Any,
    local_dir: str | Path = ".skills",
    prefix: str = ".skills/",
) -> None:
    """Sync a local skills directory into FileStore (e.g. .skills/ -> .skills/ prefix).

    Called during skills init for Local backend so skills are available without
    manual upload. Normalizes SKILL.MD -> SKILL.md so the loader discovers skills.
    """
    path = Path(local_dir).resolve()
    if not path.exists() or not path.is_dir():
        logger.debug("No local skills dir at %s; skipping sync", path)
        return
    prefix_norm = prefix.rstrip("/") + "/"
    for skill_dir in path.iterdir():
        if not skill_dir.is_dir():
            continue
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(path)
                store_path = prefix_norm + rel.as_posix()
                if store_path.endswith("/SKILL.MD"):
                    store_path = store_path.replace("/SKILL.MD", "/SKILL.md")
                file_store.store(store_path, file_path.read_bytes())
                logger.debug("Synced %s -> %s", file_path, store_path)
    logger.info("Synced local skills from %s into FileStore (prefix %s)", path, prefix_norm)


def extract_skill_content_slice(
    instructions: str,
    section: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
) -> tuple[str, str | None]:
    """Extract a slice of skill instructions by section header or line range.

    Args:
        instructions: Full skill body (e.g. from Skill.instructions).
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
            section_used = None
    elif line_start is not None or line_end is not None:
        start = max(0, (line_start or 1) - 1)
        end = line_end if line_end is not None else len(lines)
        end = min(end, len(lines))
        instructions = "\n".join(lines[start:end])

    return instructions, section_used
