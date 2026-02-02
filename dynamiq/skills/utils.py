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
