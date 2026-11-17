"""Building model-specific prompts on top of the base ReAct prompts."""

import re
from collections.abc import Iterator

from dynamiq.utils.logger import logger

# Rules the parser and the agent loop rely on: one action per turn and never writing the Observation, and, in the
# XML format, how to escape JSON. The base DEFAULT prompt has no JSON section.
DEFAULT_SECTIONS = ("## Single Action Per Turn",)
XML_SECTIONS = ("## Single Action Per Turn", "## JSON Formatting Requirements")

# A line that opens or closes fenced code: up to three spaces, then three or more backticks or tildes.
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
# A level 1 or 2 heading, which ends the section before it. A "### " subsection does not.
_SECTION_HEADING = re.compile(r"#{1,2}(?:[ \t]|$)")


def with_base_sections(prompt: str, base_prompt: str, headings: tuple[str, ...]) -> str:
    """Return ``prompt`` with each named section replaced by the same section of ``base_prompt``.

    A model-specific prompt is a full copy of a base prompt, so taking these sections from the base lets a fix
    there reach every model. A section runs from its heading line to the next level 1 or 2 heading, and lines in
    fenced code are not headings. A heading must be a line of its own exactly once in each prompt: otherwise, as
    when the base renames it, the override keeps its own section and a warning is logged. This never raises, as
    the overrides are built when the agents are imported.
    """
    for heading in headings:
        base_starts, starts = _heading_offsets(base_prompt, heading), _heading_offsets(prompt, heading)
        if len(base_starts) != 1 or len(starts) != 1:
            logger.warning(
                f"Not taking the {heading!r} section from the base prompt: the heading is a line of its own, outside "
                f"fenced code, {len(base_starts)} times in the base prompt and {len(starts)} times in the "
                "model-specific one, where once in each is needed. The model-specific prompt keeps its own section."
            )
            continue
        base_section = base_prompt[base_starts[0] : _section_end(base_prompt, base_starts[0])]
        prompt = prompt[: starts[0]] + base_section + prompt[_section_end(prompt, starts[0]) :]
    return prompt


def _lines_outside_fences(prompt: str) -> Iterator[tuple[int, str]]:
    """Yield the offset of each line of ``prompt`` outside fenced code, and the line without its line break."""
    offset, fence = 0, ""
    for line in prompt.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        marker = _FENCE.match(text)
        if fence:
            # Only a bare fence of the same character, at least as long as the opening one, closes it.
            if marker and marker.group(1).startswith(fence) and not text[marker.end() :].strip():
                fence = ""
        elif marker:
            fence = marker.group(1)
        else:
            yield offset, text
        offset += len(line)


def _heading_offsets(prompt: str, heading: str) -> list[int]:
    return [offset for offset, text in _lines_outside_fences(prompt) if text.rstrip() == heading]


def _section_end(prompt: str, start: int) -> int:
    """The offset of the first level 1 or 2 heading after ``start``, or the end of ``prompt``."""
    for offset, text in _lines_outside_fences(prompt):
        if offset > start and _SECTION_HEADING.match(text):
            return offset
    return len(prompt)
