"""Unit tests for skills utility helpers."""

from dynamiq.skills.utils import extract_skill_content_slice


def test_extract_skill_content_slice_section_found():
    instructions = "# Intro\nline 1\n## Details\nline 2\nline 3\n# Outro\nline 4\n"

    sliced, section_used = extract_skill_content_slice(instructions, section="Details")

    assert sliced == "## Details\nline 2\nline 3"
    assert section_used == "Details"


def test_extract_skill_content_slice_section_not_found_returns_empty():
    instructions = "# Intro\nline 1\n## Details\nline 2\n"

    sliced, section_used = extract_skill_content_slice(instructions, section="Missing")

    assert sliced == ""
    assert section_used is None


def test_extract_skill_content_slice_line_range():
    instructions = "a\nb\nc\nd\n"

    sliced, section_used = extract_skill_content_slice(instructions, line_start=2, line_end=3)

    assert sliced == "b\nc"
    assert section_used is None
