from datetime import UTC, datetime

from dynamiq.memory.notes import Note, NoteIndexEntry, render_notes, render_notes_index
from dynamiq.memory.notes.formatting import EMPTY_INDEX, INDEX_TRUNCATED_LINE

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def _entry(title: str, description: str) -> NoteIndexEntry:
    return NoteIndexEntry(title=title, description=description, updated_at=NOW)


def _note(title: str, content: str) -> Note:
    return Note(user_id="u1", title=title, description="d", content=content, created_at=NOW, updated_at=NOW)


def test_render_index_produces_one_pointer_line_per_note():
    rendered = render_notes_index([_entry("Deploy runbook", "steps to ship"), _entry("API conventions", "envelope")])

    assert rendered == "- Deploy runbook — steps to ship\n- API conventions — envelope"


def test_render_index_when_empty():
    assert render_notes_index([]) == EMPTY_INDEX
    assert render_notes_index([], empty="(none)") == "(none)"


def test_render_index_flags_truncation():
    rendered = render_notes_index([_entry("a", "b")], truncated=True)

    assert rendered.endswith(INDEX_TRUNCATED_LINE)


def test_render_notes_heads_each_body_with_its_title():
    rendered = render_notes([_note("Deploy runbook", "1. merge"), _note("API conventions", "envelope")])

    assert rendered == "## Deploy runbook\n\n1. merge\n\n---\n\n## API conventions\n\nenvelope"


def test_render_notes_for_a_single_note_has_no_separator():
    assert render_notes([_note("Solo", "body")]) == "## Solo\n\nbody"


def test_render_notes_when_empty():
    assert render_notes([]) == ""
