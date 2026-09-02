"""Rendering helpers for notes.

Both the agent's system-prompt injection and the note tools' results render through this
module. If the injected index and a tool's echoed index differed by so much as a bullet
character, the model would read them as two different lists.
"""

from dynamiq.memory.notes.schemas import Note, NoteIndexEntry

INDEX_SEPARATOR = " — "
INDEX_TRUNCATED_LINE = "- (more notes not shown — use a narrower set of notes or delete stale ones)"
EMPTY_INDEX = "(no notes yet)"
NOTE_SEPARATOR = "\n\n---\n\n"


def render_notes_index(
    entries: list[NoteIndexEntry],
    *,
    truncated: bool = False,
    empty: str = EMPTY_INDEX,
) -> str:
    """Render the `- title — description` block shown to the agent on every run."""
    if not entries:
        return empty
    lines = [f"- {entry.title}{INDEX_SEPARATOR}{entry.description}" for entry in entries]
    if truncated:
        lines.append(INDEX_TRUNCATED_LINE)
    return "\n".join(lines)


def render_notes(notes: list[Note]) -> str:
    """Render full note bodies, each under a `## title` heading, for `read_note`."""
    return NOTE_SEPARATOR.join(f"## {note.title}\n\n{note.content}" for note in notes)
