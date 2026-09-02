import pytest

from dynamiq.memory.notes import (
    MAX_CONTENT_LEN,
    MAX_DESCRIPTION_LEN,
    MAX_TITLE_LEN,
    NoteDeleteStatus,
    NoteIndexEntry,
    NoteValidationError,
    NoteWriteOutcome,
)
from dynamiq.memory.notes.backends import InMemoryNotesBackend


def test_write_creates_a_note(backend, user_id):
    note, outcome = backend.write(user_id=user_id, title="Title", description="desc", content="body")

    assert outcome is NoteWriteOutcome.CREATED
    assert (note.user_id, note.title, note.description, note.content) == (user_id, "Title", "desc", "body")
    assert note.created_at == note.updated_at


def test_rewriting_a_title_updates_in_place_and_preserves_created_at(backend, user_id):
    first, _ = backend.write(user_id=user_id, title="Title", description="old", content="old body")
    second, outcome = backend.write(user_id=user_id, title="Title", description="new", content="new body")

    assert outcome is NoteWriteOutcome.UPDATED
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert (second.description, second.content) == ("new", "new body")
    assert len(backend.list_all(user_id=user_id)) == 1


def test_write_strips_surrounding_whitespace(backend, user_id):
    note, _ = backend.write(user_id=user_id, title="  Title  ", description="  desc  ", content="body")

    assert (note.title, note.description) == ("Title", "desc")


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", ""),
        ("title", "   "),
        ("title", "two\nlines"),
        ("title", "x" * (MAX_TITLE_LEN + 1)),
        ("description", ""),
        ("description", "two\nlines"),
        ("description", "x" * (MAX_DESCRIPTION_LEN + 1)),
        ("content", ""),
        ("content", "   "),
        ("content", "x" * (MAX_CONTENT_LEN + 1)),
    ],
)
def test_write_rejects_invalid_fields(backend, user_id, field, value):
    kwargs = {"title": "Title", "description": "desc", "content": "body"} | {field: value}

    with pytest.raises(NoteValidationError):
        backend.write(user_id=user_id, **kwargs)


def test_write_requires_a_user_id(backend):
    with pytest.raises(NoteValidationError):
        backend.write(user_id="", title="Title", description="desc", content="body")


def test_same_title_under_two_users_are_independent(backend, user_id, other_user_id):
    backend.write(user_id=user_id, title="Shared", description="mine", content="mine")
    backend.write(user_id=other_user_id, title="Shared", description="theirs", content="theirs")

    mine, _ = backend.read(user_id=user_id, titles=["Shared"])
    theirs, _ = backend.read(user_id=other_user_id, titles=["Shared"])

    assert mine[0].content == "mine"
    assert theirs[0].content == "theirs"


def test_read_preserves_request_order_and_reports_missing(seeded_backend, user_id):
    found, missing = seeded_backend.read(user_id=user_id, titles=["API conventions", "nope", "Deploy runbook"])

    assert [note.title for note in found] == ["API conventions", "Deploy runbook"]
    assert missing == ["nope"]


def test_read_deduplicates_titles(seeded_backend, user_id):
    found, missing = seeded_backend.read(user_id=user_id, titles=["Deploy runbook", " Deploy runbook "])

    assert len(found) == 1
    assert missing == []


def test_read_never_crosses_users(seeded_backend, other_user_id):
    found, missing = seeded_backend.read(user_id=other_user_id, titles=["Deploy runbook"])

    assert found == []
    assert missing == ["Deploy runbook"]


def test_read_requires_at_least_one_usable_title(seeded_backend, user_id):
    with pytest.raises(NoteValidationError):
        seeded_backend.read(user_id=user_id, titles=["  "])


def test_delete_reports_deleted_and_not_found(seeded_backend, user_id):
    assert seeded_backend.delete(user_id=user_id, title="Deploy runbook") is NoteDeleteStatus.DELETED
    assert seeded_backend.delete(user_id=user_id, title="Deploy runbook") is NoteDeleteStatus.NOT_FOUND


def test_delete_cannot_touch_another_users_note(seeded_backend, user_id, other_user_id):
    status = seeded_backend.delete(user_id=other_user_id, title="Deploy runbook")

    assert status is NoteDeleteStatus.NOT_FOUND
    found, _ = seeded_backend.read(user_id=user_id, titles=["Deploy runbook"])
    assert len(found) == 1


def test_index_is_sorted_by_title_and_carries_no_content(seeded_backend, user_id):
    entries, truncated = seeded_backend.index(user_id=user_id)

    assert [entry.title for entry in entries] == ["API conventions", "Deploy runbook"]
    assert truncated is False
    assert all(isinstance(entry, NoteIndexEntry) for entry in entries)
    assert "content" not in NoteIndexEntry.model_fields


def test_index_can_order_by_recency(backend, user_id):
    backend.write(user_id=user_id, title="alpha", description="d", content="c")
    backend.write(user_id=user_id, title="beta", description="d", content="c")
    backend.write(user_id=user_id, title="alpha", description="d", content="updated")

    entries, _ = backend.index(user_id=user_id, order_by="updated_at")

    assert [entry.title for entry in entries] == ["alpha", "beta"]


def test_index_honours_the_cap_and_reports_truncation(user_id):
    backend = InMemoryNotesBackend(max_index_entries=2)
    for i in range(5):
        backend.write(user_id=user_id, title=f"note-{i}", description="d", content="c")

    entries, truncated = backend.index(user_id=user_id)

    assert len(entries) == 2
    assert truncated is True


def test_index_limit_cannot_exceed_the_cap(user_id):
    backend = InMemoryNotesBackend(max_index_entries=2)
    for i in range(5):
        backend.write(user_id=user_id, title=f"note-{i}", description="d", content="c")

    entries, truncated = backend.index(user_id=user_id, limit=100)

    assert len(entries) == 2
    assert truncated is True


def test_index_on_an_empty_store(backend, user_id):
    assert backend.index(user_id=user_id) == ([], False)


def test_index_rejects_an_unknown_order(seeded_backend, user_id):
    with pytest.raises(NoteValidationError):
        seeded_backend.index(user_id=user_id, order_by="content")


def test_clear_user_only_touches_its_own_user(seeded_backend, user_id, other_user_id):
    seeded_backend.write(user_id=other_user_id, title="Theirs", description="d", content="c")

    deleted = seeded_backend.clear_user(user_id=user_id)

    assert deleted == 2
    assert seeded_backend.list_all(user_id=user_id) == []
    assert len(seeded_backend.list_all(user_id=other_user_id)) == 1
