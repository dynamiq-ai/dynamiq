import threading

from dynamiq.memory.notes import Note, NoteIndexEntry
from dynamiq.memory.notes.backends import InMemoryNotesBackend


def test_concurrent_writes_to_one_title_yield_a_single_note(backend, user_id):
    """`upsert` is contractually atomic — concurrent writers must not duplicate a title
    or reset its `created_at`."""
    barrier = threading.Barrier(8)
    results: list = []
    lock = threading.Lock()

    def write(i: int) -> None:
        barrier.wait()
        note, outcome = backend.write(user_id=user_id, title="Shared", description=f"d{i}", content=f"c{i}")
        with lock:
            results.append((note, outcome))

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(backend.list_all(user_id=user_id)) == 1
    assert sum(1 for _, outcome in results if outcome.value == "created") == 1
    created_ats = {note.created_at for note, _ in results}
    assert len(created_ats) == 1


def test_list_index_returns_entries_without_content(seeded_backend, user_id):
    entries = seeded_backend.list_index(user_id=user_id, limit=10, order_by="title")

    assert entries
    assert all(isinstance(entry, NoteIndexEntry) for entry in entries)
    assert all(not hasattr(entry, "content") for entry in entries)


def test_list_by_user_returns_full_notes(seeded_backend, user_id):
    notes = seeded_backend.list_by_user(user_id=user_id, limit=10)

    assert all(isinstance(note, Note) for note in notes)
    assert all(note.content for note in notes)


def test_delete_user_on_an_unknown_user_returns_zero(backend):
    assert backend.delete_user(user_id="nobody") == 0


def test_zero_limits_short_circuit(seeded_backend, user_id):
    assert seeded_backend.list_index(user_id=user_id, limit=0, order_by="title") == []
    assert seeded_backend.list_by_user(user_id=user_id, limit=0) == []
    assert seeded_backend.list_all(user_id=user_id, limit=0) == []


def test_backend_type_resolves_against_the_backends_package():
    assert InMemoryNotesBackend().type == "dynamiq.memory.notes.backends.InMemoryNotesBackend"


def test_to_dict_is_serializable(backend):
    data = backend.to_dict()

    assert data["type"] == "dynamiq.memory.notes.backends.InMemoryNotesBackend"
    assert data["max_index_entries"] == 200
